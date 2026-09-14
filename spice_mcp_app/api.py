"""The JS-facing API bridge: everything the UI can ask the Python side to do.

Kept separate from `__main__.py` so it can be exercised without opening a window - the
tests and `scripts/app_smoke.py` drive this class directly.

**This is where the approval gate lives.** `patch_component_value` defaults to a preview
on the server, but a model can set `apply=True` itself, so the executor here refuses any
apply the user has not explicitly approved through the UI and tells the model to present
its diff instead. The rule is "nothing touches disk before user approval", and it has to
be enforced somewhere the model cannot talk its way past.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from spice_mcp_server.ltspice import ltspice_is_running

from .compact import circuit_summary, compact_tool_result
from .config import Config, ConfigError, load_config
from .llm import (
    AgentResult,
    GatewayClient,
    LLMError,
    append_user_note,
    mcp_tools_to_anthropic,
    run_agent_turn,
)
from .mcp_client import MCPClientError, SpiceMCP, find_circuits
from .session import Session, Turn

log = logging.getLogger(__name__)

PATCH_TOOL = "patch_component_value"

# How many static findings are preloaded into the model's context when a circuit is
# selected. A circuit with more problems than this has one underlying problem; the list is
# the model's starting point, not the whole record, and the full result stays in the UI and
# the session log either way.
MAX_PRELOADED_FINDINGS = 12

# Tools withheld from the *model's* catalogue during ordinary diagnosis. The seven schemas
# cost 6588 chars of description and JSON schema on every round - in prompted mode all of it
# sits in the system prompt - and these two are 1342 of it. Neither is reachable from a
# diagnosis or from the Apply flow: `diff_netlist` compares two files the user would have to
# name, and `export_netlist` writes a `.net` nothing here asks for.
#
# `patch_component_value` is deliberately **not** in this list even though it is the largest
# single schema. Without a model call there is no `pending_patch`, so no diff panel and no
# Apply button, and the whole approval flow dies. Saving tokens by removing the feature is
# not saving tokens.
WITHHELD_FROM_DIAGNOSIS = ("diff_netlist", "export_netlist")

# Words that put them back for one turn. The rule can only ever *widen* the catalogue, which
# is the only shape that is safe: a rule that withholds on a guess can block a legitimate
# workflow, while a rule that offers too much only costs the tokens it was meant to save.
WIDEN_KEYWORDS = ("diff", "compar", "export", "before and after", ".net")


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


class Api:
    """Bridge object exposed to JavaScript via pywebview's `js_api`."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config
        self._config_error: str | None = None
        if self._config is None:
            try:
                self._config = load_config()
            except ConfigError as exc:
                self._config_error = str(exc)

        self._mcp: SpiceMCP | None = None
        self._client: GatewayClient | None = None
        self._session: Session | None = None
        self._tools: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._approved: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()  # one agent turn at a time
        # The leading underscore is load-bearing, not style. pywebview builds the JS bridge
        # by recursively walking every *public* attribute of this object
        # (webview/util.py:180-211) and it skips names starting with "_". Public, this held a
        # webview.Window, whose `native` is the .NET WinForms Form - unguarded, unlike
        # pywebview's own DOM/EventContainer/state, which set `_serializable = False`. The
        # walk then descended Form.AccessibilityObject.Bounds.Empty.Empty... forever and
        # never returned, so `finish.js` never fired `pywebviewready` and the window opened
        # dead. See _attach_window.
        self._window: Any = None
        # Set by __main__ when --folder/--file was passed; the UI asks for these on load.
        self.initial_folder: str | None = None
        self.initial_circuit: str | None = None
        # A non-fatal problem from before the window existed, e.g. the Explorer launcher
        # failing to start LTspice. Shown in the banner rather than lost to a dead console.
        self.startup_note: str | None = None
        # True when the launcher opened `initial_circuit` in the GUI itself moments ago.
        self.opened_in_ltspice = False
        self._warned_about_ltspice = False

    def _attach_window(self, window: Any) -> None:
        """Hand this bridge the pywebview Window, for the two file dialogs.

        Called by `__main__` once the window exists - it cannot be a constructor argument
        because `create_window` needs the `js_api` object first.

        Underscore-named on both sides on purpose. pywebview walks the public attributes of
        this class to build the JS API, recursing into any non-callable that has a
        `__module__`, and a `Window` leads it straight into the native WinForms/WebView2
        object graph, where it never comes back. Keep every *public* attribute of `Api` a
        str, bool or None; anything richer belongs behind an underscore.
        """
        self._window = window

    def get_initial_folder(self) -> dict[str, Any]:
        """Everything the UI needs on load, in one round trip.

        Extending this payload rather than adding a method keeps the bridge surface - and
        the contract asserted in tests/test_app_wiring.py - unchanged.
        """
        return _ok(
            folder=self.initial_folder,
            circuit=self.initial_circuit,
            note=self.startup_note,
        )

    # --- startup ---------------------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Spawn the MCP server and open a session. Called once as the UI loads."""
        if self._config_error:
            return _err(self._config_error)
        assert self._config is not None

        if self._mcp is None:
            try:
                self._mcp = SpiceMCP()
                mcp_tools = self._mcp.start()
            except MCPClientError as exc:
                self._mcp = None
                return _err(str(exc))
            self._tools = mcp_tools_to_anthropic(mcp_tools)

        if self._client is None:
            self._client = GatewayClient(self._config)

        if self._session is None:
            self._session = Session(
                model=self._config.model, sessions_dir=self._config.sessions_dir
            )
            self._session.save()

        return _ok(
            model=self._config.model,
            tool_mode=self._config.tool_mode,
            tools=[t["name"] for t in self._tools],
            # What a diagnosis turn actually offers the model. Reported separately so the
            # header can say "7 tools (5 offered)" rather than a number that is true of the
            # server and false of the request - the difference is exactly what this saves.
            tools_offered=[t["name"] for t in self._model_tools("")],
            session_id=self._session.session_id,
            session_path=str(self._session.path),
            config=self._config.redacted(),
        )

    def shutdown(self) -> dict[str, Any]:
        if self._mcp is not None:
            self._mcp.close()
            self._mcp = None
        return _ok()

    # --- circuit selection -----------------------------------------------------------

    def pick_folder(self) -> dict[str, Any]:
        """Open the OS folder picker and list the circuits inside."""
        import webview

        if self._window is None:
            return _err("No window is attached.")

        chosen = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not chosen:
            return _ok(cancelled=True)

        folder = chosen[0] if isinstance(chosen, (list, tuple)) else chosen
        return self.list_folder(str(folder))

    def list_folder(self, folder: str) -> dict[str, Any]:
        try:
            circuits = find_circuits(folder)
        except MCPClientError as exc:
            return _err(str(exc))

        # A .net beside a .asc is usually an ExpressPCB export, so flag it in the UI
        # rather than letting the user pick the file that cannot be parsed.
        schematics = {Path(c["path"]).stem for c in circuits if c["suffix"] == ".asc"}
        for circuit in circuits:
            circuit["shadowed"] = (
                circuit["suffix"] != ".asc" and Path(circuit["path"]).stem in schematics
            )

        return _ok(folder=str(folder), circuits=circuits)

    def select_circuit(self, path: str) -> dict[str, Any]:
        """Set the session's circuit and run the cheap static pass on it."""
        if self._session is None or self._mcp is None:
            return _err("The session is not started.")

        target = Path(path)
        if not target.is_file():
            return _err(f"No such file: {target}")

        self._session.set_circuit_file(str(target))
        try:
            raw = self._mcp.call_tool("check_netlist_static", {"path": str(target)})
        except MCPClientError as exc:
            return _err(str(exc))

        try:
            checks = json.loads(raw)
        except json.JSONDecodeError:
            checks = {"summary": raw, "findings": [], "ok": False}

        note = self._selection_note(target, checks, topology=self._preload_topology(target))
        # Folded into the pending user turn rather than appended as its own: the Messages
        # API rejects two user turns in a row, and the next question adds one.
        append_user_note(self._history, note)
        self._session.add_turn(Turn(role="user", text=note))

        return _ok(
            circuit_file=str(target),
            name=target.name,
            checks=checks,
            warning=self._ltspice_open_warning(target),
        )

    def _preload_topology(self, target: Path) -> str | None:
        """The circuit summary for the selection note, or `None` if it could not be read.

        The redundant `read_netlist` round in the baseline was not the model being
        wasteful: the static *findings* were preloaded but the topology was not, so it
        genuinely did not have the components it needed to name a fix. Telling it not to
        call the tool without first supplying the data would have made it answer from
        information it did not have.

        Failure here is not an error for the caller. An ExpressPCB `.net`, a missing
        LTspice, an unparseable file - the selection still succeeds, the summary is simply
        absent, and `_selection_note` then does not claim the topology is loaded. That
        pairing is the safety property: the model is never told it has data it does not.

        `include_raw_text=False` because the projection drops `raw_text` anyway, so
        carrying it over the MCP pipe would be pure overhead.
        """
        if self._mcp is None:
            return None
        try:
            raw = self._mcp.call_tool(
                "read_netlist", {"path": str(target), "include_raw_text": False}
            )
            payload = json.loads(raw)
        except (MCPClientError, json.JSONDecodeError) as exc:
            log.info("no topology preload for %s: %s", target, exc)
            return None
        if not isinstance(payload, dict):
            return None
        return circuit_summary(payload)

    @staticmethod
    def _selection_note(
        target: Path, checks: dict[str, Any], *, topology: str | None = None
    ) -> str:
        """The context injected when a circuit is selected.

        Three jobs. First, the absolute path: the tools take a path argument rather than
        having an implicit "current circuit", and without being told, the model guesses a
        relative name the server then cannot find.

        Second, the static findings *themselves*. They are already computed and already
        paid for by the time this runs, so summarising them as "Found 2 errors" was the
        worst of both worlds - it told the model something was wrong without saying what,
        which is an invitation to re-run the same check to find out. Handing over the
        findings both saves that round trip and gives the model something to reason from.

        Third, the topology, by the same argument one step further: the findings say *what*
        is wrong, and the components say what to do about it. With both here a simple fault
        needs no tool call at all.

        **The list of what has already run is derived from what actually succeeded**, never
        written as a fixed string. If the topology could not be read, the note says so and
        points at `read_netlist`. A model told it has data it does not have would answer
        from nothing, which is a worse failure than any number of redundant tool calls.
        """
        findings = checks.get("findings") or []
        already_ran = ["check_netlist_static"] + (["read_netlist"] if topology else [])
        lines = [
            f"[The user has opened this circuit: {target}",
            "Use that exact absolute path in tool calls.",
            "Already run for you, with the real output below - this is tool output, not a "
            "guess, so do not call these again unless the file changes: "
            + ", ".join(already_ran)
            + ".",
        ]
        if not topology:
            lines.append(
                "The topology was NOT read - call read_netlist when you need components, "
                "nodes or values."
            )
        lines.append(f"static check: {checks.get('summary', 'n/a')}")
        for finding in findings[:MAX_PRELOADED_FINDINGS]:
            if not isinstance(finding, dict):
                continue
            where = ", ".join(
                [*(finding.get("refs") or []), *(finding.get("nets") or [])]
            )
            line = (
                f"- {finding.get('severity', '?')}/{finding.get('check', '?')}"
                f"{f' [{where}]' if where else ''}: {finding.get('message', '')}"
            )
            if finding.get("suggestion"):
                line += f" -> {finding['suggestion']}"
            lines.append(line)
        if len(findings) > MAX_PRELOADED_FINDINGS:
            lines.append(
                f"- ...and {len(findings) - MAX_PRELOADED_FINDINGS} more; "
                f"re-run check_netlist_static to see them all."
            )
        if topology:
            lines.append(topology)
        return "\n".join(lines) + "]"

    def _ltspice_open_warning(self, target: Path) -> str | None:
        """Warn once that what we read from disk may not be what is on screen.

        Everything here reads the .asc from the filesystem, so unsaved GUI edits are
        invisible to it and every answer would be about a stale circuit. Once per session,
        not once per selection: a warning that repeats on every click is a warning people
        stop reading.

        Suppressed entirely on the Explorer path's first look, where the launcher opened the
        GUI on this exact file microseconds ago - disk and screen are identical, so warning
        there would fire on every single launch at the moment it is least true. The project's
        own testing philosophy is the argument: a check that cries wolf trains people to
        ignore it. Re-select the file later and the warning is live again.
        """
        if self.opened_in_ltspice and self._same_file(target, self.initial_circuit):
            self.opened_in_ltspice = False
            return None
        if self._warned_about_ltspice or not ltspice_is_running():
            return None
        self._warned_about_ltspice = True
        return (
            "LTspice is open. Everything here is read from the file on disk, so save in "
            "LTspice (Ctrl+S) before asking - unsaved edits are invisible to this client."
        )

    @staticmethod
    def _same_file(left: Path, right: str | None) -> bool:
        """Compare two paths the way Windows does: case-insensitively."""
        if not right:
            return False
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
            os.path.abspath(right)
        )

    # --- chat ------------------------------------------------------------------------

    def _tool_executor(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._mcp is not None

        if name == PATCH_TOOL and arguments.get("apply"):
            key = (
                str(Path(str(arguments.get("asc_path", ""))).resolve()),
                str(arguments.get("ref", "")).upper(),
                str(arguments.get("new_value", "")).strip(),
            )
            if key not in self._approved:
                log.info("blocked an unapproved apply of %s", key)
                return (
                    "REFUSED: writing to the schematic needs the user's approval first, "
                    "and they have not given it for this change. Call this tool again "
                    "with apply=false, show them the diff it returns, and let them press "
                    "Apply. Do not retry with apply=true."
                )

        return self._mcp.call_tool(name, arguments)

    def _model_tools(self, text: str) -> list[dict[str, Any]]:
        """The catalogue for this turn: all seven, minus the two nothing here reaches.

        `self._tools` stays the full set - `start()` reports it, and the server's seven are
        pinned by `tests/test_write_conflict.py`. This is only about what each *request*
        pays for.

        The keyword trigger widens and never narrows. If the user asks about a diff, a
        comparison or an export, they get the whole catalogue for that turn; asking about
        anything else cannot take a tool away that the withheld list did not already name.
        """
        if not self._tools:
            return self._tools
        lowered = (text or "").lower()
        if any(word in lowered for word in WIDEN_KEYWORDS):
            return list(self._tools)
        return [t for t in self._tools if t["name"] not in WITHHELD_FROM_DIAGNOSIS]

    def send_message(self, text: str) -> dict[str, Any]:
        """Run one user turn to completion and return the assistant's answer."""
        if self._session is None or self._client is None:
            return _err("The session is not started.")
        if not text or not text.strip():
            return _err("Type a message first.")

        if not self._lock.acquire(blocking=False):
            return _err("A turn is already running.")
        try:
            progress: list[str] = []
            result: AgentResult = run_agent_turn(
                self._client,
                self._session,
                text.strip(),
                tools=self._model_tools(text),
                tool_executor=self._tool_executor,
                history=self._history,
                on_progress=progress.append,
                tool_mode=self._config.tool_mode,
                # Compact for the model, complete for the record: the projection is what
                # goes into the request, `record.result` below is still the full text the
                # session log, the UI preview and `_pending_patch` read.
                compactor=compact_tool_result,
            )
        except LLMError as exc:
            return _err(str(exc))
        finally:
            self._lock.release()

        return _ok(
            text=result.text,
            rounds=result.rounds,
            tool_calls=[self._present_tool_call(c) for c in result.tool_calls],
            usage={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            totals=self.totals()["totals"],
            pending_patch=self._pending_patch(result),
        )

    def _present_tool_call(self, record: Any) -> dict[str, Any]:
        payload = record.to_dict()
        # Truncated for display only; the session log keeps the full text.
        if len(payload["result"]) > 4000:
            payload["result"] = payload["result"][:4000] + "\n[truncated for display]"
        return payload

    def _pending_patch(self, result: AgentResult) -> dict[str, Any] | None:
        """The most recent unapplied patch preview, for the diff panel."""
        for record in reversed(result.tool_calls):
            if record.name != PATCH_TOOL or record.is_error:
                continue
            try:
                payload = json.loads(record.result)
            except json.JSONDecodeError:
                continue
            if payload.get("applied"):
                continue
            return {
                "asc_path": payload.get("asc_path"),
                "ref": payload.get("ref"),
                "old_value": payload.get("old_value"),
                "new_value": payload.get("new_value"),
                "line_no": payload.get("line_no"),
                "diff": payload.get("diff", ""),
                "encoding": payload.get("encoding"),
            }
        return None

    # --- approval and verification ---------------------------------------------------

    def apply_patch(self, asc_path: str, ref: str, new_value: str) -> dict[str, Any]:
        """Approve and write a proposed value change. Only the UI calls this.

        Recording the approval before the call is what lets `_tool_executor` tell an
        approved write apart from one the model decided to make on its own.
        """
        if self._mcp is None or self._session is None:
            return _err("The session is not started.")

        resolved = Path(asc_path)
        if not resolved.is_file():
            return _err(f"No such file: {resolved}")

        key = (str(resolved.resolve()), ref.upper(), new_value.strip())
        self._approved.add(key)
        try:
            raw = self._mcp.call_tool(
                PATCH_TOOL,
                {
                    "asc_path": str(resolved),
                    "ref": ref,
                    "new_value": new_value,
                    "apply": True,
                },
            )
        except MCPClientError as exc:
            return _err(str(exc))
        finally:
            # One approval, one write. A later apply has to be approved again.
            self._approved.discard(key)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return _err(raw)
        if not payload.get("applied"):
            return _err(payload.get("summary", raw))

        # Tell the model what the user did, so the conversation stays truthful about
        # the state of the file and it can verify against the real schematic.
        note = (
            f"[The user approved your fix. {ref} is now {new_value} in "
            f"{resolved.name}; the file has been written.\n"
            # The file has changed, so the preloaded static check and topology above are now
            # out of date. Without this, the policy telling the model not to re-run what has
            # already run would suppress exactly the verification that matters most.
            "The file has changed, so the preloaded static check and topology above are now "
            "stale - re-read what you need to verify the fix.]"
        )
        append_user_note(self._history, note)
        self._session.add_turn(Turn(role="user", text=note))

        # The write is never blocked on LTspice being open - the user asked for the fix and
        # the file is theirs. But LTspice holds its own copy of the schematic and will
        # write it back over ours on save, so silently succeeding here would hand them a
        # fix that quietly disappears later.
        warning = None
        if ltspice_is_running():
            warning = (
                f"LTspice is running and will not notice this edit. If {resolved.name} is "
                f"open there, use File ▸ Revert to reload it - saving from LTspice "
                f"without reverting will overwrite this fix."
            )

        return _ok(patch=payload, summary=payload.get("summary"), warning=warning)

    def resimulate(self) -> dict[str, Any]:
        """Re-run the simulation on the session's circuit to confirm a fix."""
        if self._mcp is None or self._session is None:
            return _err("The session is not started.")
        if not self._session.circuit_file:
            return _err("No circuit is selected.")

        try:
            raw = self._mcp.call_tool(
                "run_simulation", {"path": self._session.circuit_file}
            )
            checks = self._mcp.call_tool(
                "check_netlist_static", {"path": self._session.circuit_file}
            )
        except MCPClientError as exc:
            return _err(str(exc))

        try:
            simulation = json.loads(raw)
        except json.JSONDecodeError:
            return _err(raw)
        try:
            static = json.loads(checks)
        except json.JSONDecodeError:
            static = None

        return _ok(
            succeeded=simulation.get("succeeded"),
            summary=simulation.get("summary"),
            simulation=simulation,
            checks=static,
        )

    # --- session log -----------------------------------------------------------------

    def totals(self) -> dict[str, Any]:
        if self._session is None:
            return _ok(totals={"input_tokens": 0, "output_tokens": 0, "turns": 0})
        return _ok(
            totals={
                "input_tokens": self._session.total_input_tokens,
                "output_tokens": self._session.total_output_tokens,
                "turns": len(self._session.turns),
            },
            session_path=str(self._session.path),
            resolved=self._session.resolved,
        )

    def mark_resolved(self, resolved: bool = True) -> dict[str, Any]:
        if self._session is None:
            return _err("The session is not started.")
        self._session.mark_resolved(bool(resolved))
        return _ok(resolved=self._session.resolved)

    def export_session(self) -> dict[str, Any]:
        """Save a copy of the session log wherever the user wants it."""
        if self._session is None:
            return _err("The session is not started.")

        destination: str | None = None
        if self._window is not None:
            import webview

            chosen = self._window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=f"spice-mcp-session-{self._session.session_id[:8]}.json",
                file_types=("JSON (*.json)",),
            )
            if not chosen:
                return _ok(cancelled=True)
            destination = chosen if isinstance(chosen, str) else chosen[0]

        if not destination:
            return _err("No destination chosen.")

        written = self._session.export_to(destination)
        return _ok(path=str(written))

    def reveal_session(self) -> dict[str, Any]:
        """Return the on-disk path of the live session log."""
        if self._session is None:
            return _err("The session is not started.")
        return _ok(path=str(self._session.path), data=self._session.to_dict())
