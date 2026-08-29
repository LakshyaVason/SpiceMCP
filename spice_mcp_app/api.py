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
import threading
from pathlib import Path
from typing import Any

from .config import Config, ConfigError, load_config
from .llm import (
    AgentResult,
    LLMError,
    TamuClient,
    mcp_tools_to_openai,
    run_agent_turn,
)
from .mcp_client import MCPClientError, SpiceMCP, find_circuits
from .session import Session, Turn

log = logging.getLogger(__name__)

PATCH_TOOL = "patch_component_value"


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
        self._client: TamuClient | None = None
        self._session: Session | None = None
        self._tools: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._approved: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()  # one agent turn at a time
        self.window: Any = None
        # Set by __main__ when --folder was passed; the UI asks for it on load.
        self.initial_folder: str | None = None

    def get_initial_folder(self) -> dict[str, Any]:
        return _ok(folder=self.initial_folder)

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
            self._tools = mcp_tools_to_openai(mcp_tools)

        if self._client is None:
            self._client = TamuClient(self._config)

        if self._session is None:
            self._session = Session(
                model=self._config.model, sessions_dir=self._config.sessions_dir
            )
            self._session.save()

        return _ok(
            model=self._config.model,
            tools=[t["function"]["name"] for t in self._tools],
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

        if self.window is None:
            return _err("No window is attached.")

        chosen = self.window.create_file_dialog(webview.FOLDER_DIALOG)
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

        # Tell the model which file is open, with its absolute path. Without this it has
        # no way to know and will guess a relative name, which the server then cannot
        # find - the tools take a path argument, not an implicit "current circuit".
        note = (
            f"[The user has opened this circuit: {target}\n"
            f"Use that exact absolute path in tool calls. "
            f"Static checks already run: {checks.get('summary', 'n/a')}]"
        )
        self._history.append({"role": "user", "content": note})
        self._session.add_turn(Turn(role="user", text=note))

        return _ok(circuit_file=str(target), name=target.name, checks=checks)

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
                tools=self._tools,
                tool_executor=self._tool_executor,
                history=self._history,
                on_progress=progress.append,
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
            f"{resolved.name}; the file has been written.]"
        )
        self._history.append({"role": "user", "content": note})
        self._session.add_turn(Turn(role="user", text=note))

        return _ok(patch=payload, summary=payload.get("summary"))

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
        if self.window is not None:
            import webview

            chosen = self.window.create_file_dialog(
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
