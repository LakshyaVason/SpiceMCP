"""Measure what one debugging turn costs: rounds, tool calls, tokens, answer length.

The instrument for the token/verbosity experiment. BEFORE and AFTER have to be produced
by the *same* code to be comparable, which is why this lands as the experiment branch's
first commit - before anything that changes behaviour - and why it drives the real
`spice_mcp_app.api.Api` rather than a parallel harness, exactly as `scripts/app_smoke.py`
does.

    .venv\\Scripts\\activate

    REM the professor's case: the whole point of the experiment
    python scripts/measure_turn.py --circuit RCLP.asc \
        --question "why is my circuit not getting any gain" \
        --expect R1 --expect NC_01 --label before

    REM free: no inference, just the size of the request that would be sent
    python scripts/measure_turn.py --circuit RCLP.asc \
        --question "why is my circuit not getting any gain" --count-only

Modes, and the difference matters:

  * default - runs the turn for real and **spends tokens**. Token counts come from the
    session log's own totals, so this cannot report a saving the external
    screenshot-vs-MCP comparison will not also see.
  * `--count-only` - calls `messages.count_tokens` instead. The gateway prices the input
    and runs no inference, so it is free. This is how a prompt, preload or tool-catalogue
    change gets attributed to the change that caused it without paying for a completion
    per iteration. It says nothing about output tokens or about which tools the model
    would have chosen.
  * `--breakdown` - free. Cumulative passes plus a per-segment estimate for the base
    system prompt, the protocol instructions, each tool schema, the selection note's
    preamble / findings / topology, the question and any prior turns. Its most important
    output is `unattributed`: the gap between the real reconstruction and the sum of the
    segments. **Estimates are `count_tokens` numbers on pieces cut out of context and do
    not sum to the billed total**; the gap is reported, never smoothed away.
  * `--history-probe` - **spends tokens** (one live turn). Prices the request after each
    thing a user actually does: select, re-select the same circuit, one turn, select
    again, apply the patch if one was proposed. A single-question row cannot see history
    accumulating, because a fresh process has exactly one selection note in it.

Three controls exist so a comparison is controlled rather than incidental:

  * `--dump-request DIR` writes every request the SDK really sent, paired with the
    `usage` of its own response. The seam is the Anthropic SDK's `messages` resource, not
    `GatewayClient.complete`, which writes its arguments inline in a closure and retries
    once when the gateway rejects `output_config`. Paths are redacted (`<CIRCUIT_DIR>`,
    `<HOME>`, `<REPO>`) and the run refuses to write a dump containing the gateway token.
  * `--effort {auto,light,medium,hard}` pins the reasoning effort. Without it the run
    silently inherits ambient `SPICE_MCP_EFFORT`, which contaminates every count. Effort
    changes output/thinking tokens, not input ones - report the two separately.
  * `--force-tools all|none|<names>` makes the catalogue a controlled variable instead of
    keyword luck. It can only change what a request *offers*; the approval gate in
    `Api._tool_executor` is untouched and `Api._tools` stays all seven.

**Billed and estimated are not the same number.** `usage` from a real `create` is what the
gateway charges; everything from `count_tokens` is an estimate of a request nobody paid
for. The row keys keep them apart and so should any write-up.

Always runs against a **copy** of the circuit in a temp directory. Hard constraint 2 is
not conditional on intent: LTspice writes `-netlist` output next to its input, so even a
read-only question would litter the user's folder.

Exit code 0 means every `--expect` string was found in the answer. A smaller token count
with a worse answer is a failed experiment, so the two are reported together and the
exit code follows correctness, not cost.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spice_mcp_app.api import Api  # noqa: E402
from spice_mcp_app.config import (  # noqa: E402
    REPO_ROOT,
    TOKEN_ENV_VAR,
    TOOL_MODE_PROMPTED_JSON,
    TOOL_MODES,
    load_config,
)
from spice_mcp_app.effort import UI_MODE_MAP, classify_effort  # noqa: E402
from spice_mcp_app.llm import (  # noqa: E402
    SYSTEM_PROMPT,
    append_user_note,
    prompted_protocol_prompt,
    prompted_system_prompt,
)

# What `--force-tools all` and `--force-tools none` mean, spelled out so the parser can
# reject a typo rather than silently selecting nothing.
FORCE_ALL = "all"
FORCE_NONE = "none"

# Used wherever a pass needs "no real message" but the API still wants one. See `breakdown`.
PROBE_MSG: list[dict[str, Any]] = [{"role": "user", "content": "."}]

BENCH_DIR = REPO_ROOT / "bench"


def normalized(text: str) -> str:
    """Fold whitespace and case so `100n`, `100 nF` and `100nF` all match.

    Lifted from `app_smoke.py`'s stage-3 check, for the same reason: an expectation that
    fails on a space is an expectation that trains you to ignore the harness.
    """
    return text.replace(" ", "").lower()


def measure(
    api: Api,
    question: str,
    expectations: list[str],
) -> dict[str, Any]:
    """Run one question and return its metric row.

    Tokens are read as a **difference in the session's own totals** rather than summed
    from the turn's `usage`. Same numbers, but sourced from the file the external cost
    comparison reads, so the harness cannot flatter a change the comparison would not see.
    """
    before = api.totals()["totals"]
    turn = api.send_message(question)
    after = api.totals()["totals"]

    if not turn.get("ok"):
        return {"question": question, "ok": False, "error": turn.get("error", "")}

    answer = turn["text"]
    calls = turn["tool_calls"]
    haystack = normalized(answer)
    missing = [e for e in expectations if normalized(e) not in haystack]

    return {
        "question": question,
        "ok": True,
        "rounds": turn["rounds"],
        "tool_calls": [c["name"] for c in calls],
        "tool_call_count": len(calls),
        # How much text each result put in front of the model. The point of comparison
        # for the compaction work, and invisible in the token totals alone.
        "tool_result_chars": [len(c["result"]) for c in calls],
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "total_tokens": (after["input_tokens"] - before["input_tokens"])
        + (after["output_tokens"] - before["output_tokens"]),
        "answer_words": len(answer.split()),
        "answer_chars": len(answer),
        "correct": not missing,
        "missing": missing,
        "answer": answer,
    }


# --- capturing the request that actually went on the wire -------------------------------


def _jsonable(value: Any) -> Any:
    """SDK sentinels and pydantic models into something `json.dumps` accepts.

    Builds new containers as it goes, so the result is already independent of the caller's
    objects - which matters because `messages` is the app's live, mutating history list.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if value is anthropic.NOT_GIVEN or type(value).__name__ == "NotGiven":
        return "<NOT_GIVEN>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            return _jsonable(dumper())
        except Exception:  # noqa: BLE001 - a dump is not worth failing a measurement over
            pass
    # The SDK's `Usage` is a pydantic model and takes the branch above, but a `usage` that is
    # ever anything else must still land as readable fields rather than as a `repr` string -
    # a billed token count is the one number in this file that has to be machine-readable.
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and attrs:
        return {k: _jsonable(v) for k, v in attrs.items() if not k.startswith("_")}
    return repr(value)


def _redact(value: Any, rules: list[tuple[str, str]]) -> Any:
    """Replace sensitive substrings throughout a JSON-able structure.

    Applied to the Python objects *before* `json.dumps`, not to the serialized text: a
    Windows path comes out of `dumps` escaped as `C:\\Users\\...`, so a needle built from
    `str(Path)` would no longer match it.
    """
    if isinstance(value, str):
        out = value
        for needle, replacement in rules:
            if needle:
                out = out.replace(needle, replacement)
        return out
    if isinstance(value, dict):
        return {k: _redact(v, rules) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, rules) for v in value]
    return value


def _message_shape(messages: list[Any]) -> list[dict[str, Any]]:
    """Role and size of every message, which is how duplication becomes visible.

    The same topology block appearing in two turns is obvious here and invisible in a
    single total.
    """
    shape: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            blocks, chars = 1, len(content)
            kinds = ["text"]
        elif isinstance(content, list):
            blocks = len(content)
            chars = len(json.dumps(content, ensure_ascii=False, default=str))
            kinds = [
                str(b.get("type", "?")) if isinstance(b, dict) else "?" for b in content
            ]
        else:
            blocks, chars, kinds = 0, 0, []
        shape.append(
            {"role": message.get("role"), "blocks": blocks, "chars": chars, "types": kinds}
        )
    return shape


class RequestCapture:
    """Record every request the SDK actually sends, paired with its billed usage.

    The seam is the Anthropic SDK's `messages` resource, not `GatewayClient.complete`.
    `complete` writes its arguments inline in a closure (`llm.py:532-541`), so there is no
    kwargs dict to read there; it resolves `output_config` only inside that closure; and it
    retries once when the gateway rejects the effort parameter (`llm.py:548`). Wrapping the
    SDK sees the resolved parameters and counts that retry as the separate request it is.

    Pairing each payload with the `usage` of its own response is the point of the whole
    exercise: it is the only way to say "this exact text cost N billed input tokens".
    """

    def __init__(self, api: Api, outdir: Path, rules: list[tuple[str, str]]) -> None:
        assert api._client is not None
        self._messages = api._client._client.messages
        self._outdir = outdir
        self._rules = rules
        self._original_create = self._messages.create
        self._original_count = self._messages.count_tokens
        self.events: list[dict[str, Any]] = []

    def install(self) -> None:
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._messages.create = self._wrap("create", self._original_create)
        self._messages.count_tokens = self._wrap("count_tokens", self._original_count)

    def restore(self) -> None:
        self._messages.create = self._original_create
        self._messages.count_tokens = self._original_count

    def _wrap(self, kind: str, original: Any) -> Any:
        def wrapper(**kwargs: Any) -> Any:
            response = original(**kwargs)
            try:
                self._record(kind, kwargs, response)
            except Exception as exc:  # noqa: BLE001 - never fail a run over the instrument
                print(f"  [capture] {kind} not recorded: {exc}", file=sys.stderr)
            return response

        return wrapper

    def _record(self, kind: str, kwargs: dict[str, Any], response: Any) -> None:
        payload = _jsonable(kwargs)
        system = payload.get("system")
        tools = payload.get("tools")
        messages = payload.get("messages") or []
        # `messages.create` carries a `usage` object; `messages.count_tokens` answers with a
        # bare `input_tokens`. Normalising here keeps a dump comparable across both, and
        # keeps the *source* of every number visible in `kind`.
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_out: Any = _jsonable(usage)
        elif getattr(response, "input_tokens", None) is not None:
            usage_out = {"input_tokens": int(response.input_tokens)}
        else:
            usage_out = None

        record = {
            "index": len(self.events) + 1,
            "kind": kind,
            "usage": usage_out,
            "char_sizes": {
                "system": len(system) if isinstance(system, str) else 0,
                "tools": len(json.dumps(tools, ensure_ascii=False, default=str))
                if isinstance(tools, list)
                else 0,
                "messages": len(json.dumps(messages, ensure_ascii=False, default=str)),
            },
            "tool_names": [
                t.get("name") for t in tools if isinstance(t, dict)
            ]
            if isinstance(tools, list)
            else [],
            "message_shape": _message_shape(messages),
            "request": payload,
        }

        record = _redact(record, self._rules)
        text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"

        # Belt and braces. The token lives inside the `Anthropic` instance and never appears
        # in these kwargs, so this should be unreachable - which is exactly why it is cheap
        # to assert rather than to reason about.
        token = (os.environ.get(TOKEN_ENV_VAR) or "").strip()
        if token and token in text:
            raise RuntimeError("refusing to write a dump that contains the gateway token")

        self.events.append(record)
        (self._outdir / f"{record['index']:02d}-{kind}.json").write_text(
            text, encoding="utf-8"
        )


def redaction_rules(workdir: Path) -> list[tuple[str, str]]:
    """Longest first: the measurement temp dir usually sits *inside* the home directory,
    so replacing the home path first would leave a half-redacted path behind."""
    rules = [(str(workdir), "<CIRCUIT_DIR>")]
    try:
        rules.append((str(Path.home()), "<HOME>"))
    except (OSError, RuntimeError):
        pass
    rules.append((str(REPO_ROOT), "<REPO>"))
    return sorted(rules, key=lambda rule: -len(rule[0]))


# --- reconstructing the request, one place only -----------------------------------------


def request_triple(
    api: Api, question: str, tool_mode: str
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """The exact `(messages, system, tools)` the loop would send, plus the catalogue.

    One implementation, shared by `count_only` and `breakdown`. They had a copy each, which
    is how `breakdown`'s "verification" pass ended up comparing a value with itself.

    Deep, not shallow: `append_user_note` folds into a *list* content block in place when the
    pending turn has one, which on a shallow copy would edit the live history and make the
    app's next real turn carry the measurement's question.
    """
    history = copy.deepcopy(api._history)
    append_user_note(history, question)

    # The catalogue the *request* carries, not the seven the server exposes: `_model_tools`
    # uses tool_policy to select based on preload state.
    catalogue = api._model_tools(question)
    if tool_mode == TOOL_MODE_PROMPTED_JSON:
        return history, prompted_system_prompt(catalogue), None, catalogue
    return history, SYSTEM_PROMPT, catalogue, catalogue


def resolve_forced_tools(api: Api, spec: str) -> list[dict[str, Any]] | None:
    """Turn a `--force-tools` spec into a catalogue, or print why it cannot be one.

    Making the catalogue a controlled variable is the point: keyword matching in
    `tool_policy` decides it otherwise, so a 5-vs-2-vs-0 comparison would be measuring the
    question's wording as much as the change under test.

    This narrows what a request *offers*. It cannot widen the server's seven tools, and it
    does not touch `Api._tool_executor`, which is the only thing that grants a write.
    """
    available = {t["name"]: t for t in api._tools}
    wanted = spec.strip().lower()

    if wanted == FORCE_ALL:
        return list(api._tools)
    if wanted == FORCE_NONE:
        return []

    names = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        print(
            f"--force-tools: no such tool(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}",
            file=sys.stderr,
        )
        return None
    # Server order, not command-line order, so the serialized catalogue is byte-comparable
    # with an unforced run that happened to select the same subset.
    return [t for t in api._tools if t["name"] in set(names)]


def resolved_effort(api: Api, question: str) -> str | None:
    """The effort `Api.send_message` would pick for this question (`api.py:449-454`).

    A free count that omitted it would price a different request than the live one, since
    `count_tokens` sends `output_config` too (`llm.py:614`).
    """
    if api._effort_ui_mode == "auto":
        return classify_effort(question, api._preload_state)
    return UI_MODE_MAP.get(api._effort_ui_mode)


def count_only(api: Api, question: str, tool_mode: str) -> dict[str, Any]:
    """Size the request that *would* be sent, without paying for a completion.

    Reaches into `api._history` and `api._tools` deliberately: the number is only worth
    having if it is the exact `system` + `messages` + `tools` the loop would send, and
    reconstructing that from the public surface would be a second implementation to keep
    in step. A measurement script may know more than the UI does.

    The effort is resolved the way `send_message` resolves it, because `count_tokens`
    sends `output_config` too (`llm.py:614`) - a count that omitted it would be pricing a
    request the app never makes.
    """
    assert api._client is not None
    history, system, tools, catalogue = request_triple(api, question, tool_mode)
    effort = resolved_effort(api, question)

    return {
        "question": question,
        "ok": True,
        "count_only": True,
        "tool_mode": tool_mode,
        "effort": effort,
        "input_tokens": api._client.count_tokens(
            history, tools=tools, system=system, effort=effort
        ),
        "system_chars": len(system),
        "history_messages": len(history),
        "history_chars": sum(s["chars"] for s in _message_shape(history)),
        "message_shape": _message_shape(history),
        "tool_catalogue_names": [t["name"] for t in catalogue],
        "tools_sent": len(catalogue),
    }


# --- segmenting the selection note ------------------------------------------------------

# The three parts of `Api._selection_note` (`api.py:295-356`), identified by the literal text
# that function writes. Matching on its own markers rather than on line offsets: the note is
# built by appending conditional lines, so its shape changes with what the preload achieved.
NOTE_OPENER = "[The user has opened this circuit:"
NOTE_STATIC_MARKER = "static check: "
NOTE_TOPOLOGY_MARKER = "Components:"


def _message_text(message: Any) -> str:
    """The text a message contributes, string content or text blocks alike."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def note_segments(note: str) -> dict[str, str]:
    """Split a selection note into preamble / static findings / topology.

    Returns only the parts that are present, so a preload that failed to read the topology
    yields no `note_topology` key rather than an empty one - the distinction the note itself
    is careful to make.
    """
    lines = note.split("\n")
    static_at = next(
        (i for i, line in enumerate(lines) if line.startswith(NOTE_STATIC_MARKER)), None
    )
    topo_at = next(
        (i for i, line in enumerate(lines) if line.startswith(NOTE_TOPOLOGY_MARKER)), None
    )

    out: dict[str, str] = {}
    if static_at is None and topo_at is None:
        return {"note_preamble": note}

    first_end = static_at if static_at is not None else topo_at
    out["note_preamble"] = "\n".join(lines[:first_end])
    if static_at is not None:
        static_end = topo_at if topo_at is not None else len(lines)
        out["note_static_findings"] = "\n".join(lines[static_at:static_end])
    if topo_at is not None:
        out["note_topology"] = "\n".join(lines[topo_at:])
    return {k: v for k, v in out.items() if v}


def breakdown(api: Api, question: str, tool_mode: str) -> dict[str, Any]:
    """Attribute the request's input tokens to the pieces that produced them.

    Two layers, kept separate on purpose.

    **Cumulative passes**, as before, so rows stay comparable with earlier bench files:
    system only, + tools, + user message, + history, and the real reconstruction. Two
    corrections from the previous version:

      * the first passes sent `messages=[]`, which the Messages API rejects; they now send a
        one-character `PROBE_MSG` and the scaffolding cost is reported as `probe_floor` so it
        can be subtracted. `empty_messages_probe` records what `messages=[]` actually does,
        since that was a prediction and not a measurement.
      * the fifth pass was `c_full = count(full_history, ...)` immediately after
        `c_with_history = count(full_history, ...)` - the same call with the same arguments.
        It can never fail, so it verified nothing. It now goes through `request_triple`, the
        same reconstruction `count_only` uses, which is what the docstring always claimed.

    **Fine segments**, which are what the brief actually asks for: base system prompt,
    protocol instructions, each tool's schema, the note's preamble / static findings /
    topology, the question, and any prior turns. Every one is prefixed `estimate_` because
    they are `count_tokens` numbers on pieces cut out of context; they do not sum to the
    billed total, and the difference is reported as `unattributed` rather than smoothed
    away. That gap is the thing this investigation is looking for.
    """
    assert api._client is not None
    full_history, full_system, full_tools, catalogue = request_triple(
        api, question, tool_mode
    )
    effort = resolved_effort(api, question)
    prompted = tool_mode == TOOL_MODE_PROMPTED_JSON

    def count(msgs, *, sys=SYSTEM_PROMPT, t=None):
        return api._client.count_tokens(msgs, tools=t, system=sys, effort=effort)

    # The irreducible cost of a request with nothing in it: message envelope plus a
    # one-character system prompt. Subtracted from every segment estimate below.
    probe_floor = count(PROBE_MSG, sys=".")

    def text_cost(text: str) -> int:
        """What one piece of text costs as a lone user message, floor removed."""
        return count([{"role": "user", "content": text}], sys=".") - probe_floor

    empty_probe: str
    try:
        empty_probe = f"accepted: {count([], sys=SYSTEM_PROMPT)} tokens"
    except Exception as exc:  # noqa: BLE001 - recording the failure *is* the measurement
        empty_probe = f"{type(exc).__name__}: {exc}"

    # --- cumulative passes
    c_system_only = count(PROBE_MSG, sys=SYSTEM_PROMPT)
    c_with_tools = count(PROBE_MSG, sys=full_system, t=full_tools)
    c_with_user = count(
        [{"role": "user", "content": question}], sys=full_system, t=full_tools
    )
    c_with_history = count(full_history, sys=full_system, t=full_tools)
    c_reconstruction = count_only(api, question, tool_mode)["input_tokens"]

    # --- fine segments
    segments: dict[str, int] = {"probe_floor": probe_floor}
    segments["base_system"] = c_system_only - probe_floor

    if prompted:
        header_only_system = SYSTEM_PROMPT + "\n\n" + prompted_protocol_prompt([])
        c_header = count(PROBE_MSG, sys=header_only_system)
        segments["protocol_header"] = c_header - c_system_only
        for tool in catalogue:
            one = SYSTEM_PROMPT + "\n\n" + prompted_protocol_prompt([tool])
            segments[f"schema_{tool['name']}"] = count(PROBE_MSG, sys=one) - c_header
    else:
        segments["protocol_header"] = 0
        for tool in catalogue:
            segments[f"schema_{tool['name']}"] = (
                count(PROBE_MSG, sys=SYSTEM_PROMPT, t=[tool]) - c_system_only
            )

    # The note is one message among possibly several; everything else in the history is
    # prior conversation and is counted whole.
    prior_turns = 0
    note_found = False
    for message in full_history:
        text = _message_text(message)
        if not note_found and NOTE_OPENER in text:
            note_found = True
            for name, piece in note_segments(text).items():
                segments[name] = segments.get(name, 0) + text_cost(piece)
            continue
        if text.strip() == question.strip():
            continue
        prior_turns += text_cost(text) if text else 0
    segments["history_prior_turns"] = prior_turns
    segments["question"] = text_cost(question)

    sum_of_segments = sum(segments.values())

    return {
        "question": question,
        "ok": True,
        "breakdown": True,
        "tool_mode": tool_mode,
        "effort": effort,
        "tool_catalogue_names": [t["name"] for t in catalogue],
        "tools_sent": len(catalogue),
        "system_chars": len(full_system),
        "note_found_in_history": note_found,
        "empty_messages_probe": empty_probe,
        "components": {
            "probe_floor": probe_floor,
            "system_only": c_system_only,
            "plus_tools": c_with_tools,
            "plus_user_msg": c_with_user,
            "plus_history": c_with_history,
            "reconstruction": c_reconstruction,
        },
        "deltas": {
            "tools": c_with_tools - c_system_only,
            "user_msg": c_with_user - c_with_tools,
            "history": c_with_history - c_with_user,
            # Non-zero means the cumulative passes and the real reconstruction disagree,
            # which is the check the old fifth pass was supposed to be.
            "reconstruction_vs_plus_history": c_reconstruction - c_with_history,
        },
        "estimate": {f"estimate_{k}": v for k, v in segments.items()},
        "estimate_sum_of_segments": sum_of_segments,
        # The residual. Positive means the real request carries text no segment accounts
        # for; negative means the segments double-count. Either is worth chasing.
        "unattributed": c_reconstruction - sum_of_segments,
    }


def audit_session(api: Api) -> list[str]:
    """Re-run app_smoke's stage-9 checks, so a token win cannot hide broken accounting.

    The session log's schema is fixed by the external cost comparison. A harness that
    reported a saving while the log that proves it had drifted would be worse than no
    harness.
    """
    data = api.reveal_session()["data"]
    problems: list[str] = []

    expected_keys = [
        "session_id", "started_at", "model", "circuit_file", "turns",
        "total_input_tokens", "total_output_tokens", "resolved",
    ]
    if list(data.keys()) != expected_keys:
        problems.append(f"session schema keys drifted: {list(data.keys())}")
    if any(
        t["input_tokens"] is None or t["output_tokens"] is None for t in data["turns"]
    ):
        problems.append("a turn has a null token count")
    if data["total_input_tokens"] != sum(t["input_tokens"] for t in data["turns"]):
        problems.append("total_input_tokens does not match the sum of the turns")
    if data["total_output_tokens"] != sum(t["output_tokens"] for t in data["turns"]):
        problems.append("total_output_tokens does not match the sum of the turns")
    assistant = [t for t in data["turns"] if t["role"] == "assistant"]
    if not assistant:
        problems.append("no assistant turn was logged")
    elif any(t["input_tokens"] <= 0 or t["output_tokens"] <= 0 for t in assistant):
        problems.append("an assistant turn has a zero token count")
    return problems


def history_probe(
    api: Api, circuit: Path, question: str, tool_mode: str
) -> list[dict[str, Any]]:
    """Watch the request grow as the session is used, which a single-question row cannot.

    Part 4 of the brief - whether history accumulates content the model is billed for
    twice - is structurally invisible to a one-shot measurement: the first request of a
    fresh process has one selection note in it no matter what. So this walks the session
    through the things a user actually does and prices the request after each:

      1. select the circuit (the baseline every other row measures)
      2. select **the same** circuit again, as a second sidebar click would
      3. one live turn, which spends tokens - the only paid step here
      4. select again, now with a turn behind it
      5. apply the patch, *if* the turn proposed one

    Step 5 is conditional on a real `pending_patch` rather than a fabricated approval: the
    gate is keyed on `(path, ref, value)` and inventing one would be measuring a code path
    the app does not have.
    """
    steps: list[dict[str, Any]] = []

    def snapshot(stage: str) -> None:
        row = count_only(api, question, tool_mode)
        row["stage"] = stage
        row["question"] = f"[{stage}] {question}"
        steps.append(row)

    snapshot("1-after-first-select")

    again = api.select_circuit(str(circuit))
    if not again.get("ok"):
        steps.append({"question": "[2-reselect]", "ok": False, "error": again.get("error", "")})
        return steps
    snapshot("2-after-reselect-same-circuit")

    turn = api.send_message(question)
    if not turn.get("ok"):
        steps.append({"question": "[3-live-turn]", "ok": False, "error": turn.get("error", "")})
        return steps
    snapshot("3-after-one-live-turn")

    third = api.select_circuit(str(circuit))
    if third.get("ok"):
        snapshot("4-after-select-following-a-turn")

    pending = turn.get("pending_patch")
    if not pending:
        steps.append(
            {
                "question": "[5-apply-patch]",
                "ok": True,
                "count_only": True,
                "skipped": "the turn proposed no patch, so there was nothing to approve",
                "tool_mode": tool_mode,
                "effort": None,
                "input_tokens": 0,
                "system_chars": 0,
                "history_messages": 0,
                "history_chars": 0,
                "message_shape": [],
                "tool_catalogue_names": [],
                "tools_sent": 0,
            }
        )
        return steps

    applied = api.apply_patch(
        pending["asc_path"], pending["ref"], pending["new_value"]
    )
    if not applied.get("ok"):
        steps.append(
            {"question": "[5-apply-patch]", "ok": False, "error": applied.get("error", "")}
        )
        return steps
    snapshot("5-after-apply-patch")
    return steps


def report(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print()
        print(f"  question: {row['question']}")
        if row.get("skipped"):
            print(f"  SKIPPED: {row['skipped']}")
            continue
        if not row.get("ok"):
            print(f"  FAILED: {row.get('error')}")
            continue
        if row.get("breakdown"):
            print(f"  mode:          {row['tool_mode']}")
            print(f"  effort:        {row['effort'] or '(gateway default)'}")
            print(f"  system prompt: {row['system_chars']} chars")
            print(f"  tools sent:    {row['tools_sent']}")
            print(f"  messages=[]:   {row['empty_messages_probe']}")
            c = row["components"]
            d = row["deltas"]
            print()
            print("  Token attribution (cumulative / delta):")
            print(f"    probe_floor:  {c['probe_floor']:>6}  (empty request)")
            print(f"    system_only:  {c['system_only']:>6}")
            print(f"    + tools:      {c['plus_tools']:>6}  (+{d['tools']})")
            print(f"    + user_msg:   {c['plus_user_msg']:>6}  (+{d['user_msg']})")
            print(f"    + history:    {c['plus_history']:>6}  (+{d['history']})")
            print(
                f"    reconstruct:  {c['reconstruction']:>6}"
                f"  (vs + history: {d['reconstruction_vs_plus_history']:+})"
            )
            print()
            print("  Per-segment estimates (count_tokens on pieces, NOT billed):")
            for name, value in row["estimate"].items():
                print(f"    {name:<34}{value:>6}")
            print(f"    {'sum of segments':<34}{row['estimate_sum_of_segments']:>6}")
            print(
                f"    {'UNATTRIBUTED':<34}{row['unattributed']:>6}"
                "  (reconstruction - sum)"
            )
            if not row["note_found_in_history"]:
                print("    NOTE: no selection note found in history")
            continue
        if row.get("count_only"):
            print(f"  mode:          {row['tool_mode']}")
            print(f"  effort:        {row['effort'] or '(gateway default)'}")
            print(f"  system prompt: {row['system_chars']} chars")
            print(f"  tools sent:    {row['tools_sent']}")
            names = ", ".join(row["tool_catalogue_names"]) or "(none)"
            print(f"  catalogue:     {names}")
            print(
                f"  history:       {row['history_messages']} messages,"
                f" {row['history_chars']} chars"
            )
            for i, shape in enumerate(row["message_shape"], start=1):
                print(
                    f"    [{i}] {shape['role']:<9} {shape['chars']:>6} chars"
                    f"  {shape['types']}"
                )
            print(f"  INPUT TOKENS:  {row['input_tokens']}  (no inference run)")
            continue
        print(f"  rounds:        {row['rounds']}")
        print(
            f"  tool calls:    {row['tool_call_count']}"
            + (f"  ({', '.join(row['tool_calls'])})" if row["tool_calls"] else "")
        )
        if row["tool_result_chars"]:
            print(f"  result chars:  {row['tool_result_chars']}")
        print(f"  input tokens:  {row['input_tokens']}")
        print(f"  output tokens: {row['output_tokens']}")
        print(f"  TOTAL tokens:  {row['total_tokens']}")
        print(f"  answer:        {row['answer_words']} words")
        print(
            f"  correct:       {'yes' if row['correct'] else 'NO'}"
            + (f"  (missing: {', '.join(row['missing'])})" if row["missing"] else "")
        )
        print("  --- answer ---")
        for line in row["answer"].splitlines():
            print(f"  | {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit", required=True,
        help="Path to a .asc, absolute or relative to the repo root.",
    )
    parser.add_argument(
        "--question", action="append", required=True,
        help="Repeatable. Asked in order, in one session, so later questions see the "
             "earlier answers.",
    )
    parser.add_argument(
        "--expect", action="append", default=[],
        help="Repeatable. A substring every answer set must contain for the run to count "
             "as correct. Checked whitespace- and case-insensitively.",
    )
    parser.add_argument("--label", help="Write the rows to bench/<label>.json.")
    parser.add_argument("--tool-mode", choices=TOOL_MODES, help="Override the configured mode.")
    parser.add_argument(
        "--count-only", action="store_true",
        help="Price the request without running inference. Free.",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Cumulative passes plus per-segment estimates and the unattributed residual. "
             "Free.",
    )
    parser.add_argument(
        "--history-probe", action="store_true",
        help="Price the request after each thing a user does (re-select, one turn, apply). "
             "Spends tokens: it runs one live turn.",
    )
    parser.add_argument(
        "--dump-request", metavar="DIR",
        help="Write every request the SDK actually sends, with the usage of its own "
             "response, to DIR. Paths are redacted.",
    )
    parser.add_argument(
        "--effort", choices=["auto", *sorted(UI_MODE_MAP)],
        help="Pin the reasoning effort instead of inheriting ambient SPICE_MCP_EFFORT. "
             "'auto' is the app's own classifier.",
    )
    parser.add_argument(
        "--force-tools", metavar="SPEC",
        help=f"Override the catalogue for this run: '{FORCE_ALL}', '{FORCE_NONE}', or a "
             "comma-separated list of tool names. Only changes what is *offered* - it "
             "cannot grant write access, which the approval gate owns.",
    )
    parser.add_argument("--json", action="store_true", help="Print the rows as JSON too.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    source = Path(args.circuit)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.is_file():
        print(f"No such circuit: {source}", file=sys.stderr)
        return 2

    config = load_config()
    if args.tool_mode:
        config = dataclasses.replace(config, tool_mode=args.tool_mode)

    workdir = Path(tempfile.mkdtemp(prefix="spice_mcp_measure_"))
    circuit = workdir / source.name
    shutil.copy2(source, circuit)

    api = Api(config)
    # `--history-probe` runs a live turn, so it is not free and its session log must be
    # audited like any other paid row.
    free_mode = args.count_only or args.breakdown
    mode_label = (
        "breakdown (free)" if args.breakdown
        else "count-only (free)" if args.count_only
        else "history probe (spends tokens)" if args.history_probe
        else "live (spends tokens)"
    )
    print(f"circuit:   {circuit}")
    print(f"model:     {config.model}")
    print(f"tool mode: {config.tool_mode}")
    print(f"mode:      {mode_label}")
    print(f"effort:    {args.effort or 'inherited from config/env'}")
    if args.force_tools:
        print(f"tools:     forced to {args.force_tools}")

    rows: list[dict[str, Any]] = []
    capture: RequestCapture | None = None
    dump_dir: Path | None = None
    session_totals: dict[str, Any] | None = None
    try:
        started = api.start()
        if not started.get("ok"):
            print(f"start failed: {started.get('error')}", file=sys.stderr)
            return 1
        print(f"tools:     {len(started['tools'])} exposed by the server")

        if args.effort:
            pinned = api.set_effort_mode(args.effort)
            if not pinned.get("ok"):
                print(f"effort failed: {pinned.get('error')}", file=sys.stderr)
                return 2

        if args.force_tools:
            forced = resolve_forced_tools(api, args.force_tools)
            if forced is None:
                return 2
            # Replacing the selector, not the server's catalogue: `Api._tools` stays all
            # seven, so `start()` and the approval gate see exactly what they always do.
            api._model_tools = lambda _text, _forced=forced: list(_forced)  # type: ignore[method-assign]
            names = ", ".join(t["name"] for t in forced) or "(none)"
            print(f"forced:    {len(forced)} tool(s) -> {names}")

        if args.dump_request:
            dump_dir = Path(args.dump_request)
            if not dump_dir.is_absolute():
                dump_dir = REPO_ROOT / dump_dir
            capture = RequestCapture(api, dump_dir, redaction_rules(workdir))
            capture.install()
            print(f"capturing: {dump_dir}")

        selection = api.select_circuit(str(circuit))
        if not selection.get("ok"):
            print(f"select failed: {selection.get('error')}", file=sys.stderr)
            return 1
        print(f"static:    {selection['checks'].get('summary', 'n/a')}")

        for question in args.question:
            if args.breakdown:
                rows.append(breakdown(api, question, config.tool_mode))
            elif args.count_only:
                rows.append(count_only(api, question, config.tool_mode))
            elif args.history_probe:
                rows.extend(history_probe(api, circuit, question, config.tool_mode))
            else:
                rows.append(measure(api, question, args.expect))

        report(rows)

        problems = audit_session(api) if not free_mode else []
        session_path = api.reveal_session()["path"]
        # The session's own totals, read before shutdown. Authoritative for anything paid:
        # they are the numbers the external cost comparison reads, and unlike a sum over
        # rows they include a live turn that a probe ran without producing a `measure` row.
        session_totals = api.totals()["totals"] if not free_mode else None
    finally:
        if capture is not None:
            capture.restore()
        api.shutdown()

    if capture is not None:
        print()
        print(f"captured {len(capture.events)} SDK call(s) -> {dump_dir}")
        for event in capture.events:
            usage = event.get("usage") or {}
            sizes = event["char_sizes"]
            print(
                f"  [{event['index']:02d}] {event['kind']:<12}"
                f" in={usage.get('input_tokens', '?')}"
                f" out={usage.get('output_tokens', '?')}"
                f"  chars: system={sizes['system']}"
                f" tools={sizes['tools']} messages={sizes['messages']}"
                f"  tools={len(event['tool_names'])}"
            )

    if not free_mode:
        # Only rows from a real turn. `--history-probe` mixes paid turns with free
        # `count_only` snapshots, and adding an estimate into a billed total is precisely
        # the confusion this whole investigation exists to clear up.
        total_in = (session_totals or {}).get("input_tokens", 0)
        total_out = (session_totals or {}).get("output_tokens", 0)
        print()
        print(f"SESSION TOTAL: {total_in} in / {total_out} out / {total_in + total_out}")
        print("  (billed, from the session log; count-only rows are estimates and are")
        print("   deliberately not added in - see the module docstring)")
        print(f"session log:   {session_path}")
        print(
            "session log audit: "
            + ("clean" if not problems else "PROBLEMS -> " + "; ".join(problems))
        )

    payload = {
        "label": args.label,
        "circuit": source.name,
        "model": config.model,
        "tool_mode": config.tool_mode,
        "count_only": args.count_only,
        "breakdown": args.breakdown,
        "history_probe": args.history_probe,
        # Provenance for the controlled comparison: a row whose effort or catalogue was
        # pinned is not comparable with one that inherited them.
        "effort_mode": args.effort or "inherited",
        "force_tools": args.force_tools,
        "dump_request": str(dump_dir) if dump_dir else None,
        "billed_usage": [
            {"kind": e["kind"], "usage": e["usage"], "tool_names": e["tool_names"]}
            for e in capture.events
        ] if capture is not None else None,
        "session_log": None if free_mode else session_path,
        "session_totals_billed": session_totals,
        "rows": rows,
    }
    if args.json:
        print()
        print(json.dumps(payload, indent=2))
    if args.label:
        BENCH_DIR.mkdir(exist_ok=True)
        out = BENCH_DIR / f"{args.label}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    failed = [r for r in rows if not r.get("ok") or r.get("missing")]
    if failed or (not free_mode and problems):
        print("\nFAILED: the run did not produce a correct, well-accounted answer.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
