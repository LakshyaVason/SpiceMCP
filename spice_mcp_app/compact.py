"""Model-facing projections of MCP tool results.

The MCP result is a complete record. What the model needs is a fraction of it, and the
rest is not free: the `read_netlist` result for a four-component circuit is 2332 chars of
pretty-printed JSON, two thirds of which is `%TEMP%` staging paths, a `raw_text` copy of
the netlist that is already spelled out component by component above it, and per-component
`prefix`/`kind`/`line_no`/`raw_line`/`nodes_uncertain` bookkeeping. Escaped Windows paths
tokenise at roughly two characters per token, so that one result cost about 1100 input
tokens - and it was re-sent on every subsequent round.

This module projects. It does not replace: `ToolCallRecord.result` keeps the full JSON, so
the session log, the UI's preview and `api._pending_patch` all see exactly what they saw
before. Compact for the model, complete for the record.

**This lives in the app, never in the server.** The server must not learn what an LLM is -
that invariant is what makes it reusable for the later EDA-tool work - and a projection is
by definition a decision about what a model needs.

Two failure directions, chosen deliberately and differently:

  * `compact_tool_result` fails **open**. Anything it cannot parse or does not recognise
    returns `None`, meaning "send it unchanged". Losing a real tool result to a bug in a
    projection would be far worse than paying for the full text.
  * `circuit_summary` fails **closed**. Handed something that is not a `Netlist` it returns
    `None` rather than a half-rendered summary, because its other caller
    (`api._selection_note`) uses the return value to decide whether to *tell the model the
    topology is already loaded*. A summary that renders from the wrong payload would put a
    false claim in the system context, which is the one way this optimisation could turn
    into a hallucination.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Tools whose results are projected. Everything else passes through untouched - in
# particular `patch_component_value`, whose JSON `api._pending_patch` parses to drive the
# Apply button, and `diff_netlist`/`export_netlist`, whose payload *is* the answer.
COMPACTED_TOOLS = frozenset(
    {"read_netlist", "check_netlist_static", "run_simulation", "read_sim_log"}
)

# How much of a failed run's raw log text to keep. LTspice's fatal messages have no common
# shape - of five distinct failures only two carry an `ERROR:`/`WARNING:` prefix - so an
# unclassified fatal message exists nowhere but the raw text. Dropping it on a failed run
# would remove grounding to win the metric. The tail is kept rather than the head because
# the banner is at the top and the diagnosis is at the bottom.
MAX_RAW_TAIL = 4000


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _join(values: Any, *, sep: str = " ") -> str:
    """Join a list of net names or refs, tolerating a payload that is not a list."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ""
    return sep.join(_text(v) for v in values)


def _rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """The list at `key`, with non-mapping entries dropped rather than crashing."""
    raw = payload.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def circuit_summary(netlist: Mapping[str, Any]) -> str | None:
    """Project a `Netlist` into the netlist-shaped summary a model can read directly.

        Components:
          V1 Vin 0 AC 0.7 3000
          R1 Vout NC_01 10k
        Nets:
          Vin: V1
          NC_01: R1
        Directives:
          .ac dec 10 1k 100k

    Keeps every reference designator, every node in pin order, every value as written,
    every net with the refs touching it, and every directive - which is everything needed
    to reason about topology or to name a component in a fix.

    Drops `source_path`, `netlist_path`, `title` and `raw_text` (paths the model must not
    quote back at the user, and a duplicate of the components above), and per-component
    `prefix`, `kind`, `line_no` and `raw_line` (the element letter is the first character
    of the ref, and the line number is not something a model should cite).

    `connection_count` goes too - it is `len(connected_refs)`, and a net printed with one
    ref is visibly a floating one.

    `nodes_uncertain` is the exception that does not get dropped: it means the pin count
    could not be determined, which is why the server *downgrades* its own connectivity
    checks. Silently omitting it would let the model assert a floating net the server was
    careful not to assert.

    Returns `None` when the payload is not a netlist - see the module docstring on why
    this direction fails closed.
    """
    if not isinstance(netlist, Mapping):
        return None
    components = _rows(netlist, "components")
    if not components or "nets" not in netlist:
        # No components, or a payload from some other tool. Either way there is nothing
        # here worth claiming the topology from.
        return None

    lines = ["Components:"]
    for component in components:
        parts = (
            _text(component.get("ref")),
            _join(component.get("nodes")),
            _text(component.get("value")),
        )
        lines.append("  " + " ".join(p for p in parts if p))

    nets = _rows(netlist, "nets")
    if nets:
        lines.append("Nets:")
        for net in nets:
            refs = _join(net.get("connected_refs"), sep=", ")
            lines.append(f"  {_text(net.get('name'))}: {refs}")

    directives = _rows(netlist, "directives")
    if directives:
        lines.append("Directives:")
        lines.extend(f"  {_text(d.get('text'))}" for d in directives)

    subcircuits = _rows(netlist, "subcircuits")
    if subcircuits:
        lines.append("Subcircuits:")
        for sub in subcircuits:
            ports = _join(sub.get("ports"))
            lines.append(f"  .subckt {_text(sub.get('name'))} {ports}".rstrip())

    uncertain = [_text(c.get("ref")) for c in components if c.get("nodes_uncertain")]
    if uncertain:
        lines.append(
            "Note: the pin count of "
            + ", ".join(uncertain)
            + " could not be determined, so the node lists above may be wrong for those "
            "elements and connectivity claims about them are unsafe."
        )

    return "\n".join(lines)


def _findings(payload: Mapping[str, Any]) -> list[str]:
    """Findings rendered one per line, keeping severity, refs, nets and the suggestion.

    `line_no` is dropped: the model cannot see the file the number indexes into, and citing
    one invites it to describe a location the user cannot check.
    """
    out: list[str] = []
    for finding in _rows(payload, "findings"):
        where = [
            part
            for part in (
                _join(finding.get("refs"), sep=", "),
                _join(finding.get("nets"), sep=", "),
            )
            if part
        ]
        head = f"  {_text(finding.get('severity'))}/{_text(finding.get('check'))}"
        if where:
            head += f" [{', '.join(where)}]"
        line = f"{head}: {_text(finding.get('message'))}"
        suggestion = _text(finding.get("suggestion"))
        if suggestion:
            line += f" -> {suggestion}"
        out.append(line)
    return out


def static_summary(result: Mapping[str, Any]) -> str | None:
    """Project a `StaticCheckResult`. Drops only `source_path` - the findings are the point."""
    if not isinstance(result, Mapping) or "findings" not in result:
        return None
    lines = [f"static check: {_text(result.get('summary'))}"]
    if result.get("ok") is True and not _rows(result, "findings"):
        lines.append("  no findings.")
    lines.extend(_findings(result))
    return "\n".join(lines)


def log_summary(log: Mapping[str, Any], *, prefix: str = "") -> list[str]:
    """The lines shared by `read_sim_log` and the `log` nested inside `run_simulation`."""
    lines = [f"{prefix}succeeded: {bool(log.get('succeeded'))}"]
    summary = _text(log.get("summary"))
    if summary:
        lines.append(f"{prefix}{summary}")
    lines.extend(_findings(log))

    measurements = _rows(log, "measurements")
    if measurements:
        lines.append(f"{prefix}measurements:")
        lines.extend(
            f"  {_text(m.get('name'))} = {_text(m.get('value'))}" for m in measurements
        )
    if log.get("step_count"):
        lines.append(f"{prefix}.step iterations: {log.get('step_count')}")

    # Only on a failure, and only the tail. See MAX_RAW_TAIL.
    if not log.get("succeeded"):
        raw = _text(log.get("raw_text")).strip()
        if raw:
            if len(raw) > MAX_RAW_TAIL:
                raw = "[earlier log output omitted]\n" + raw[-MAX_RAW_TAIL:]
            lines.append(f"{prefix}raw log (the run failed, so this may hold the reason):")
            lines.append(raw)
    return lines


def sim_summary(result: Mapping[str, Any]) -> str | None:
    """Project a `SimulationResult`.

    Drops `circuit_path`, `simulated_path`, `log_path`, `raw_path` (all staging paths in
    `%TEMP%`), `returncode`, `elapsed_s`, `solver`, `method` and `files_loaded`. The exit
    code in particular is worth removing rather than merely shrinking: LTspice exits 0 for
    several genuinely failed runs, so showing the model a number it is told not to trust is
    an invitation to trust it.
    """
    if not isinstance(result, Mapping) or "succeeded" not in result:
        return None
    lines = [f"simulation succeeded: {bool(result.get('succeeded'))}"]
    if result.get("timed_out"):
        lines.append(
            "TIMED OUT and was killed - usually an oscillation or a collapsed timestep, "
            "which is itself the diagnosis."
        )
    summary = _text(result.get("summary"))
    if summary:
        lines.append(summary)

    log = result.get("log")
    if isinstance(log, Mapping):
        lines.extend(log_summary(log))
    else:
        lines.append("LTspice produced no log at all.")
    return "\n".join(lines)


def sim_log_summary(log: Mapping[str, Any]) -> str | None:
    """Project a `SimLog` from `read_sim_log`. Drops `log_path`, the version banner and friends."""
    if not isinstance(log, Mapping) or "succeeded" not in log:
        return None
    return "\n".join(log_summary(log))


def compact_tool_result(name: str, result: str) -> str | None:
    """The projection for one tool result, or `None` to send it unchanged.

    `None` is the safe answer and is returned for every case this cannot handle: a tool
    with no projection, text that is not JSON, JSON that is not an object, an error result,
    a payload whose shape does not match. Failing open means a bug here costs tokens rather
    than evidence.
    """
    if name not in COMPACTED_TOOLS:
        return None
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        # Error results are plain prose - "REFUSED: ...", "Tool x failed: ..." - and must
        # reach the model verbatim.
        return None
    if not isinstance(payload, Mapping):
        return None

    try:
        if name == "read_netlist":
            return circuit_summary(payload)
        if name == "check_netlist_static":
            return static_summary(payload)
        if name == "run_simulation":
            return sim_summary(payload)
        if name == "read_sim_log":
            return sim_log_summary(payload)
    except Exception:  # pragma: no cover - defensive; the fail-open path
        logger.warning("could not compact the %s result; sending it in full", name)
        return None
    return None
