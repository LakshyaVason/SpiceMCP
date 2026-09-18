"""Tool catalogue selection — decides which tools each request is offered.

The MCP server always exposes all 7 tools. This module decides which subset the
*request* sends to the model. Fewer schemas = fewer billed input tokens, since
every schema costs tokens in every request (prompted_json: in the system prompt;
native: in the tools parameter).

**Invariant**: This module can only withhold tools. The approval gate in
`api.Api._tool_executor` is the only thing that matters for safety — offering a tool
does not grant write access.

**Preload-complete = zero tools**: When the circuit topology and static check have
already been injected into context, a simple diagnosis turn needs no tool calls. The
model answers from what it already has. If the answer requires more information, it
can still say so and the next turn will expand the catalogue.

**Recovery**: Preload failure always exposes the recovery tools so the model can get
what it needs. Never tell the model to rely on something that is not there.

**Write intent exposes patch tool**: patch_component_value is only offered when the
text suggests the user wants a change applied. Without it, the model states its fix
in text, no pending_patch is set, the diff panel never opens, and there is nothing
to approve. It cannot bypass the gate — only the user's explicit Apply does that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .effort import PreloadState

# Words that trigger write-tool exposure.
_WRITE_WORDS = frozenset({"change", "set", "fix", "apply", "update", "replace", "adjust", "correct"})

# Words that trigger simulation-tool exposure.
_SIM_WORDS = frozenset({"simulate", "run", "log", "ltspice", "waveform", "transient", "ac", "dc"})

# Words that widen the catalogue to all 7.
_WIDEN_WORDS = ("diff", "compar", "export", "before and after", ".net")


@dataclass
class ToolIntent:
    """What this turn probably needs."""
    needs_write: bool
    needs_simulation: bool
    needs_read: bool       # topology not loaded
    needs_static: bool     # static check not loaded
    widen: bool            # user asked for diff, export, comparison


def classify_intent(user_text: str, state: PreloadState | None) -> ToolIntent:
    lowered = (user_text or "").lower()
    tokens = set(lowered.split())

    needs_read = state is None or not state.topology_ok
    needs_static = state is None or not state.static_ok
    widen = any(w in lowered for w in _WIDEN_WORDS)
    needs_write = bool(tokens & _WRITE_WORDS)
    needs_simulation = bool(tokens & _SIM_WORDS)

    return ToolIntent(
        needs_write=needs_write,
        needs_simulation=needs_simulation,
        needs_read=needs_read,
        needs_static=needs_static,
        widen=widen,
    )


def _tool_name(tool: dict) -> str:
    """Extract name from an Anthropic-format tool dict."""
    return tool.get("name", "")


def select_tools(
    all_tools: list[dict], user_text: str, state: PreloadState | None
) -> list[dict]:
    """Return the subset of tools to offer for this request.

    The return value is always a sublist of all_tools, preserving order.
    An empty list is valid and means "answer from what you already have".

    This can only withhold tools — it cannot grant capabilities the server
    does not have, and it does not affect the approval gate.
    """
    intent = classify_intent(user_text, state)

    if intent.widen:
        return list(all_tools)

    selected: set[str] = set()

    if intent.needs_read:
        selected.add("read_netlist")
    if intent.needs_static:
        selected.add("check_netlist_static")
    if intent.needs_write:
        selected.add("patch_component_value")
    if intent.needs_simulation:
        selected.add("run_simulation")
        selected.add("read_sim_log")

    return [t for t in all_tools if _tool_name(t) in selected]
