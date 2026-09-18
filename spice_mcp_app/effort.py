"""Reasoning-effort classification for the LLM gateway.

The gateway's `output_config={"effort": ...}` parameter controls how hard the model
thinks. It does NOT reduce input tokens — only output/thinking tokens (roughly 770
per turn on Opus 5). That distinction matters: the token-cost experiment here must
attribute savings to the correct lever.

Verified values: "low", "medium", "high", "xhigh", "max".
The app exposes "light" (→ "low"), "medium" (→ "medium"), "hard" (→ "high") to avoid
leaking provider vocabulary into the UI.

None means "don't send the parameter" — the gateway uses its own default.
"""

from __future__ import annotations

from dataclasses import dataclass

# Keywords that suggest a simple preloaded diagnosis needs little reasoning.
_DIAG_WORDS = frozenset({"what", "why", "wrong", "broken", "issue", "bad", "fail"})

# Lookup / connectivity questions — also cheap.
_LOOKUP_WORDS = frozenset({"which", "where", "connected", "path", "node", "list"})

# Questions that need actual calculation or simulation reasoning.
_SIM_WORDS = frozenset({
    "simulat",  # matches "simulate", "simulation", "simulating", "simulated"
    "frequency", "bandwidth", "cutoff", "gain",
    "impedance", "calculat",  # matches "calculate", "calculating", "calculation"
    "db", "phase",
})


@dataclass(frozen=True)
class PreloadState:
    """What was successfully loaded when the current circuit was selected."""
    topology_ok: bool
    static_ok: bool
    static_finding_count: int


# Mapping from the UI mode names to provider values.
UI_MODE_MAP: dict[str, str] = {
    "light": "low",
    "medium": "medium",
    "hard": "high",
}


def classify_effort(user_text: str, state: PreloadState | None) -> str | None:
    """Return the effort string to use for this turn, or None for gateway default.

    Rules (first match wins):
    1. No preload state, or preload incomplete → None (gateway decides)
    2. static_finding_count > 0 AND any diagnosis word in text → "low"
       (findings already known → answering them requires little reasoning)
    3. Any lookup/connectivity word in text, AND no sim word → "low"
    4. Any sim/calc word in text → "medium"
    5. Default → None

    All checks are `in`-substring on lowercased text. No regex.
    """
    if state is None or not state.topology_ok or not state.static_ok:
        return None

    words = user_text.lower()

    has_sim = any(w in words for w in _SIM_WORDS)

    if state.static_finding_count > 0 and any(w in words for w in _DIAG_WORDS):
        return "low"

    if any(w in words for w in _LOOKUP_WORDS) and not has_sim:
        return "low"

    if has_sim:
        return "medium"

    return None
