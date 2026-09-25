"""Tests for tool_policy.py — the tool catalogue selector."""

from __future__ import annotations

import pytest

from spice_mcp_app.effort import PreloadState
from spice_mcp_app.tool_policy import ToolIntent, classify_intent, select_tools


# --- helpers --------------------------------------------------------------------------

ALL_TOOL_NAMES = [
    "read_netlist",
    "check_netlist_static",
    "run_simulation",
    "read_sim_log",
    "patch_component_value",
    "diff_netlist",
    "export_netlist",
]

ALL_TOOLS = [{"name": n} for n in ALL_TOOL_NAMES]


def loaded(findings=0):
    return PreloadState(topology_ok=True, static_ok=True, static_finding_count=findings)


def names(tools):
    return [t["name"] for t in tools]


# --- zero tools when preload complete and no intent ----------------------------------


def test_zero_tools_when_preload_complete_no_intent():
    result = select_tools(ALL_TOOLS, "why is my circuit wrong", loaded())
    assert result == []


def test_zero_tools_for_any_pure_diagnosis_with_loaded_preload():
    for text in (
        "what is wrong",
        "explain this circuit",
        "why no gain",
        "help me understand",
    ):
        assert select_tools(ALL_TOOLS, text, loaded()) == [], f"Expected [] for {text!r}"


# --- recovery tools when preload incomplete ------------------------------------------


def test_read_netlist_added_when_topology_not_loaded():
    state = PreloadState(topology_ok=False, static_ok=True, static_finding_count=0)
    result = names(select_tools(ALL_TOOLS, "why no gain", state))
    assert "read_netlist" in result


def test_static_added_when_static_not_loaded():
    state = PreloadState(topology_ok=True, static_ok=False, static_finding_count=0)
    result = names(select_tools(ALL_TOOLS, "why no gain", state))
    assert "check_netlist_static" in result


def test_both_recovery_tools_when_no_preload_state():
    result = names(select_tools(ALL_TOOLS, "why is my circuit wrong", None))
    assert "read_netlist" in result
    assert "check_netlist_static" in result


def test_no_widen_tools_with_recovery_only():
    result = names(select_tools(ALL_TOOLS, "why no gain", None))
    assert "diff_netlist" not in result
    assert "export_netlist" not in result


# --- write intent exposes patch tool -------------------------------------------------


def test_patch_offered_for_fix_intent():
    result = names(select_tools(ALL_TOOLS, "fix C1 to 100n", loaded()))
    assert "patch_component_value" in result


def test_patch_offered_for_change_intent():
    result = names(select_tools(ALL_TOOLS, "change R1 to 10k", loaded()))
    assert "patch_component_value" in result


def test_patch_offered_for_set_intent():
    result = names(select_tools(ALL_TOOLS, "set the capacitor to 1u", loaded()))
    assert "patch_component_value" in result


def test_patch_offered_for_apply_intent():
    result = names(select_tools(ALL_TOOLS, "apply the fix now", loaded()))
    assert "patch_component_value" in result


def test_patch_not_offered_for_pure_diagnosis():
    result = names(select_tools(ALL_TOOLS, "why no gain", loaded()))
    assert "patch_component_value" not in result


def test_patch_not_offered_for_lookup():
    result = names(select_tools(ALL_TOOLS, "what components are connected", loaded()))
    assert "patch_component_value" not in result


# --- simulation tools exposed for sim keywords ---------------------------------------


def test_sim_tools_offered_for_simulate_keyword():
    result = names(select_tools(ALL_TOOLS, "simulate this and check gain", loaded()))
    assert "run_simulation" in result
    assert "read_sim_log" in result


def test_sim_tools_offered_for_run_keyword():
    result = names(select_tools(ALL_TOOLS, "run the simulation", loaded()))
    assert "run_simulation" in result
    assert "read_sim_log" in result


def test_sim_tools_offered_for_ltspice_keyword():
    result = names(select_tools(ALL_TOOLS, "what does ltspice say", loaded()))
    assert "run_simulation" in result


def test_no_sim_tools_for_pure_diagnosis():
    result = names(select_tools(ALL_TOOLS, "why no gain", loaded()))
    assert "run_simulation" not in result
    assert "read_sim_log" not in result


# --- widen keywords return all 7 -----------------------------------------------------


def test_widen_returns_all_seven_for_diff():
    result = select_tools(ALL_TOOLS, "diff the two netlists for me", loaded())
    assert len(result) == 7


def test_widen_returns_all_seven_for_compare():
    result = select_tools(ALL_TOOLS, "compare it against the backup", loaded())
    assert len(result) == 7


def test_widen_returns_all_seven_for_export():
    result = select_tools(ALL_TOOLS, "export this to a .net file", loaded())
    assert len(result) == 7


def test_widen_returns_all_seven_for_before_and_after():
    result = select_tools(ALL_TOOLS, "show me before and after", loaded())
    assert len(result) == 7


def test_diff_export_not_offered_without_widen():
    result = names(select_tools(ALL_TOOLS, "why no gain", loaded()))
    assert "diff_netlist" not in result
    assert "export_netlist" not in result


def test_diff_export_not_offered_even_without_preload():
    result = names(select_tools(ALL_TOOLS, "why no gain", None))
    assert "diff_netlist" not in result
    assert "export_netlist" not in result


# --- widen overrides preload state ---------------------------------------------------


def test_widen_returns_all_seven_even_with_preload_complete():
    """Full preload + widen keyword → all 7, not zero."""
    result = select_tools(ALL_TOOLS, "diff the netlists", loaded())
    assert len(result) == 7


# --- select_tools preserves order from all_tools -------------------------------------


def test_select_tools_preserves_order():
    result = names(select_tools(ALL_TOOLS, "why no gain", None))
    indices = [ALL_TOOL_NAMES.index(n) for n in result]
    assert indices == sorted(indices), "select_tools must preserve the order of all_tools"


# --- empty tools list ----------------------------------------------------------------


def test_select_tools_with_empty_all_tools():
    assert select_tools([], "any text", loaded()) == []


def test_select_tools_with_empty_all_tools_no_state():
    assert select_tools([], "why no gain", None) == []
