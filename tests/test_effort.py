"""Tests for effort.py — the reasoning-effort classifier."""

from __future__ import annotations

import pytest

from spice_mcp_app.effort import PreloadState, UI_MODE_MAP, classify_effort


# --- helpers --------------------------------------------------------------------------

def loaded(findings=0):
    """A fully-loaded preload state."""
    return PreloadState(topology_ok=True, static_ok=True, static_finding_count=findings)


def partial_load(*, topology=True, static=True, findings=0):
    return PreloadState(
        topology_ok=topology, static_ok=static, static_finding_count=findings
    )


# --- no preload state → None ----------------------------------------------------------


def test_classify_effort_none_with_no_preload_state():
    assert classify_effort("why is my circuit wrong", None) is None


def test_classify_effort_none_when_topology_not_loaded():
    state = partial_load(topology=False)
    assert classify_effort("why is my circuit wrong", state) is None


def test_classify_effort_none_when_static_not_loaded():
    state = partial_load(static=False)
    assert classify_effort("why does this fail", state) is None


# --- simple diagnosis → "low" ---------------------------------------------------------


def test_classify_effort_low_for_preloaded_simple_diagnosis():
    # findings>0 + diagnosis word ("why") fires rule 2 before "gain" triggers sim rule.
    state = loaded(findings=2)
    assert classify_effort("why is my circuit not getting any gain", state) == "low"


def test_classify_effort_low_for_what_question():
    # findings > 0 required for rule 2 to fire on diagnosis words.
    assert classify_effort("what is wrong with this circuit", loaded(findings=1)) == "low"


def test_classify_effort_low_for_broken_question():
    # findings > 0 required for rule 2 to fire on diagnosis words.
    assert classify_effort("my circuit is broken, help", loaded(findings=1)) == "low"


def test_classify_effort_low_for_fail_question():
    # findings > 0 required for rule 2 to fire on diagnosis words.
    assert classify_effort("why does this fail", loaded(findings=1)) == "low"


def test_classify_effort_low_for_bad_question():
    # findings > 0 required for rule 2 to fire on diagnosis words.
    assert classify_effort("something looks bad here", loaded(findings=1)) == "low"


# --- lookup/connectivity → "low" -----------------------------------------------------


def test_classify_effort_low_for_lookup_question():
    assert classify_effort("which component is connected here", loaded()) == "low"


def test_classify_effort_low_for_where_question():
    assert classify_effort("where does node VCC connect", loaded()) == "low"


def test_classify_effort_low_for_list_question():
    assert classify_effort("list all the components", loaded()) == "low"


def test_classify_effort_low_for_path_question():
    assert classify_effort("what path does the signal take", loaded()) == "low"


# --- sim/calc → "medium" -------------------------------------------------------------


def test_classify_effort_medium_for_sim_question():
    # "simulation" matches "simulat" prefix; "gain" would match sim too.
    assert classify_effort("run the simulation", loaded()) == "medium"


def test_classify_effort_medium_for_bandwidth_question():
    assert classify_effort("what is the bandwidth", loaded()) == "medium"


def test_classify_effort_medium_for_cutoff_question():
    assert classify_effort("calculate the cutoff frequency", loaded()) == "medium"


def test_classify_effort_medium_for_impedance_question():
    assert classify_effort("what is the input impedance", loaded()) == "medium"


def test_classify_effort_medium_for_frequency_question():
    assert classify_effort("at what frequency does gain drop", loaded()) == "medium"


# --- sim words take priority over diagnosis words ------------------------------------


def test_sim_words_take_priority_over_diagnosis_words():
    """"simulation" contains "simulate" (substring match). With findings=0, rule 2 does
    not fire; "simulate" is in sim words → medium."""
    result = classify_effort("why does the simulation fail", loaded(findings=0))
    assert result == "medium"


# --- default (no keyword match) → None -----------------------------------------------


def test_classify_effort_default_returns_none():
    assert classify_effort("show me the circuit", loaded()) is None


def test_classify_effort_default_for_short_message():
    assert classify_effort("ok", loaded()) is None


# --- UI mode mapping ------------------------------------------------------------------


def test_manual_modes_map_to_correct_provider_values():
    assert UI_MODE_MAP["light"] == "low"
    assert UI_MODE_MAP["medium"] == "medium"
    assert UI_MODE_MAP["hard"] == "high"


def test_auto_mode_not_in_ui_mode_map():
    """'auto' is handled by the caller (classify_effort), not as a static mapping."""
    assert "auto" not in UI_MODE_MAP


def test_all_mapped_values_are_valid_effort_strings():
    """Values must be strings the gateway accepts (no capitals, no extras)."""
    valid = {"low", "medium", "high", "xhigh", "max"}
    for val in UI_MODE_MAP.values():
        assert val in valid
