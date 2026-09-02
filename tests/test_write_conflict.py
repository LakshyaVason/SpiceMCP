"""Tests for the LTspice-is-open warnings.

The decision behind these is "warn, then write" - the write is never blocked. The user asked
for the fix and the file is theirs. But LTspice reads a .asc once, at open, and writes its
own in-memory copy back on save, so it will happily overwrite a fix it never noticed. A
silent success would hand the user a change that quietly disappears later.

`ltspice_is_running` is monkeypatched in every test. Without that these would pass or fail
depending on whether the developer happens to have LTspice open, which is the worst kind of
flaky test - it would go green on the machine where the feature is broken.
"""

from __future__ import annotations

import pytest

# The `api` fixture - an Api on a FakeMCP with a temp session - lives in conftest.py, shared
# with the approval-gate tests in test_llm.py. Two copies of that setup drifted apart once.


@pytest.fixture
def asc(tmp_path):
    target = tmp_path / "lowpass.asc"
    target.write_text("Version 4.1\n", encoding="utf-8")
    return target


def running(monkeypatch, state: bool) -> None:
    monkeypatch.setattr("spice_mcp_app.api.ltspice_is_running", lambda: state)


# --- after a write --------------------------------------------------------------------


def test_the_write_happens_and_the_warning_comes_after(api, asc, monkeypatch):
    running(monkeypatch, True)

    result = api.apply_patch(str(asc), "C1", "100n")

    assert result["ok"]
    # The write must not have been blocked.
    assert api._mcp.calls[-1][1]["apply"] is True
    assert result["warning"]


def test_the_warning_names_the_file_and_the_way_out(api, asc, monkeypatch):
    running(monkeypatch, True)

    warning = api.apply_patch(str(asc), "C1", "100n")["warning"]

    assert "lowpass.asc" in warning
    assert "Revert" in warning
    assert "overwrite" in warning


def test_there_is_no_warning_when_ltspice_is_closed(api, asc, monkeypatch):
    running(monkeypatch, False)

    result = api.apply_patch(str(asc), "C1", "100n")

    assert result["ok"]
    assert result["warning"] is None


def test_the_approval_gate_still_refuses_an_unapproved_write(api, asc, monkeypatch):
    """The new code path must not have weakened the one promise that matters."""
    running(monkeypatch, True)

    refused = api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": True},
    )

    assert "REFUSED" in refused
    assert api._mcp.calls == [], "the write reached the server despite the gate"


# --- before a read --------------------------------------------------------------------


def test_selecting_a_circuit_warns_that_disk_may_be_stale(api, asc, monkeypatch):
    running(monkeypatch, True)

    warning = api.select_circuit(str(asc))["warning"]

    assert warning and "unsaved" in warning


def test_the_stale_warning_is_not_repeated(api, asc, monkeypatch):
    """A warning that fires on every click is one people learn to skip past."""
    running(monkeypatch, True)

    assert api.select_circuit(str(asc))["warning"]
    assert api.select_circuit(str(asc))["warning"] is None


def test_no_stale_warning_when_ltspice_is_closed(api, asc, monkeypatch):
    running(monkeypatch, False)

    assert api.select_circuit(str(asc))["warning"] is None


def test_the_file_we_just_opened_ourselves_is_not_flagged_as_stale(api, asc, monkeypatch):
    """On the Explorer path we opened the GUI microseconds ago - disk and screen agree.

    Warning here would fire on every single launch, at the moment it is least true.
    """
    running(monkeypatch, True)
    api.initial_circuit = str(asc)
    api.opened_in_ltspice = True

    assert api.select_circuit(str(asc))["warning"] is None
    # But a later look is fair game again: by then the user may have edited in the GUI.
    assert api.select_circuit(str(asc))["warning"]


def test_the_suppression_applies_only_to_the_launched_circuit(api, asc, tmp_path, monkeypatch):
    running(monkeypatch, True)
    other = tmp_path / "other.asc"
    other.write_text("Version 4.1\n", encoding="utf-8")
    api.initial_circuit = str(asc)
    api.opened_in_ltspice = True

    assert api.select_circuit(str(other))["warning"], "a different file was wrongly suppressed"


def test_path_comparison_is_case_insensitive_like_windows(api, asc, monkeypatch):
    running(monkeypatch, True)
    api.initial_circuit = str(asc).upper()
    api.opened_in_ltspice = True

    assert api.select_circuit(str(asc))["warning"] is None


# --- the detector itself ---------------------------------------------------------------


def test_the_detector_answers_without_raising():
    """It runs against the real process list, so it must be robust on any machine."""
    from spice_mcp_server.ltspice import ltspice_is_running

    assert ltspice_is_running() in (True, False)


def test_the_detector_is_not_an_mcp_tool():
    """A UX detail the model never needs; an eighth schema would cost tokens every turn.

    Asserted as the exact tool set rather than just the absence of one name, so that any
    tool added without a deliberate decision fails here.
    """
    import asyncio

    from spice_mcp_server.server import mcp

    assert sorted(tool.name for tool in asyncio.run(mcp.list_tools())) == [
        "check_netlist_static",
        "diff_netlist",
        "export_netlist",
        "patch_component_value",
        "read_netlist",
        "read_sim_log",
        "run_simulation",
    ]
