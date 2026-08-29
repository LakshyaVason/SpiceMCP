"""Tests for the Explorer launcher.

Nothing here opens a window, starts LTspice, or shows a message box. `subprocess.Popen`,
the app entry point and the dialog are all monkeypatched to recorders, which is what makes
the interesting assertions possible: that a bad argument spawns *nothing*, and that a
missing LTspice still opens the client.

The "writes nothing beside the user's circuit" test is modelled on
test_ltspice.py's directory-snapshot test. Same reason: the launcher is a new code path that
runs with the user's own folder as its working directory, which is exactly the situation the
original incident happened in.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from conftest import FIXTURES, REPO_ROOT
from spice_mcp_app import launch


@pytest.fixture
def recorder(monkeypatch):
    """Replace every side effect with a recorder. Returns the record."""
    record: dict[str, object] = {"popen": [], "boxes": [], "app": []}

    def fake_popen(args, **kwargs):
        record["popen"].append({"args": args, "kwargs": kwargs})
        return object()

    def fake_app(argv, *, startup_note=None, opened_in_ltspice=False):
        record["app"].append(
            {
                "argv": list(argv),
                "startup_note": startup_note,
                "opened_in_ltspice": opened_in_ltspice,
            }
        )
        return 0

    monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launch, "message_box", lambda text, title="x": record["boxes"].append(text))
    # Imported inside main(), so patch it where it is looked up.
    monkeypatch.setattr("spice_mcp_app.__main__.main", fake_app)
    return record


@pytest.fixture
def keep_cwd():
    original = os.getcwd()
    yield
    os.chdir(original)


@pytest.fixture
def asc(tmp_path):
    target = tmp_path / "wrong_value_lowpass.asc"
    shutil.copy2(FIXTURES / "wrong_value_lowpass.asc", target)
    return target


# --- argument validation --------------------------------------------------------------


def test_a_missing_file_is_refused_and_nothing_is_spawned(recorder, tmp_path, keep_cwd):
    code = launch.main([str(tmp_path / "nope.asc"), "--no-ltspice"])

    assert code != 0
    assert recorder["popen"] == []
    assert recorder["app"] == [], "the window opened on a file that does not exist"
    assert "No such file" in recorder["boxes"][0]


def test_a_directory_is_refused(recorder, tmp_path, keep_cwd):
    code = launch.main([str(tmp_path), "--no-ltspice"])

    assert code != 0
    assert recorder["app"] == []


def test_a_non_asc_file_is_refused(recorder, tmp_path, keep_cwd):
    net = tmp_path / "RCLP.net"
    net.write_text("not a schematic\n", encoding="utf-8")

    code = launch.main([str(net), "--no-ltspice"])

    assert code != 0
    assert recorder["app"] == []
    assert ".asc" in recorder["boxes"][0]


def test_an_uppercase_suffix_is_accepted(recorder, tmp_path, keep_cwd):
    """Explorer will hand back whatever case the file actually has."""
    shouty = tmp_path / "RCLP.ASC"
    shutil.copy2(FIXTURES / "good_lowpass.asc", shouty)

    assert launch.main([str(shouty), "--no-ltspice"]) == 0
    assert len(recorder["app"]) == 1


def test_the_app_is_given_one_resolved_absolute_path(recorder, asc, keep_cwd):
    launch.main([str(asc), "--no-ltspice"])

    assert recorder["app"][0]["argv"] == ["--file", str(asc.resolve())]


def test_debug_is_passed_through(recorder, asc, keep_cwd):
    launch.main([str(asc), "--no-ltspice", "--debug"])

    assert recorder["app"][0]["argv"] == ["--file", str(asc.resolve()), "--debug"]


# --- the LTspice launch ---------------------------------------------------------------


def test_ltspice_is_started_detached_with_no_pipes(recorder, asc, monkeypatch, keep_cwd):
    fake_exe = Path(r"C:\fake\LTspice.exe")
    monkeypatch.setattr(
        "spice_mcp_server.ltspice.find_ltspice_exe", lambda: fake_exe
    )
    import subprocess

    launch.main([str(asc)])

    assert len(recorder["popen"]) == 1
    call = recorder["popen"][0]
    # Exactly the exe and the file. An accidental -b here would run a simulation in the
    # user's own directory, which is the incident the staging rule exists for.
    assert call["args"] == [str(fake_exe), str(asc.resolve())]

    flags = call["kwargs"]["creationflags"]
    assert flags & subprocess.DETACHED_PROCESS
    assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    # A pipe nobody drains can block a GUI process we never wait on.
    for stream in ("stdin", "stdout", "stderr"):
        assert call["kwargs"][stream] is subprocess.DEVNULL


def test_the_users_own_path_is_handed_to_the_gui_not_a_staged_copy(
    recorder, asc, monkeypatch, keep_cwd
):
    """Opening the user's file for editing is the point; a temp copy would be useless."""
    monkeypatch.setattr(
        "spice_mcp_server.ltspice.find_ltspice_exe", lambda: Path(r"C:\fake\LTspice.exe")
    )
    launch.main([str(asc)])

    handed = Path(recorder["popen"][0]["args"][1])
    assert handed == asc.resolve()
    assert "spice_mcp_work" not in str(handed)


def test_a_missing_ltspice_does_not_stop_the_client(recorder, asc, monkeypatch, keep_cwd):
    """The client can read, check and patch a schematic without the GUI."""
    from spice_mcp_server.ltspice import LTSpiceNotFound

    def no_exe():
        raise LTSpiceNotFound("Could not find LTspice.exe. Set LTSPICE_EXE.")

    monkeypatch.setattr("spice_mcp_server.ltspice.find_ltspice_exe", no_exe)

    assert launch.main([str(asc)]) == 0
    assert len(recorder["app"]) == 1, "the client refused to open because LTspice was missing"
    assert recorder["boxes"] == [], "a recoverable problem should not raise a dialog"

    note = recorder["app"][0]["startup_note"]
    assert note and "LTSPICE_EXE" in note
    assert recorder["app"][0]["opened_in_ltspice"] is False


def test_a_successful_gui_launch_is_reported_to_the_api(recorder, asc, monkeypatch, keep_cwd):
    """This is what suppresses the false 'save first' warning on the Explorer path."""
    monkeypatch.setattr(
        "spice_mcp_server.ltspice.find_ltspice_exe", lambda: Path(r"C:\fake\LTspice.exe")
    )
    launch.main([str(asc)])

    assert recorder["app"][0]["opened_in_ltspice"] is True
    assert recorder["app"][0]["startup_note"] is None


def test_no_ltspice_flag_skips_the_gui_entirely(recorder, asc, keep_cwd):
    launch.main([str(asc), "--no-ltspice"])

    assert recorder["popen"] == []
    assert recorder["app"][0]["opened_in_ltspice"] is False


# --- not disturbing the user's directory ----------------------------------------------


def test_the_launcher_writes_nothing_beside_the_users_circuit(recorder, asc, keep_cwd):
    """launch.log must not land in the folder the schematic came from."""
    sibling = asc.parent / "wrong_value_lowpass.net"
    sibling.write_bytes(b'"ExpressPCB Netlist"\r\n')
    before = {p.name: p.read_bytes() for p in asc.parent.iterdir()}

    launch.main([str(asc), "--no-ltspice"])

    after = {p.name: p.read_bytes() for p in asc.parent.iterdir()}
    assert after == before, "the launcher wrote into the user's source directory"


def test_it_does_not_hold_the_users_folder_as_the_working_directory(recorder, asc, keep_cwd):
    """Explorer hands us the circuit's folder; holding it locks it against rename."""
    os.chdir(asc.parent)

    launch.main([str(asc), "--no-ltspice"])

    assert Path(os.getcwd()).resolve() == REPO_ROOT.resolve()


def test_a_relative_path_still_resolves_correctly(recorder, asc, keep_cwd):
    """The chdir to the repo must happen after the argument is resolved, not before."""
    os.chdir(asc.parent)

    launch.main([asc.name, "--no-ltspice"])

    assert recorder["app"][0]["argv"] == ["--file", str(asc.resolve())]
