"""Tests for LTspice invocation, and for not damaging the user's files.

The file-safety tests exist because of a real incident during development: LTspice
writes -netlist and -b output *next to the input file*, and an invocation made with the
project directory as the working directory overwrote a hand-exported RCLP.net. The
server stages inputs into a scratch directory precisely to prevent that, and these
tests pin that behaviour down.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import FIXTURES, needs_ltspice
from spice_mcp_server import ltspice

SENTINEL = b'"ExpressPCB Netlist"\r\n"do not touch me"\r\n'


@needs_ltspice
def test_reading_a_schematic_never_writes_to_its_directory(tmp_path):
    """Converting a .asc must not create or modify anything beside the original."""
    from spice_mcp_server.netlist import load_netlist

    asc = tmp_path / "good_lowpass.asc"
    shutil.copy2(FIXTURES / "good_lowpass.asc", asc)

    # A sibling .net, as LTspice's "Export Netlist" would leave behind.
    net = tmp_path / "good_lowpass.net"
    net.write_bytes(SENTINEL)

    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    netlist = load_netlist(asc)

    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert after == before, "reading a schematic modified files in the source directory"

    # The generated netlist must live somewhere else entirely.
    assert tmp_path not in Path(netlist.netlist_path).parents


@needs_ltspice
def test_simulating_never_writes_to_the_source_directory(tmp_path):
    asc = tmp_path / "good_lowpass.asc"
    shutil.copy2(FIXTURES / "good_lowpass.asc", asc)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    ltspice.run_batch(asc, timeout_s=90)

    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert after == before, "simulating wrote artifacts beside the user's schematic"


def test_work_dir_is_outside_the_source_directory(tmp_path):
    asc = tmp_path / "x.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")
    work = ltspice.work_dir_for(asc)
    assert tmp_path not in work.parents and work != tmp_path


def test_work_dir_is_stable_for_the_same_file(tmp_path):
    """Repeated reads must reuse one scratch dir rather than filling the temp folder."""
    asc = tmp_path / "x.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")
    assert ltspice.work_dir_for(asc) == ltspice.work_dir_for(asc)


def test_work_dir_differs_for_different_files(tmp_path):
    a, b = tmp_path / "a.asc", tmp_path / "b.asc"
    for p in (a, b):
        p.write_text("Version 4.1\n", encoding="utf-8")
    assert ltspice.work_dir_for(a) != ltspice.work_dir_for(b)


def test_ltspice_exe_override_is_validated(monkeypatch):
    monkeypatch.setenv("LTSPICE_EXE", r"C:\definitely\not\here\LTspice.exe")
    with pytest.raises(ltspice.LTSpiceNotFound, match="does not exist"):
        ltspice.find_ltspice_exe()


@needs_ltspice
def test_netlist_generation_places_output_in_scratch_dir(tmp_path):
    asc = tmp_path / "good_lowpass.asc"
    shutil.copy2(FIXTURES / "good_lowpass.asc", asc)
    produced = ltspice.generate_netlist(asc)
    assert produced.is_file()
    assert produced.suffix == ".net"
    assert produced.parent == ltspice.work_dir_for(asc)
    text = produced.read_text(encoding="utf-8", errors="replace")
    assert "R1" in text and ".ac" in text


def test_expresspcb_file_is_not_mistaken_for_spice(tmp_path):
    """An ExpressPCB .net with no sibling .asc must be rejected, not misparsed."""
    from spice_mcp_server.netlist import NetlistFormatError, load_netlist

    net = tmp_path / "orphan.net"
    net.write_bytes(SENTINEL)
    with pytest.raises(NetlistFormatError, match="ExpressPCB"):
        load_netlist(net)


@needs_ltspice
def test_expresspcb_falls_back_to_sibling_schematic(tmp_path):
    """Handed an ExpressPCB export, use the .asc beside it rather than failing.

    This is the repo's own RCLP.net/RCLP.asc pair: the .net is a PCB export, so the
    only simulatable truth is the schematic. netlist_path must report the *generated*
    netlist, not the ExpressPCB file we were handed.
    """
    from spice_mcp_server.netlist import load_netlist

    shutil.copy2(FIXTURES / "good_lowpass.asc", tmp_path / "good_lowpass.asc")
    net = tmp_path / "good_lowpass.net"
    net.write_bytes(SENTINEL)

    netlist = load_netlist(net)

    assert Path(netlist.source_path) == net
    assert Path(netlist.netlist_path) != net
    assert Path(netlist.netlist_path).parent == ltspice.work_dir_for(tmp_path / "good_lowpass.asc")
    assert [c.ref for c in netlist.components] == ["V1", "R1", "C1"]
    # And the ExpressPCB file we were handed is left exactly as it was.
    assert net.read_bytes() == SENTINEL


def test_work_dir_has_no_short_path_components(tmp_path):
    """8.3 short paths (C:\\Users\\EXPERT~1) leak into model-visible tool output."""
    asc = tmp_path / "x.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")
    assert "~" not in str(ltspice.work_dir_for(asc))
