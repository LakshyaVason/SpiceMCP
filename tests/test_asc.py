"""Tests for byte-preserving `.asc` patching.

The acceptance bar for write-back is not "produces a valid schematic" but "changes
exactly one line and leaves every other byte alone". A patch that silently renormalised
line endings would still open in LTspice while making the review diff useless and
polluting git history, so the encoding and terminator tests below are the real contract.

Two format traps get their own tests because both are present in this repo:
`SYMATTR Value2` (a different attribute that a prefix match would clobber) and a
component with no `SYMATTR Value` line at all.
"""

from __future__ import annotations

import shutil

import pytest

from conftest import FIXTURES, needs_ltspice
from spice_mcp_server.asc import (
    AscError,
    locate_value,
    patch_component_value,
    read_asc,
)

REPO_ROOT = FIXTURES.parent


def copy_fixture(tmp_path, name, source=None):
    target = tmp_path / name
    shutil.copy2(source or (FIXTURES / name), target)
    return target


# --- encoding and terminator detection ----------------------------------------------


def test_reads_bare_lf_without_bom(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    parsed = read_asc(asc)
    assert parsed.encoding == "utf-8"
    assert parsed.to_bytes() == asc.read_bytes(), "round trip changed the bytes"


def test_round_trip_preserves_crlf_and_bom(tmp_path):
    original = (FIXTURES / "wrong_value_lowpass.asc").read_text(encoding="utf-8")
    crlf = tmp_path / "crlf.asc"
    crlf.write_bytes(b"\xef\xbb\xbf" + original.replace("\n", "\r\n").encode("utf-8"))

    parsed = read_asc(crlf)
    assert parsed.encoding == "utf-8-sig"
    assert parsed.to_bytes() == crlf.read_bytes()


# --- locating the right line ---------------------------------------------------------


def test_value2_is_not_mistaken_for_value(tmp_path):
    """RCLP.asc's V1 has both `SYMATTR Value ""` and `SYMATTR Value2 AC 0.7 3000`."""
    asc = copy_fixture(tmp_path, "RCLP.asc", REPO_ROOT / "RCLP.asc")
    parsed = read_asc(asc)
    location = locate_value(parsed, "V1")

    line = parsed.lines[location.value_index].strip()
    assert line.split()[1] == "Value", f"matched the wrong attribute: {line!r}"
    assert location.current_value == '""'


def test_value_is_scoped_to_its_own_symbol_block(tmp_path):
    asc = copy_fixture(tmp_path, "RCLP.asc", REPO_ROOT / "RCLP.asc")
    parsed = read_asc(asc)

    # Each component must report its own value, not a neighbouring block's.
    assert locate_value(parsed, "C1").current_value == "6.63n"
    assert locate_value(parsed, "R2").current_value == "15k"
    assert locate_value(parsed, "R1").current_value == "10k"


def test_unknown_ref_lists_what_is_present(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    with pytest.raises(AscError) as excinfo:
        locate_value(read_asc(asc), "R99")

    message = str(excinfo.value)
    assert "R99" in message
    # The error is the diagnosis the model acts on, so it must name the alternatives.
    assert "C1" in message and "R1" in message


def test_ref_matching_is_case_insensitive(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    assert locate_value(read_asc(asc), "c1").current_value == "1u"


# --- the byte-fidelity bar ------------------------------------------------------------


def test_patch_changes_exactly_one_line(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    before = asc.read_bytes().decode("utf-8").splitlines(keepends=True)

    outcome = patch_component_value(asc, "C1", "100n", apply=True)

    after = asc.read_bytes().decode("utf-8").splitlines(keepends=True)
    assert len(after) == len(before), "line count changed"

    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert differing == [outcome.line_no - 1], f"expected one changed line, got {differing}"
    assert after[differing[0]].strip() == "SYMATTR Value 100n"


def test_patch_preserves_encoding_and_line_endings(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    raw_before = asc.read_bytes()

    patch_component_value(asc, "C1", "100n", apply=True)
    raw_after = asc.read_bytes()

    assert not raw_after.startswith(b"\xef\xbb\xbf"), "a BOM was added"
    assert raw_after.count(b"\r\n") == 0, "CRLF was introduced into a bare-LF file"
    assert raw_after.count(b"\n") == raw_before.count(b"\n")
    assert raw_after.endswith(b"\n") == raw_before.endswith(b"\n")


def test_patch_preserves_crlf_and_bom(tmp_path):
    """The mirror case: a CRLF/BOM file must not be downgraded to bare LF."""
    original = (FIXTURES / "wrong_value_lowpass.asc").read_text(encoding="utf-8")
    asc = tmp_path / "crlf.asc"
    asc.write_bytes(b"\xef\xbb\xbf" + original.replace("\n", "\r\n").encode("utf-8"))
    lf_count = asc.read_bytes().count(b"\n")

    patch_component_value(asc, "C1", "100n", apply=True)
    raw = asc.read_bytes()

    assert raw.startswith(b"\xef\xbb\xbf"), "the BOM was dropped"
    assert raw.count(b"\r\n") == lf_count, "a CRLF line ending was lost"
    assert b"SYMATTR Value 100n\r\n" in raw


def test_dry_run_is_the_default(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    raw_before = asc.read_bytes()

    outcome = patch_component_value(asc, "C1", "100n")

    assert asc.read_bytes() == raw_before, "the default call wrote to disk"
    assert outcome.applied is False
    assert outcome.old_value == "1u"
    assert outcome.new_value == "100n"
    # The diff is what the user reviews before approving, so it must be populated.
    assert "-SYMATTR Value 1u" in outcome.diff
    assert "+SYMATTR Value 100n" in outcome.diff


def test_patch_reports_the_previous_value(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    outcome = patch_component_value(asc, "C1", "100n", apply=True)
    assert outcome.old_value == "1u"
    assert outcome.inserted is False
    assert outcome.applied is True


def test_patching_a_blank_value_rewrites_only_that_line(tmp_path):
    """RCLP.asc's V1 has an empty value; filling it must not disturb Value2."""
    asc = copy_fixture(tmp_path, "RCLP.asc", REPO_ROOT / "RCLP.asc")
    before = asc.read_text(encoding="utf-8").splitlines()

    patch_component_value(asc, "V1", "AC 1", apply=True)

    after = asc.read_text(encoding="utf-8").splitlines()
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert after[differing[0]] == "SYMATTR Value AC 1"
    assert "SYMATTR Value2 AC 0.7 3000" in after, "Value2 was damaged"


def test_missing_value_line_is_inserted_after_instname(tmp_path):
    asc = tmp_path / "novalue.asc"
    asc.write_bytes(
        b"Version 4.1\n"
        b"SHEET 1 880 680\n"
        b"SYMBOL res 192 -48 R90\n"
        b"SYMATTR InstName R1\n"
        b"SYMBOL cap 256 -32 R0\n"
        b"SYMATTR InstName C1\n"
        b"SYMATTR Value 1u\n"
    )

    outcome = patch_component_value(asc, "R1", "1.6k", apply=True)

    assert outcome.inserted is True
    assert outcome.old_value is None
    lines = asc.read_text(encoding="utf-8").splitlines()
    assert lines[3:5] == ["SYMATTR InstName R1", "SYMATTR Value 1.6k"]
    # The other block must be untouched, not merged into.
    assert lines[5:] == ["SYMBOL cap 256 -32 R0", "SYMATTR InstName C1", "SYMATTR Value 1u"]


def test_empty_new_value_is_rejected(tmp_path):
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    with pytest.raises(AscError):
        patch_component_value(asc, "C1", "   ", apply=True)


# --- does the patched file still work? -----------------------------------------------


@needs_ltspice
def test_patched_schematic_still_converts_and_simulates(tmp_path):
    """The patched file must remain a schematic LTspice accepts, not just valid text.

    Converting via -netlist and simulating is the closest automatable proxy for "still
    opens in the GUI"; both go through the same schematic parser LTspice uses.
    """
    from spice_mcp_server.netlist import load_netlist
    from spice_mcp_server.ltspice import run_batch
    from spice_mcp_server.logparse import read_sim_log

    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    patch_component_value(asc, "C1", "100n", apply=True)

    netlist = load_netlist(asc)
    capacitor = next(c for c in netlist.components if c.ref == "C1")
    assert capacitor.value == "100n", "LTspice did not read back the patched value"

    batch = run_batch(asc, timeout_s=60)
    assert batch.log_path is not None
    assert read_sim_log(batch.log_path).succeeded


@needs_ltspice
def test_patching_does_not_write_anything_else_into_the_folder(tmp_path):
    """Patching is pure text editing - it must not invoke LTspice or leave artifacts."""
    asc = copy_fixture(tmp_path, "wrong_value_lowpass.asc")
    sibling = tmp_path / "wrong_value_lowpass.net"
    sibling.write_bytes(b'"ExpressPCB Netlist"\r\n"do not touch me"\r\n')

    patch_component_value(asc, "C1", "100n", apply=True)

    assert sibling.read_bytes() == b'"ExpressPCB Netlist"\r\n"do not touch me"\r\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "wrong_value_lowpass.asc",
        "wrong_value_lowpass.net",
    ]
