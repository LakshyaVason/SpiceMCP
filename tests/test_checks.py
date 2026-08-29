"""Static-check tests.

The parametrised fixture matrix at the bottom is the important one: it asserts the
*exact* set of checks that fire for each circuit. Asserting equality rather than
membership is what makes it a false-positive guard - a new check that misfires on the
known-good circuit fails the suite instead of quietly degrading the tool.
"""

from __future__ import annotations

import pytest

from conftest import FIXTURES, check_ids, needs_ltspice, parse
from spice_mcp_server.checks import run_static_checks

GOOD_RC = """* good
V1 Vin 0 AC 1
R1 Vin Vout 1.6k
C1 Vout 0 100n
.ac dec 20 10 100k
.end
"""


def test_clean_circuit_produces_no_findings():
    assert check_ids(GOOD_RC) == set()


def test_result_ok_flag_tracks_error_severity_only():
    # A lone warning must not flip ok to False.
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 10M\n.ac dec 10 1 1k\n")
    assert "suspicious_suffix" in ids
    result = run_static_checks(parse("* t\nV1 a 0 AC 1\nR1 a 0 10M\n.ac dec 10 1 1k\n"))
    assert result.ok is True
    assert all(f.severity != "error" for f in result.findings)


def test_findings_sorted_most_severe_first():
    text = "* t\nV1 a 0 AC 1\nR1 a b 10M\n.ac dec 10 1 1k\n"
    findings = run_static_checks(parse(text)).findings
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s])


# --- individual checks -------------------------------------------------------------


def test_no_ground():
    assert "no_ground" in check_ids("* t\nV1 a b AC 1\nR1 a b 1k\n.ac dec 10 1 1k\n")


def test_no_ground_names_mislabelled_ground_net():
    """'GND' as a net label is a real mistake: only node 0 is the SPICE reference."""
    result = run_static_checks(parse("* t\nV1 a GND AC 1\nR1 a GND 1k\n.ac dec 10 1 1k\n"))
    finding = next(f for f in result.findings if f.check == "no_ground")
    assert "GND" in finding.message
    assert finding.nets == ["GND"]


def test_floating_net_single_connection():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a dangling 1k\n.ac dec 10 1 1k\n")
    assert "floating_net" in ids


def test_floating_pin_uses_ltspice_nc_naming():
    result = run_static_checks(parse("* t\nV1 a 0 AC 1\nR1 a NC_01 1k\n.ac dec 10 1 1k\n"))
    finding = next(f for f in result.findings if f.check == "floating_pin")
    assert finding.refs == ["R1"]
    assert "NC_01" in finding.message


def test_ground_itself_is_never_reported_as_floating():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 1k\n.ac dec 10 1 1k\n")
    assert "floating_net" not in ids and "floating_pin" not in ids


def test_floating_downgraded_when_pin_count_unknown():
    """An 'A' device hides pins, so a floating claim would be unsafe as an error."""
    text = "* t\nV1 a 0 AC 1\nR1 a b 1k\nA1 b c d e f g h SOMEFUNC\n.ac dec 10 1 1k\n"
    result = run_static_checks(parse(text))
    floating = [f for f in result.findings if f.check in {"floating_net", "floating_pin"}]
    assert floating, "expected a floating finding for net b"
    assert all(f.severity == "warning" for f in floating)


def test_duplicate_ref():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 1k\nR1 a 0 2k\n.ac dec 10 1 1k\n")
    assert "duplicate_ref" in ids


def test_missing_value():
    result = run_static_checks(parse("* t\nV1 a 0 AC 1\nR1 a 0\n.ac dec 10 1 1k\n"))
    assert any(f.check == "missing_value" and f.refs == ["R1"] for f in result.findings)


def test_empty_quoted_value_is_missing():
    """LTspice writes '""' into the netlist for a blank Value attribute."""
    ids = check_ids('* t\nV1 a 0 AC 1\nR1 a 0 ""\n.ac dec 10 1 1k\n')
    assert "missing_value" in ids


def test_malformed_value():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 ten_ohms\n.ac dec 10 1 1k\n")
    assert "malformed_value" in ids


@pytest.mark.parametrize("value", ["10k", "1.6k", "4.7u", "100n", "1µ", "2.2Meg", "1e3", "0.5"])
def test_valid_values_not_flagged(value):
    ids = check_ids(f"* t\nV1 a 0 AC 1\nR1 a 0 {value}\n.ac dec 10 1 1k\n")
    assert "malformed_value" not in ids


@pytest.mark.parametrize("value", ["{Rload}", "R=1k", "1k tol=5"])
def test_expression_values_not_flagged(value):
    ids = check_ids(f"* t\nV1 a 0 AC 1\nR1 a 0 {value}\n.ac dec 10 1 1k\n")
    assert "malformed_value" not in ids


def test_suspicious_m_suffix_on_resistor():
    """'10M' is 10 milliohms in SPICE. Nothing errors; the answer is just wrong."""
    result = run_static_checks(parse("* t\nV1 a 0 AC 1\nR1 a 0 10M\n.ac dec 10 1 1k\n"))
    finding = next(f for f in result.findings if f.check == "suspicious_suffix")
    assert "milli" in finding.message
    assert "Meg" in (finding.suggestion or "")


def test_meg_suffix_is_not_suspicious():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 10Meg\n.ac dec 10 1 1k\n")
    assert "suspicious_suffix" not in ids


def test_milliohm_capacitor_not_flagged():
    """The M trap applies to R and L. '1m' on a capacitor is a normal 1 mF."""
    ids = check_ids("* t\nV1 a 0 AC 1\nC1 a 0 1m\n.ac dec 10 1 1k\n")
    assert "suspicious_suffix" not in ids


def test_suspicious_m_suffix_in_directive_frequency():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 1k\n.ac dec 10 1 10M\n")
    assert "suspicious_suffix" in ids


def test_no_analysis_directive():
    ids = check_ids("* t\nV1 a 0 AC 1\nR1 a 0 1k\n")
    assert "no_analysis_directive" in ids


def test_empty_circuit():
    assert "empty_circuit" in check_ids("* nothing here\n.end\n")


def test_isolated_section_without_ground():
    """A second, ungrounded island of circuit cannot be solved."""
    text = """* t
V1 a 0 AC 1
R1 a 0 1k
R2 x y 1k
R3 x y 2k
.ac dec 10 1 1k
"""
    result = run_static_checks(parse(text))
    finding = next(f for f in result.findings if f.check == "isolated_section")
    assert set(finding.nets) == {"x", "y"}


def test_no_dc_path_to_ground():
    """Capacitor-isolated nodes are the classic non-convergence cause."""
    text = """* t
V1 Vin 0 PULSE(0 5 0 1u 1u 1m 2m)
C1 Vin Vmid 1u
R1 Vmid Vout 10k
C2 Vout 0 100n
.op
"""
    result = run_static_checks(parse(text))
    finding = next(f for f in result.findings if f.check == "no_dc_path_to_ground")
    assert set(finding.nets) == {"Vmid", "Vout"}


def test_dc_path_through_resistor_is_fine():
    text = """* t
V1 Vin 0 PULSE(0 5 0 1u 1u 1m 2m)
C1 Vin Vmid 1u
R1 Vmid 0 1Meg
.op
"""
    assert "no_dc_path_to_ground" not in check_ids(text)


def test_dc_path_check_skipped_without_dc_analysis():
    """A pure .ac sweep with no operating point should not trigger the DC check."""
    text = "* t\nV1 a 0 AC 1\nC1 a b 1u\nC2 b 0 1u\n.four 1k V(b)\n"
    assert "no_dc_path_to_ground" not in check_ids(text)


# --- fixture matrix ----------------------------------------------------------------

# The exact set of checks each fixture must produce. Equality, not membership.
#
# Note the two deliberately empty entries. source_conflict.asc is statically clean but
# fails to simulate; wrong_value_lowpass.asc is statically clean and simulates fine but
# is out of spec. Neither is something static analysis can or should catch, so they
# double as false-positive guards.
EXPECTED: dict[str, set[str]] = {
    "good_lowpass.asc": set(),
    "wrong_value_lowpass.asc": set(),
    "source_conflict.asc": set(),
    "floating_node.asc": {"floating_pin"},
    "missing_ground.asc": {"no_ground"},
    "no_dc_path.asc": {"no_dc_path_to_ground"},
}


@needs_ltspice
@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_fixture_produces_exactly_expected_checks(name, expected):
    from spice_mcp_server.netlist import load_netlist

    result = run_static_checks(load_netlist(FIXTURES / name))
    assert {f.check for f in result.findings} == expected


@needs_ltspice
def test_subtly_broken_fixture_is_statically_clean():
    """The whole point of wrong_value_lowpass: static analysis cannot catch it.

    If this ever starts failing, a check has become value-judgemental and will
    misfire on legitimate designs.
    """
    from spice_mcp_server.netlist import load_netlist

    result = run_static_checks(load_netlist(FIXTURES / "wrong_value_lowpass.asc"))
    assert result.ok is True
    assert result.findings == []


@needs_ltspice
def test_repo_rclp_schematic_finds_both_real_bugs():
    """RCLP.asc ships broken two ways: R1 dangles and Vin drives nothing."""
    from conftest import REPO_ROOT
    from spice_mcp_server.netlist import load_netlist

    result = run_static_checks(load_netlist(REPO_ROOT / "RCLP.asc"))
    assert {f.check for f in result.findings} == {"floating_net", "floating_pin"}
    assert result.ok is False
