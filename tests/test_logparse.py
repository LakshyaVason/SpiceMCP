"""Log parser tests.

Every log sample below is real LTspice 26.0.2 output, captured by running deliberately
broken decks. Nothing here is invented, because the whole value of the parser is that it
matches what LTspice actually prints - and the messages are inconsistent enough that
plausible-looking fabricated samples would test the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FIXTURES, needs_ltspice
from spice_mcp_server.logparse import parse_log_text

CLEAN = r"""LTspice 26.0.2 for Windows
Circuit: X:\good_lowpass.net
Start Time: Sat Aug 29 13:59:26 2026
solver = Normal
Maximum thread count: 24
tnom = 27
temp = 27
method = trap
.OP point found by inspection.
.OP point found by inspection.
Total elapsed time: 0.022 seconds.

Files loaded:
X:\good_lowpass.net
"""

SINGULAR = r"""LTspice 26.0.2 for Windows
Circuit: X:\missing_ground.net
solver = Normal
method = trap
WARNING: Node n001 is floating.

.OP point found by inspection.

Simulation Failed: Matrix is singular

Total elapsed time: 0.015 seconds.
"""

OVER_DEFINED = r"""LTspice 26.0.2 for Windows
Circuit: X:\source_conflict.net
method = trap
Total elapsed time: 0.000 seconds.
Voltage source V2 and voltage source V1 are paralleled making an over-defined circuit matrix.
You will need to correct the circuit or add some series resistance.
"""

UNDEFINED_MODEL = r"""LTspice 26.0.2 for Windows
Circuit: X:\nomodel.cir
Start Time: Sat Aug 29 14:01:33 2026
C:\Users\Expertician\Temp\nomodel.cir(3): Undefined model "nosuchtransistor".
Q1 c b 0 NOSUCHTRANSISTOR
         ^^^^^^^^^^^^^^^^
"""

FLOATING_ISRC = r"""LTspice 26.0.2 for Windows
Circuit: X:\isrc.cir
method = trap
ERROR: Node n1 is floating and connected to current source I1
Direct Newton iteration succeeded in finding operating point.
Total elapsed time: 0.01 seconds.
"""

TIMESTEP = r"""LTspice 26.0.2 for Windows
Circuit: X:\tstep.cir
method = trap
Analysis: Time step too small; time = 1.2e-9, timestep = 1e-20
Total elapsed time: 3.0 seconds.
"""

MISSING_LIB = r"""LTspice 26.0.2 for Windows
Circuit: X:\p.cir
Could not open library file opamp.lib
Total elapsed time: 0.0 seconds.
"""


def parse(text: str):
    return parse_log_text(text, Path("test.log"))


def ids(text: str) -> set[str]:
    return {f.check for f in parse(text).findings}


# --- success detection -------------------------------------------------------------


def test_clean_log_succeeds_with_no_findings():
    result = parse(CLEAN)
    assert result.succeeded is True
    assert result.findings == []
    assert result.summary == "Simulation completed."


def test_header_fields_extracted():
    result = parse(CLEAN)
    assert result.ltspice_version == "LTspice 26.0.2 for Windows"
    assert result.circuit == r"X:\good_lowpass.net"
    assert result.elapsed_s == pytest.approx(0.022)
    assert result.solver == "Normal"
    assert result.method == "trap"


def test_files_loaded_block_captured():
    assert parse(CLEAN).files_loaded == [r"X:\good_lowpass.net"]


@pytest.mark.parametrize(
    "text", [SINGULAR, OVER_DEFINED, UNDEFINED_MODEL, FLOATING_ISRC, TIMESTEP, MISSING_LIB]
)
def test_every_failure_sample_is_reported_as_failed(text):
    """The one property that must never regress: a failed run must not read as success."""
    assert parse(text).succeeded is False


# --- specific diagnoses ------------------------------------------------------------


def test_singular_matrix_recognised_and_explained():
    result = parse(SINGULAR)
    finding = next(f for f in result.findings if f.check == "singular_matrix")
    assert finding.severity == "error"
    assert "no conductive path to ground" in finding.message
    assert finding.suggestion and "node 0" in finding.suggestion


def test_singular_matrix_log_also_keeps_the_floating_warning():
    """Both lines are informative: the warning names the actual net."""
    result = parse(SINGULAR)
    warning = next(f for f in result.findings if f.check == "floating_node")
    assert warning.nets == ["n001"]
    assert warning.severity == "warning"


def test_over_defined_matrix_names_both_sources():
    """This message carries no ERROR: prefix, so a prefix-only parser would miss it."""
    finding = next(f for f in parse(OVER_DEFINED).findings if f.check == "over_defined_matrix")
    assert set(finding.refs) == {"V1", "V2"}
    assert finding.severity == "error"


def test_undefined_model_extracts_netlist_line_number():
    finding = next(f for f in parse(UNDEFINED_MODEL).findings if f.check == "undefined_model")
    assert finding.line_no == 3
    assert "nosuchtransistor" in finding.message


def test_undefined_model_aborts_before_elapsed_time():
    """No 'Total elapsed time' line at all - the run never started."""
    result = parse(UNDEFINED_MODEL)
    assert result.elapsed_s is None
    assert result.succeeded is False


def test_floating_current_source_is_error_despite_ltspice_exit_zero():
    finding = next(
        f for f in parse(FLOATING_ISRC).findings if f.check == "floating_current_source"
    )
    assert finding.severity == "error"
    assert finding.nets == ["n1"]
    assert finding.refs == ["I1"]


def test_floating_current_source_not_double_reported():
    """The specific pattern must claim the line so the generic ERROR: pass skips it."""
    checks = [f.check for f in parse(FLOATING_ISRC).findings]
    assert checks.count("floating_current_source") == 1
    assert "ltspice_error" not in checks


def test_timestep_collapse_classified_as_convergence():
    finding = next(f for f in parse(TIMESTEP).findings if f.check == "convergence_failure")
    assert "timestep collapsed" in finding.message
    # The advice must point at the circuit, not at loosening solver tolerances.
    assert "reltol" in (finding.suggestion or "")


def test_missing_library_file_reported():
    finding = next(f for f in parse(MISSING_LIB).findings if f.check == "missing_file")
    assert "opamp.lib" in finding.message


def test_summary_does_not_truncate_at_a_filename_dot():
    """'opamp.lib' must not be cut to 'opamp.' when building the one-line summary."""
    assert parse(MISSING_LIB).summary.endswith("opamp.lib.")


def test_findings_sorted_error_before_warning():
    severities = [f.severity for f in parse(SINGULAR).findings]
    assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s])


def test_unrecognised_error_still_surfaces_generically():
    text = "LTspice 26.0.2 for Windows\nERROR: Something entirely new went wrong\n"
    result = parse(text)
    finding = next(f for f in result.findings if f.check == "ltspice_error")
    assert "entirely new" in finding.message


def test_empty_log_is_not_silently_successful():
    result = parse("")
    assert result.succeeded is False
    assert any(f.check == "aborted" for f in result.findings)


# --- against the real simulator ----------------------------------------------------


@needs_ltspice
def test_source_conflict_fixture_really_fails_to_simulate():
    """The fixture that must fail at simulation time, not static-check time.

    Also pins the two behaviours that make exit codes and .raw files untrustworthy.
    """
    from spice_mcp_server.logparse import read_sim_log
    from spice_mcp_server.ltspice import run_batch

    batch = run_batch(FIXTURES / "source_conflict.asc", timeout_s=90)
    result = read_sim_log(batch.log_path)

    assert result.succeeded is False
    assert {f.check for f in result.findings} == {"over_defined_matrix"}
    # LTspice writes a waveform file even though the analysis failed.
    assert batch.raw_path is not None


@needs_ltspice
def test_no_dc_path_fixture_simulates_but_warns():
    """Documents the surprise: LTspice accepts this circuit and exits 0.

    The static check is the only thing that flags it, which is the argument for running
    check_netlist_static before spending time on a simulation.
    """
    from spice_mcp_server.logparse import read_sim_log
    from spice_mcp_server.ltspice import run_batch

    batch = run_batch(FIXTURES / "no_dc_path.asc", timeout_s=90)
    result = read_sim_log(batch.log_path)

    assert batch.returncode == 0
    assert result.succeeded is True
    assert {f.check for f in result.findings} == {"floating_node"}


@needs_ltspice
def test_good_fixture_simulates_cleanly():
    from spice_mcp_server.logparse import read_sim_log
    from spice_mcp_server.ltspice import run_batch

    batch = run_batch(FIXTURES / "good_lowpass.asc", timeout_s=90)
    result = read_sim_log(batch.log_path)
    assert result.succeeded is True
    assert result.findings == []


@needs_ltspice
def test_measure_results_are_read_back():
    """.measure output sits after the 'Files loaded' block and needs spicelib to read."""
    import tempfile

    from spice_mcp_server.logparse import read_sim_log
    from spice_mcp_server.ltspice import run_batch

    deck = """measure probe
V1 in 0 AC 1
R1 in out 1.6k
C1 out 0 100n
.ac dec 20 10 100k
.measure AC vmax MAX V(out)
.end
"""
    work = Path(tempfile.gettempdir()) / "spice_mcp_tests"
    work.mkdir(exist_ok=True)
    deck_path = work / "measure_probe.cir"
    deck_path.write_text(deck, encoding="utf-8")

    batch = run_batch(deck_path, timeout_s=90)
    result = read_sim_log(batch.log_path)

    assert result.succeeded is True
    assert "vmax" in {m.name for m in result.measurements}
