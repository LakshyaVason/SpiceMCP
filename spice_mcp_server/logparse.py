"""Parsing LTspice simulation logs.

The log is where the real diagnosis lives. A netlist can be topologically perfect and
still fail to solve, and LTspice's own explanation of *why* only ever appears here.

Every pattern below was taken from actual LTspice 26.0.2 output captured by running
deliberately broken decks, not from documentation. That matters, because the messages
do not share a common shape:

    WARNING: Node n001 is floating.
    ERROR: Node n1 is floating and connected to current source I1
    Simulation Failed: Matrix is singular
    Voltage source V2 and voltage source V1 are paralleled making an over-defined
    circuit matrix.
    C:\\path\\deck.cir(3): Undefined model "nosuchtransistor".

Only two of those five carry a severity prefix, and the last carries a line number in a
compile-style `(N):` form followed by the offending source line and a caret marker. So a
plain "grep for ERROR" pass would miss the two most serious failures outright.

Two further traps, both verified empirically:

  * **The exit code lies.** `ERROR: Node n1 is floating and connected to current source`
    exits 0. A missing ground exits 1. Success must come from the log text.
  * **A .raw file is written even for failed runs.** The singular-matrix failure still
    produced a complete-looking .raw, so its existence proves nothing either.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import Finding, Measurement, SimLog

log = logging.getLogger(__name__)

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

# --- header fields -----------------------------------------------------------------

_RE_VERSION = re.compile(r"^(LTspice .*?)\s*$", re.MULTILINE)
_RE_CIRCUIT = re.compile(r"^Circuit:\s*(.+?)\s*$", re.MULTILINE)
_RE_ELAPSED = re.compile(r"^Total elapsed time:\s*([0-9.]+)\s*seconds", re.MULTILINE)
_RE_SOLVER = re.compile(r"^solver\s*=\s*(\S+)", re.MULTILINE)
_RE_METHOD = re.compile(r"^method\s*=\s*(\S+)", re.MULTILINE)

# --- diagnostics -------------------------------------------------------------------

# A floating node connected to a current source is unsolvable in principle: there is
# nowhere for the current to go. LTspice calls this ERROR but still exits 0.
_RE_FLOAT_ISRC = re.compile(
    r"^ERROR:\s*Node\s+(?P<net>\S+)\s+is floating and connected to current source\s+(?P<ref>\S+)",
    re.IGNORECASE | re.MULTILINE,
)
_RE_FLOATING = re.compile(
    r"^WARNING:\s*Node\s+(?P<net>\S+)\s+is floating\.?", re.IGNORECASE | re.MULTILINE
)
_RE_SIM_FAILED = re.compile(r"^Simulation Failed:\s*(?P<reason>.+?)\s*$", re.MULTILINE)
_RE_OVERDEFINED = re.compile(
    r"^(?P<msg>.*?\bparalleled making an over-defined circuit matrix\.?)", re.MULTILINE
)
# Compile-style: <path>(<line>): <message>, then the source line, then a caret marker.
_RE_COMPILE_ERR = re.compile(
    r"^(?P<path>[A-Za-z]:\\[^\n(]+|/[^\n(]+)\((?P<line>\d+)\):\s*(?P<msg>.+?)\s*$",
    re.MULTILINE,
)
_RE_UNDEFINED_MODEL = re.compile(r'Undefined model\s+"?(?P<model>[^"\n]+?)"?\.?\s*$', re.IGNORECASE)

# Convergence family. LTspice phrases these several ways depending on the analysis.
_CONVERGENCE_PATTERNS = (
    (re.compile(r"time step too small", re.IGNORECASE), "The transient timestep collapsed."),
    (re.compile(r"iteration limit", re.IGNORECASE), "The solver hit its iteration limit."),
    (re.compile(r"failed to converge", re.IGNORECASE), "The solver failed to converge."),
    (re.compile(r"convergence (?:failure|problem)", re.IGNORECASE), "The solver did not converge."),
    (re.compile(r"gmin stepping failed", re.IGNORECASE), "Gmin stepping failed."),
    (
        re.compile(r"source stepping failed", re.IGNORECASE),
        "Source stepping failed to find an operating point.",
    ),
)

_RE_MISSING_FILE = re.compile(
    r"^(?:ERROR:\s*)?(?:Could not open|Can't find|Cannot find|Failed to open)\s+(?P<what>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Anything ERROR:/WARNING: prefixed that the specific patterns above did not claim.
_RE_GENERIC = re.compile(r"^(?P<sev>ERROR|WARNING|FATAL):\s*(?P<msg>.+?)\s*$", re.MULTILINE)

# Categories that mean the analysis did not produce trustworthy results.
_FATAL_CATEGORIES = {
    "singular_matrix",
    "over_defined_matrix",
    "undefined_model",
    "convergence_failure",
    "simulation_failed",
    "missing_file",
    "netlist_error",
    "floating_current_source",
    "timed_out",
    "aborted",
}


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop duplicate findings, keeping the first (most specific) of each."""
    seen: set[tuple[str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.check, f.message)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _line_of(text: str, offset: int) -> int:
    """1-indexed line number within `text` for a character offset."""
    return text.count("\n", 0, offset) + 1


def parse_log_text(text: str, log_path: Path | str) -> SimLog:
    """Parse LTspice log text into a structured result."""
    findings: list[Finding] = []
    claimed: set[int] = set()  # log line numbers already explained by a specific pattern

    def claim(match: re.Match) -> int:
        ln = _line_of(text, match.start())
        claimed.add(ln)
        return ln

    # --- fatal solver failures ---
    for m in _RE_SIM_FAILED.finditer(text):
        claim(m)
        reason = m.group("reason")
        lowered = reason.lower()
        if "singular" in lowered:
            findings.append(
                Finding(
                    check="singular_matrix",
                    severity="error",
                    message=(
                        f"LTspice could not solve the circuit: {reason}. A singular matrix "
                        "means at least one node has no conductive path to ground, so its "
                        "voltage is undefined."
                    ),
                    suggestion=(
                        "Check that a ground symbol (node 0) is present and that every "
                        "section of the circuit reaches it through something other than a "
                        "capacitor. Run check_netlist_static for the specific nets."
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    check="simulation_failed",
                    severity="error",
                    message=f"LTspice reported: {reason}",
                )
            )

    for m in _RE_OVERDEFINED.finditer(text):
        claim(m)
        msg = m.group("msg").strip()
        refs = re.findall(r"\bvoltage source\s+(\S+)", msg, flags=re.IGNORECASE)
        findings.append(
            Finding(
                check="over_defined_matrix",
                severity="error",
                message=(
                    f"{msg} Two ideal sources in parallel disagree about the same node "
                    "voltage, so there is no solution."
                ),
                refs=refs,
                suggestion=(
                    "Remove one source, or add a small series resistance to one of them."
                ),
            )
        )

    # --- netlist-level errors, which carry a source line number ---
    for m in _RE_COMPILE_ERR.finditer(text):
        claim(m)
        msg = m.group("msg").strip()
        src_line = int(m.group("line"))
        model = _RE_UNDEFINED_MODEL.search(msg)
        if model:
            name = model.group("model")
            findings.append(
                Finding(
                    check="undefined_model",
                    severity="error",
                    message=(
                        f"Netlist line {src_line} references model {name!r}, which is not "
                        "defined anywhere LTspice looked."
                    ),
                    line_no=src_line,
                    suggestion=(
                        f"Add a .model statement for {name!r}, or a .include/.lib pointing at "
                        "the file that defines it. Check the spelling of the model name."
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    check="netlist_error",
                    severity="error",
                    message=f"Netlist line {src_line}: {msg}",
                    line_no=src_line,
                )
            )

    # --- convergence family ---
    for pattern, explanation in _CONVERGENCE_PATTERNS:
        for m in pattern.finditer(text):
            ln = _line_of(text, m.start())
            claimed.add(ln)
            line_text = text.splitlines()[ln - 1].strip() if ln <= len(text.splitlines()) else ""
            findings.append(
                Finding(
                    check="convergence_failure",
                    severity="error",
                    message=f"{explanation} LTspice said: {line_text or m.group(0)}",
                    suggestion=(
                        "Convergence failures usually come from an unrealistic model or an "
                        "impossible operating point rather than from the solver settings. "
                        "Look for missing series resistance, a source with an implausible "
                        "value or polarity, or a node with no DC path to ground before "
                        "reaching for .options reltol/gmin."
                    ),
                )
            )

    # --- floating nodes ---
    for m in _RE_FLOAT_ISRC.finditer(text):
        claim(m)
        net, ref = m.group("net"), m.group("ref")
        findings.append(
            Finding(
                check="floating_current_source",
                severity="error",
                message=(
                    f"Net {net!r} is driven by current source {ref} but has no other "
                    "connection, so the current has nowhere to flow and the node voltage "
                    "is undefined. LTspice reports this as an error but still exits 0."
                ),
                refs=[ref],
                nets=[net],
                suggestion=f"Add a resistive path from {net!r} to ground.",
            )
        )

    for m in _RE_FLOATING.finditer(text):
        ln = _line_of(text, m.start())
        if ln in claimed:
            continue
        claimed.add(ln)
        net = m.group("net")
        findings.append(
            Finding(
                check="floating_node",
                severity="warning",
                message=(
                    f"LTspice reports net {net!r} as floating. It has no DC path to ground, "
                    "so its operating point had to be guessed."
                ),
                nets=[net],
                suggestion=(
                    f"Give {net!r} a DC path to ground - often a large resistor is enough if "
                    "the node is only meant to be AC-coupled."
                ),
            )
        )

    for m in _RE_MISSING_FILE.finditer(text):
        ln = _line_of(text, m.start())
        if ln in claimed:
            continue
        claimed.add(ln)
        findings.append(
            Finding(
                check="missing_file",
                severity="error",
                message=f"LTspice could not open a referenced file: {m.group('what')}",
                suggestion=(
                    "Check the .include/.lib path. Relative paths resolve against the "
                    "circuit file's directory."
                ),
            )
        )

    # --- anything else that announced itself as an error or warning ---
    for m in _RE_GENERIC.finditer(text):
        ln = _line_of(text, m.start())
        if ln in claimed:
            continue
        claimed.add(ln)
        sev = "error" if m.group("sev").upper() in {"ERROR", "FATAL"} else "warning"
        findings.append(
            Finding(
                check="ltspice_error" if sev == "error" else "ltspice_warning",
                severity=sev,
                message=m.group("msg").strip(),
            )
        )

    findings = _dedupe(findings)

    # --- header/telemetry ---
    version = _RE_VERSION.search(text)
    circuit = _RE_CIRCUIT.search(text)
    elapsed = _RE_ELAPSED.search(text)
    solver = _RE_SOLVER.search(text)
    method = _RE_METHOD.search(text)

    # No "Total elapsed time" line means LTspice gave up before running the analysis -
    # that is how an undefined-model abort presents.
    if elapsed is None and not any(f.severity == "error" for f in findings):
        findings.append(
            Finding(
                check="aborted",
                severity="error",
                message=(
                    "The log has no 'Total elapsed time' line, so LTspice exited before "
                    "completing the analysis, without explaining why."
                ),
            )
        )

    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))

    fatal = [f for f in findings if f.check in _FATAL_CATEGORIES]
    succeeded = not fatal and elapsed is not None

    measurements, step_count = _read_measurements(log_path)
    files_loaded = _files_loaded(text)

    return SimLog(
        log_path=str(log_path),
        ltspice_version=version.group(1).strip() if version else None,
        circuit=circuit.group(1).strip() if circuit else None,
        succeeded=succeeded,
        elapsed_s=float(elapsed.group(1)) if elapsed else None,
        solver=solver.group(1) if solver else None,
        method=method.group(1) if method else None,
        findings=findings,
        measurements=measurements,
        step_count=step_count,
        files_loaded=files_loaded,
        summary=_summarise(succeeded, findings, measurements),
        raw_text=text,
    )


def _files_loaded(text: str) -> list[str]:
    """Extract the 'Files loaded:' block, which is useful for .include debugging."""
    marker = "Files loaded:"
    idx = text.find(marker)
    if idx == -1:
        return []
    out: list[str] = []
    for line in text[idx + len(marker) :].splitlines():
        stripped = line.strip()
        if not stripped:
            # A blank line ends the block; .measure output follows it.
            if out:
                break
            continue
        if ":" in stripped and not re.match(r"^[A-Za-z]:[\\/]", stripped):
            break  # a new labelled section, not a path
        out.append(stripped)
    return out


def _read_measurements(log_path: Path | str) -> tuple[list[Measurement], int]:
    """Read .measure results via spicelib, which understands LTspice's odd formats.

    AC measurements come back as '(-0.0004dB,-0.57 deg)' rather than a plain number, and
    WHEN measurements split into a value plus a companion '<name>_at' entry. Rather than
    reimplement that, lean on spicelib and keep the values as text.
    """
    path = Path(log_path)
    if not path.is_file():
        return [], 0
    try:
        from spicelib.log.ltsteps import LTSpiceLogReader

        reader = LTSpiceLogReader(str(path))
        names = reader.get_measure_names()
        out = [
            Measurement(name=n, value=str(reader.get_measure_value(n)))
            for n in names
        ]
        return out, int(getattr(reader, "step_count", 0) or 0)
    except Exception:
        # A log with no .measure directives, or one truncated by a crash, is normal.
        log.debug("spicelib could not read measurements from %s", path, exc_info=True)
        return [], 0


def _summarise(
    succeeded: bool, findings: list[Finding], measurements: list[Measurement]
) -> str:
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")

    if succeeded:
        parts = ["Simulation completed"]
        if measurements:
            parts.append(f"with {len(measurements)} measurement(s)")
        if warnings:
            parts.append(f"but produced {warnings} warning(s)")
        return " ".join(parts) + "."

    if errors:
        return f"Simulation failed: {_first_sentence(findings[0].message)}"
    return "Simulation did not complete successfully."


def _first_sentence(message: str, limit: int = 160) -> str:
    """First sentence of a finding, for the one-line summary.

    Splitting on a bare '.' would cut 'opamp.lib' in half, so require the period to be
    followed by whitespace or the end of the string.
    """
    match = re.search(r"\.(?:\s|$)", message)
    sentence = message[: match.start() + 1] if match else message
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence if sentence.endswith((".", "…")) else sentence + "."


def read_sim_log(log_path: Path | str) -> SimLog:
    """Read and parse an LTspice .log file."""
    path = Path(log_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such log file: {path}")
    # Logs are usually cp1252-ish; LTspice emits the degree sign in .measure output.
    from .netlist import read_text_guess

    return parse_log_text(read_text_guess(path), path)
