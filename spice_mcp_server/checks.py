"""Static checks over a parsed netlist. No simulation is run.

This is the cheap first pass: it costs no LTspice invocation and no tokens beyond the
findings themselves, and it catches the majority of hand-drafting mistakes.

Three of these checks go beyond simple syntax and model what actually makes SPICE
fail, which is where most of the diagnostic value is:

  * `isolated_section`  - the circuit splits into pieces and a piece has no ground.
    SPICE cannot solve it ("singular matrix" / "circuit cannot be analyzed").
  * `no_dc_path_to_ground` - a node reachable from ground only through capacitors.
    The classic floating-node convergence failure in .op/.tran.
  * `suspicious_suffix` - `M` means milli in SPICE, not mega. `10M` on a resistor is
    10 milliohms. This produces a plausible-looking wrong answer rather than an
    error, so nothing else catches it. LTspice 26 added its own GUI warning for it.

False positives matter more than coverage here: a check that cries wolf trains the
user (and the model) to ignore the output. Anything uncertain is downgraded rather
than dropped.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import Finding, Netlist, StaticCheckResult
from .netlist import has_analysis_directive

GROUND = "0"

# Aliases LTspice/SPICE treat as ground, or that users expect to be ground.
_GROUND_ALIASES = {"0", "gnd", "gnd!", "agnd", "dgnd"}

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

# Elements that conduct at DC. A capacitor does not, and an ideal current source is an
# open circuit, so neither establishes a DC path to ground.
_NON_DC_PREFIXES = {"C", "I"}

# Value with an optional SPICE engineering suffix and optional trailing unit text.
_VALUE_RE = re.compile(
    r"""^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*
        (meg|mil|t|g|k|m|u|µ|n|p|f)?
        [a-zA-Z]*\s*$""",
    re.VERBOSE | re.IGNORECASE,
)

# Values that are expressions or source specs rather than plain numbers. These are
# valid but not worth strict-checking.
_COMPLEX_VALUE_RE = re.compile(
    r"[{}()=]|^\s*(ac|dc|sine|pulse|pwl|exp|sffm|noise|table|value|laplace|r_?\d*)\b",
    re.IGNORECASE,
)


def _sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.check, f.line_no or 0),
    )


def _check_has_components(netlist: Netlist) -> list[Finding]:
    if netlist.components:
        return []
    return [
        Finding(
            check="empty_circuit",
            severity="error",
            message="The netlist contains no circuit elements.",
            suggestion="Check that the schematic actually has components placed, and that "
            "the file converted correctly.",
        )
    ]


def _check_ground(netlist: Netlist) -> list[Finding]:
    """Every SPICE circuit needs node 0 as the voltage reference."""
    if not netlist.components:
        return []

    net_names = {n.name for n in netlist.nets}
    if GROUND in net_names:
        return []

    # Distinguish "no ground at all" from "ground drawn but labelled wrong", which is a
    # much more actionable message.
    mislabelled = sorted(n for n in net_names if n.lower() in _GROUND_ALIASES)
    if mislabelled:
        return [
            Finding(
                check="no_ground",
                severity="error",
                message=(
                    f"There is no node 0 (ground). The net(s) {', '.join(mislabelled)} look "
                    "like a ground net but SPICE only treats node '0' as the reference."
                ),
                nets=mislabelled,
                suggestion="Place the LTspice GND symbol (which nets to 0) instead of a "
                f"net label named {mislabelled[0]!r}.",
            )
        ]

    return [
        Finding(
            check="no_ground",
            severity="error",
            message="There is no ground reference (node 0) anywhere in the circuit. SPICE "
            "measures all voltages against node 0 and cannot solve a circuit without it.",
            suggestion="Place a GND symbol in the schematic (LTspice shortcut: G) and wire "
            "it to the circuit's reference node.",
        )
    ]


def _check_floating_nets(netlist: Netlist) -> list[Finding]:
    """A net with a single pin attached is a wire that goes nowhere."""
    if any(c.nodes_uncertain for c in netlist.components):
        # Some element's pin count is unknown, so connection counts may be understated.
        # Report at reduced severity rather than risk a false accusation.
        severity = "warning"
        caveat = (
            " (Reported as a warning because this circuit contains an element whose pin "
            "count could not be determined, so the connection count may be incomplete.)"
        )
    else:
        severity = "error"
        caveat = ""

    findings: list[Finding] = []
    for net in netlist.nets:
        if net.name == GROUND or net.connection_count != 1:
            continue

        ref = net.connected_refs[0] if net.connected_refs else "?"

        # LTspice auto-names pins it found unconnected as NC_nn when it generates the
        # netlist, so that prefix is a direct signal rather than an inference.
        if net.name.upper().startswith("NC_"):
            findings.append(
                Finding(
                    check="floating_pin",
                    severity=severity,
                    message=(
                        f"{ref} has an unconnected pin. LTspice auto-named it {net.name!r}, "
                        f"which is how it labels a pin with no wire attached." + caveat
                    ),
                    refs=[ref],
                    nets=[net.name],
                    suggestion=f"Wire the dangling pin of {ref} to the node it belongs to. "
                    "In the schematic the pin will have no wire touching it.",
                )
            )
        else:
            findings.append(
                Finding(
                    check="floating_net",
                    severity=severity,
                    message=(
                        f"Net {net.name!r} has only one connection ({ref}), so no current can "
                        "flow through it." + caveat
                    ),
                    refs=[ref],
                    nets=[net.name],
                    suggestion=f"Either connect {net.name!r} to the rest of the circuit or "
                    f"remove {ref}.",
                )
            )
    return findings


def _check_duplicate_refs(netlist: Netlist) -> list[Finding]:
    counts = Counter(c.ref.upper() for c in netlist.components)
    findings: list[Finding] = []
    for ref, count in sorted(counts.items()):
        if count < 2:
            continue
        lines = [c.line_no for c in netlist.components if c.ref.upper() == ref]
        findings.append(
            Finding(
                check="duplicate_ref",
                severity="error",
                message=(
                    f"Reference designator {ref} is used {count} times (netlist lines "
                    f"{', '.join(str(n) for n in lines)}). SPICE requires unique designators."
                ),
                refs=[ref],
                line_no=lines[0],
                suggestion=f"Rename the duplicates so each is unique, e.g. {ref} and "
                f"{ref[0]}{max(counts.values()) + 1}.",
            )
        )
    return findings


def _check_values(netlist: Netlist) -> list[Finding]:
    findings: list[Finding] = []
    for comp in netlist.components:
        # K (mutual inductance) and X (subcircuit) carry names, not numeric values.
        if comp.prefix in {"K", "X", "A"}:
            continue

        if comp.value is None or not comp.value.strip():
            findings.append(
                Finding(
                    check="missing_value",
                    severity="error",
                    message=f"{comp.ref} ({comp.kind}) has no value.",
                    refs=[comp.ref],
                    line_no=comp.line_no,
                    suggestion=f"Set a value on {comp.ref}. In the schematic, right-click the "
                    "component and fill in the Value field.",
                )
            )
            continue

        value = comp.value.strip()

        # An empty pair of quotes is what LTspice writes for a blank Value attribute.
        if value in {'""', "''"}:
            findings.append(
                Finding(
                    check="missing_value",
                    severity="error",
                    message=f"{comp.ref} ({comp.kind}) has an empty value ({value}).",
                    refs=[comp.ref],
                    line_no=comp.line_no,
                    suggestion=f"Fill in the Value field for {comp.ref}.",
                )
            )
            continue

        if comp.prefix in {"R", "C", "L"} and not _COMPLEX_VALUE_RE.search(value):
            if not _VALUE_RE.match(value):
                findings.append(
                    Finding(
                        check="malformed_value",
                        severity="error",
                        message=f"{comp.ref} has a value {value!r} that is not valid SPICE "
                        "number syntax.",
                        refs=[comp.ref],
                        line_no=comp.line_no,
                        suggestion="Use a plain number with an optional SPICE suffix, e.g. "
                        "'10k', '4.7u', '100n'.",
                    )
                )

    return findings


def _check_suspicious_suffix(netlist: Netlist) -> list[Finding]:
    """Catch the SPICE `M` = milli trap, which yields a wrong answer, not an error."""
    findings: list[Finding] = []

    for comp in netlist.components:
        if comp.prefix not in {"R", "L"} or not comp.value:
            continue
        value = comp.value.strip()
        if _COMPLEX_VALUE_RE.search(value):
            continue
        match = _VALUE_RE.match(value)
        if not match:
            continue
        suffix = (match.group(1) or "").lower()
        if suffix != "m":
            continue

        unit = "ohms" if comp.prefix == "R" else "henries"
        findings.append(
            Finding(
                check="suspicious_suffix",
                severity="warning",
                message=(
                    f"{comp.ref} has value {value!r}. In SPICE the suffix 'M' means milli, "
                    f"not mega, so this is {value.rstrip('mM')} milli{unit}. If you meant "
                    f"mega, write 'Meg'."
                ),
                refs=[comp.ref],
                line_no=comp.line_no,
                suggestion=f"If mega was intended, change {comp.ref} to "
                f"{value.rstrip('mM')}Meg.",
            )
        )

    # Same trap in analysis directive frequency arguments, e.g. `.ac dec 10 1 10M`.
    for directive in netlist.directives:
        if directive.kind not in {".ac", ".tran", ".noise"}:
            continue
        for token in directive.text.split()[1:]:
            if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)[mM]", token):
                findings.append(
                    Finding(
                        check="suspicious_suffix",
                        severity="warning",
                        message=(
                            f"{directive.kind} uses {token!r}. 'M' is milli in SPICE, so this "
                            f"is {token[:-1]} millihertz/milliseconds. Use 'Meg' for mega."
                        ),
                        line_no=directive.line_no,
                        suggestion=f"Change {token!r} to {token[:-1]}Meg if mega was intended.",
                    )
                )

    return findings


def _check_analysis_directive(netlist: Netlist) -> list[Finding]:
    if not netlist.components or has_analysis_directive(netlist):
        return []
    return [
        Finding(
            check="no_analysis_directive",
            severity="warning",
            message="The circuit has no analysis directive (.tran, .ac, .dc, .op, ...), so "
            "LTspice has nothing to simulate.",
            suggestion="Add a directive on the schematic, e.g. '.tran 1m' for a transient "
            "run or '.op' for the operating point.",
        )
    ]


def _connected_groups(netlist: Netlist, dc_only: bool) -> list[set[str]]:
    """Union-find over nets, joined by element pins.

    With dc_only=True, capacitors and current sources are excluded, so the result
    reflects DC connectivity only.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for net in netlist.nets:
        find(net.name)

    for comp in netlist.components:
        if dc_only and comp.prefix in _NON_DC_PREFIXES:
            continue
        if len(comp.nodes) < 2:
            continue
        first = comp.nodes[0]
        for node in comp.nodes[1:]:
            union(first, node)

    groups: dict[str, set[str]] = {}
    for net in netlist.nets:
        groups.setdefault(find(net.name), set()).add(net.name)
    return list(groups.values())


def _check_isolated_sections(netlist: Netlist) -> list[Finding]:
    """A disconnected piece of circuit with no ground cannot be solved."""
    if not netlist.components or any(c.nodes_uncertain for c in netlist.components):
        return []
    if not any(n.name == GROUND for n in netlist.nets):
        return []  # already reported by _check_ground

    findings: list[Finding] = []
    for group in _connected_groups(netlist, dc_only=False):
        if GROUND in group:
            continue
        nets = sorted(group)
        refs = sorted(
            {c.ref for c in netlist.components if any(n in group for n in c.nodes)}
        )
        findings.append(
            Finding(
                check="isolated_section",
                severity="error",
                message=(
                    f"Nets {', '.join(nets)} form a section of circuit with no connection to "
                    f"ground. SPICE cannot solve it and will report a singular matrix or "
                    f"refuse to analyze the circuit. Components involved: {', '.join(refs)}."
                ),
                refs=refs,
                nets=nets,
                suggestion="Connect this section to the main circuit, or give it its own "
                "ground reference.",
            )
        )
    return findings


def _check_dc_path_to_ground(netlist: Netlist) -> list[Finding]:
    """Nodes reachable from ground only through capacitors fail to converge.

    This is the textbook SPICE floating-node error: a node with no resistive path to
    ground has an undefined DC operating point.
    """
    if not netlist.components or any(c.nodes_uncertain for c in netlist.components):
        return []
    if not any(n.name == GROUND for n in netlist.nets):
        return []

    # Only meaningful when a DC solution is actually computed.
    dc_analyses = {".op", ".tran", ".dc", ".ac", ".noise", ".tf"}
    if not any(d.kind in dc_analyses for d in netlist.directives):
        return []

    dc_groups = _connected_groups(netlist, dc_only=True)
    ground_group = next((g for g in dc_groups if GROUND in g), set())

    findings: list[Finding] = []
    for group in dc_groups:
        if GROUND in group:
            continue
        nets = sorted(group)
        refs = sorted(
            {c.ref for c in netlist.components if any(n in group for n in c.nodes)}
        )
        findings.append(
            Finding(
                check="no_dc_path_to_ground",
                severity="error",
                message=(
                    f"Net(s) {', '.join(nets)} have no DC path to ground - they connect to the "
                    "rest of the circuit only through capacitors (or current sources). The DC "
                    "operating point is undefined, so the simulation will fail to converge or "
                    "report a singular matrix."
                ),
                refs=refs,
                nets=nets,
                suggestion=f"Add a high-value resistor (e.g. 1Meg) from {nets[0]} to ground to "
                "define its DC level, or provide a resistive bias path.",
            )
        )
    _ = ground_group
    return findings


_CHECKS = (
    _check_has_components,
    _check_ground,
    _check_isolated_sections,
    _check_dc_path_to_ground,
    _check_floating_nets,
    _check_duplicate_refs,
    _check_values,
    _check_suspicious_suffix,
    _check_analysis_directive,
)


def run_static_checks(netlist: Netlist) -> StaticCheckResult:
    """Run every static check and return the aggregated result."""
    findings: list[Finding] = []
    for check in _CHECKS:
        findings.extend(check(netlist))

    findings = _sort_findings(findings)
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")

    if not findings:
        summary = "No static issues found. The netlist parses cleanly and is fully connected."
    else:
        parts = []
        if errors:
            parts.append(f"{errors} error{'s' if errors != 1 else ''}")
        if warnings:
            parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
        infos = len(findings) - errors - warnings
        if infos:
            parts.append(f"{infos} info")
        summary = f"Found {', '.join(parts)} without running a simulation."

    return StaticCheckResult(
        source_path=netlist.source_path,
        ok=errors == 0,
        findings=findings,
        summary=summary,
    )
