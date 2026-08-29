"""Parser tests. These run without LTspice installed."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import parse
from spice_mcp_server.netlist import (
    NetlistFormatError,
    looks_like_expresspcb,
    parse_netlist_text,
)

RC = """* test deck
V1 Vin 0 AC 1
R1 Vin Vout 1.6k
C1 Vout 0 100n
.ac dec 20 10 100k
.end
"""


def test_parses_components_nodes_and_values():
    nl = parse(RC)
    assert [c.ref for c in nl.components] == ["V1", "R1", "C1"]
    assert nl.components[1].nodes == ["Vin", "Vout"]
    assert nl.components[1].value == "1.6k"
    assert nl.components[1].kind == "resistor"
    assert nl.components[1].prefix == "R"


def test_title_is_first_line():
    assert parse(RC).title == "test deck"


def test_directives_captured_and_end_terminates():
    nl = parse(RC)
    assert [d.kind for d in nl.directives] == [".ac"]


def test_net_connection_counts():
    nl = parse(RC)
    counts = {n.name: n.connection_count for n in nl.nets}
    assert counts == {"0": 2, "Vin": 2, "Vout": 2}


def test_net_counts_pins_not_components():
    """A part with both pins on one node contributes two connections."""
    nl = parse("* t\nR1 A A 1k\n.op\n")
    counts = {n.name: n.connection_count for n in nl.nets}
    assert counts["A"] == 2


def test_comment_and_blank_lines_ignored():
    nl = parse("* t\n\n* a comment\nR1 a 0 1k ; trailing comment\n.op\n")
    assert len(nl.components) == 1
    assert nl.components[0].value == "1k"


def test_continuation_lines_folded():
    nl = parse("* t\nR1 a 0\n+ 1k\n.op\n")
    assert nl.components[0].nodes == ["a", "0"]
    assert nl.components[0].value == "1k"


def test_micro_sign_value_survives():
    """LTspice rewrites 'u' as the micro sign when it generates a netlist."""
    nl = parse("* t\nC1 a 0 1µ\n.op\n")
    assert nl.components[0].value == "1µ"


@pytest.mark.parametrize(
    "line,expected_nodes",
    [
        # Fixed-pin-count elements.
        ("R1 a b 1k", ["a", "b"]),
        ("C1 a b 1n", ["a", "b"]),
        ("L1 a b 1m", ["a", "b"]),
        ("V1 a b DC 5", ["a", "b"]),
        ("E1 a b c d 2", ["a", "b", "c", "d"]),
        ("F1 a b Vsense 2", ["a", "b"]),
        # Model-terminated: pin count varies by device, so it is derived.
        ("D1 anode cathode 1N4148", ["anode", "cathode"]),
        ("Q1 c b e 2N2222", ["c", "b", "e"]),
        ("Q2 c b e sub 2N2222", ["c", "b", "e", "sub"]),
        ("M1 d g s bulk NMOS", ["d", "g", "s", "bulk"]),
        ("X1 in out vcc gnd OPAMP", ["in", "out", "vcc", "gnd"]),
        # Trailing key=value params must not be mistaken for nodes.
        ("X2 in out MYSUB Rser=1 tol=5", ["in", "out"]),
        # Mutual inductance couples named inductors and has no nodes.
        ("K1 L1 L2 0.99", []),
    ],
)
def test_element_node_splitting(line, expected_nodes):
    nl = parse(f"* t\n{line}\n.op\n")
    assert nl.components[0].nodes == expected_nodes


def test_unknown_element_marked_uncertain_not_guessed():
    """An 'A' device has a variable pin count, so we must not invent nodes."""
    nl = parse("* t\nA1 a b c d e f g h SOMEFUNC\n.op\n")
    comp = nl.components[0]
    assert comp.nodes == []
    assert comp.nodes_uncertain is True


def test_subckt_body_excluded_from_top_level_connectivity():
    text = """* t
X1 in out MYAMP
.subckt MYAMP p n
R9 p n 1k
.ends
V1 in 0 AC 1
.op
"""
    nl = parse(text)
    assert [c.ref for c in nl.components] == ["X1", "V1"]
    assert [s.name for s in nl.subcircuits] == ["MYAMP"]
    assert nl.subcircuits[0].ports == ["p", "n"]
    # R9 lives inside the definition, so its nodes must not appear as top-level nets.
    assert "p" not in {n.name for n in nl.nets}


def test_expresspcb_is_detected_and_rejected():
    """LTspice's Export Netlist menu item writes this, and it is not SPICE."""
    text = '"ExpressPCB Netlist"\n"LTspice"\n1\n0\n'
    assert looks_like_expresspcb(text)
    with pytest.raises(NetlistFormatError, match="ExpressPCB"):
        parse_netlist_text(text, Path("RCLP.net"), Path("RCLP.net"))


def test_expresspcb_error_points_at_the_asc():
    text = '"ExpressPCB Netlist"\n"LTspice"\n'
    with pytest.raises(NetlistFormatError) as exc:
        parse_netlist_text(text, Path("RCLP.net"), Path("RCLP.net"))
    assert ".asc" in str(exc.value)
