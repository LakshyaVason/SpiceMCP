"""SPICE netlist parsing.

Produces the structured view that gets sent to the LLM in place of a screenshot.

Two formats matter here and they are easy to confuse:

  * A **SPICE netlist** (`.net` from `LTspice -netlist`, or `.cir`) - lines like
    `R1 Vin Vout 10k`. This is what we can reason about.
  * An **ExpressPCB netlist** - what LTspice's "Export Netlist" menu item and the
    `-PCBnetlist` switch produce. It is a quoted-table PCB format with no element
    values in circuit terms. `RCLP.net` in this repo is one of these. Parsing it as
    SPICE would silently produce nonsense, so we detect and reject it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Component, Directive, Net, Netlist, Subcircuit

log = logging.getLogger(__name__)

# Elements with a fixed pin count; everything after the pins is the value expression.
# Counts follow standard SPICE / the LTspice circuit-elements reference.
_FIXED_NODE_COUNTS: dict[str, int] = {
    "R": 2,  # resistor
    "C": 2,  # capacitor
    "L": 2,  # inductor
    "V": 2,  # independent voltage source
    "I": 2,  # independent current source
    "B": 2,  # arbitrary behavioural source
    "E": 4,  # voltage-controlled voltage source
    "G": 4,  # voltage-controlled current source
    "F": 2,  # current-controlled current source (+ controlling source name in value)
    "H": 2,  # current-controlled voltage source (+ controlling source name in value)
    "S": 4,  # voltage-controlled switch
    "W": 2,  # current-controlled switch (+ controlling source name in value)
    "T": 4,  # lossless transmission line
    "O": 4,  # lossy transmission line
    "U": 3,  # uniform RC line
    "K": 0,  # mutual inductance: couples named inductors, has no nodes of its own
}

# Elements whose line ends with a model or subcircuit name: the pins are everything
# between the designator and that trailing name (after stripping key=value params).
# This handles Q with or without a substrate node, M with 4 or more, and X with any
# port count, without hardcoding counts that vary by device.
_MODEL_TERMINATED = {"D", "Q", "M", "J", "Z", "X"}

_KINDS: dict[str, str] = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "V": "voltage source",
    "I": "current source",
    "D": "diode",
    "Q": "bipolar transistor",
    "M": "MOSFET",
    "J": "JFET",
    "Z": "MESFET/IGBT",
    "X": "subcircuit instance",
    "E": "voltage-controlled voltage source",
    "F": "current-controlled current source",
    "G": "voltage-controlled current source",
    "H": "current-controlled voltage source",
    "S": "voltage-controlled switch",
    "W": "current-controlled switch",
    "T": "transmission line",
    "O": "lossy transmission line",
    "U": "uniform RC line",
    "K": "mutual inductance",
    "B": "behavioural source",
    "A": "special function device",
}

_ANALYSIS_DIRECTIVES = {".tran", ".ac", ".dc", ".op", ".noise", ".tf", ".fra", ".four"}

_EXPRESSPCB_MARKER = "ExpressPCB Netlist"


class NetlistFormatError(ValueError):
    """Raised when a file is not a SPICE netlist we can parse."""


@dataclass
class _LogicalLine:
    text: str
    line_no: int


def read_text_guess(path: Path) -> str:
    """Read a text file, tolerating the encodings LTspice has used over the years.

    Older LTspice versions wrote UTF-16LE schematics; current ones write UTF-8. We try
    the BOM-aware codecs first and fall back to cp1252 so a stray extended character
    never hard-fails a read.
    """
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    for codec in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(codec)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def looks_like_expresspcb(text: str) -> bool:
    head = text.lstrip()[:200]
    return _EXPRESSPCB_MARKER in head


def _strip_inline_comment(line: str) -> str:
    """Remove a trailing `;` comment. LTspice treats `;` as a comment starter."""
    idx = line.find(";")
    return line[:idx] if idx >= 0 else line


def _logical_lines(text: str) -> tuple[str | None, list[_LogicalLine]]:
    """Fold continuation lines and drop comments.

    Returns (title, lines). SPICE treats the first line of a deck as a title; LTspice's
    -netlist output makes it a `*` comment naming the source schematic.
    """
    title: str | None = None
    out: list[_LogicalLine] = []

    raw_lines = text.splitlines()
    for idx, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()

        if idx == 1:
            # First line is the title, whether or not it is marked as a comment.
            title = stripped.lstrip("*").strip() or None
            continue

        if not stripped or stripped.startswith("*"):
            continue

        content = _strip_inline_comment(stripped).strip()
        if not content:
            continue

        if content.startswith("+") and out:
            # Continuation of the previous logical line.
            out[-1].text += " " + content[1:].strip()
            continue

        out.append(_LogicalLine(text=content, line_no=idx))

    return title, out


def _parse_element(line: _LogicalLine) -> Component | None:
    tokens = line.text.split()
    if not tokens:
        return None

    ref = tokens[0]
    prefix = ref[0].upper()
    rest = tokens[1:]
    kind = _KINDS.get(prefix, f"unknown element type '{prefix}'")
    uncertain = False

    if prefix in _FIXED_NODE_COUNTS:
        count = _FIXED_NODE_COUNTS[prefix]
        nodes = rest[:count]
        value_tokens = rest[count:]
        if len(nodes) < count:
            # Malformed line; report what we have and let the checks flag it.
            uncertain = True
    elif prefix in _MODEL_TERMINATED:
        # Drop trailing key=value parameters, then the last token is the model name.
        body = list(rest)
        while body and "=" in body[-1]:
            body.pop()
        if len(body) >= 2:
            nodes = body[:-1]
            value_tokens = rest[len(body) - 1 :]
        else:
            nodes = body
            value_tokens = []
            uncertain = True
    else:
        # 'A' devices and anything unrecognised: pin count is not knowable from the
        # line alone. Record the element but declare its connectivity unknown rather
        # than inventing nodes, which would produce bogus floating-net findings.
        return Component(
            ref=ref,
            prefix=prefix,
            kind=kind,
            nodes=[],
            value=" ".join(rest) or None,
            line_no=line.line_no,
            raw_line=line.text,
            nodes_uncertain=True,
        )

    value = " ".join(value_tokens).strip() or None
    return Component(
        ref=ref,
        prefix=prefix,
        kind=kind,
        nodes=list(nodes),
        value=value,
        line_no=line.line_no,
        raw_line=line.text,
        nodes_uncertain=uncertain,
    )


def parse_netlist_text(text: str, source_path: Path, netlist_path: Path) -> Netlist:
    """Parse SPICE netlist text into the structured model."""
    if looks_like_expresspcb(text):
        raise NetlistFormatError(
            f"{netlist_path.name} is an ExpressPCB netlist, not a SPICE netlist. This is "
            "what LTspice's 'Export Netlist' menu item (and the -PCBnetlist switch) "
            "writes; it describes PCB connectivity and cannot be simulated or reasoned "
            "about as a circuit.\n\n"
            "Point this tool at the .asc schematic instead - it will be converted with "
            "'LTspice -netlist', which emits real SPICE."
        )

    title, lines = _logical_lines(text)

    components: list[Component] = []
    directives: list[Directive] = []
    subcircuits: list[Subcircuit] = []
    subckt_depth = 0

    for line in lines:
        lowered = line.text.lower()

        if lowered.startswith(".subckt"):
            tokens = line.text.split()
            # .subckt <name> <port> <port> ... [params]
            ports = [t for t in tokens[2:] if "=" not in t]
            subcircuits.append(
                Subcircuit(
                    name=tokens[1] if len(tokens) > 1 else "?",
                    ports=ports,
                    line_no=line.line_no,
                )
            )
            subckt_depth += 1
            continue

        if lowered.startswith(".ends"):
            subckt_depth = max(0, subckt_depth - 1)
            continue

        if lowered.startswith(".end"):
            break

        if line.text.startswith("."):
            directives.append(
                Directive(
                    kind=lowered.split()[0],
                    text=line.text,
                    line_no=line.line_no,
                )
            )
            continue

        if subckt_depth:
            # Inside a .subckt definition. These are not top-level instances, so they
            # must not contribute to top-level net connectivity.
            continue

        component = _parse_element(line)
        if component is not None:
            components.append(component)

    nets = _build_nets(components)

    return Netlist(
        source_path=str(source_path),
        netlist_path=str(netlist_path),
        title=title,
        components=components,
        directives=directives,
        subcircuits=subcircuits,
        nets=nets,
        raw_text=text,
    )


def _build_nets(components: list[Component]) -> list[Net]:
    """Aggregate element pins into nets, counting pins (not distinct components).

    A two-pin part wired across the same node counts twice, which is what makes the
    single-connection test meaningful.
    """
    counts: dict[str, int] = {}
    refs: dict[str, list[str]] = {}
    for comp in components:
        for node in comp.nodes:
            counts[node] = counts.get(node, 0) + 1
            refs.setdefault(node, [])
            if comp.ref not in refs[node]:
                refs[node].append(comp.ref)

    return [
        Net(name=name, connection_count=counts[name], connected_refs=sorted(refs[name]))
        for name in sorted(counts)
    ]


def has_analysis_directive(netlist: Netlist) -> bool:
    return any(d.kind in _ANALYSIS_DIRECTIVES for d in netlist.directives)


def load_netlist(path: str | Path) -> Netlist:
    """Load a circuit from a .asc, .net or .cir path.

    `.asc` schematics are converted with `LTspice -netlist` into a scratch directory,
    so the user's own `.net` file is never overwritten.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"No such file: {source}")

    suffix = source.suffix.lower()

    if suffix == ".asc":
        from .ltspice import generate_netlist

        netlist_path = generate_netlist(source)
    elif suffix in {".net", ".cir", ".sp", ".spi"}:
        netlist_path = source
    else:
        raise NetlistFormatError(
            f"Unsupported file type {source.suffix!r}. Expected a .asc schematic or a "
            ".net/.cir SPICE netlist."
        )

    text = read_text_guess(netlist_path)

    # A .net sitting next to a .asc is frequently the ExpressPCB export. If we were
    # handed one directly, fall back to converting the schematic automatically rather
    # than making the caller figure it out.
    if looks_like_expresspcb(text) and suffix != ".asc":
        sibling = source.with_suffix(".asc")
        if sibling.is_file():
            log.info(
                "%s is ExpressPCB format; converting sibling schematic %s instead",
                source.name,
                sibling.name,
            )
            from .ltspice import generate_netlist

            netlist_path = generate_netlist(sibling)
            text = read_text_guess(netlist_path)
            return parse_netlist_text(text, source, netlist_path)

    return parse_netlist_text(text, source, netlist_path)
