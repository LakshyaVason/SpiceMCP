"""Byte-preserving surgical edits to LTspice `.asc` schematics.

The `.asc` is the source of truth: it carries the GUI coordinates, so a fix written
here stays openable and editable in LTspice. That is the whole reason we patch the
schematic rather than the flattened netlist.

Written by hand rather than through `spicelib`'s `AscEditor.save_netlist`, which
reformats the file and raises on library-sourced components. The contract here is
stricter than "produces a valid file":

  * exactly one line changes;
  * every other byte, including each line's own terminator, is preserved;
  * the original encoding is written back unchanged.

Both halves of that matter in this repo: `RCLP.asc` is bare-LF with no BOM while
`RCLP.net` is CRLF, and older LTspice versions wrote UTF-16 schematics. A patch that
"helpfully" normalised either would produce a diff the user cannot review.

Two traps in the format, both live in this repo:

  * A value belongs to the `SYMBOL` block it appears in, so `SYMATTR Value` must be
    looked up *within* the block whose `SYMATTR InstName` matches - not by scanning the
    file for the nearest value line.
  * `SYMATTR Value2` is a real, different attribute (`RCLP.asc`'s `V1` has both). The
    attribute name is therefore matched as a whole token; a `startswith("Value")` test
    silently patches the wrong line.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# BOM-carrying encodings, longest marker first so utf-8-sig is not shadowed.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# A SYMBOL block ends when any of these appears; SYMATTR/WINDOW belong to the block.
_BLOCK_ENDERS = ("SYMBOL", "WIRE", "FLAG", "TEXT", "DATAFLAG", "IOPIN", "SHEET", "BUSTAP")


class AscError(RuntimeError):
    """The schematic could not be read or the requested edit does not apply."""


@dataclass
class AscFile:
    """A schematic held as decoded lines plus enough metadata to rebuild it byte-exactly."""

    path: Path
    lines: list[str]  # each line keeps its own terminator
    encoding: str

    def text(self) -> str:
        return "".join(self.lines)

    def to_bytes(self) -> bytes:
        return self.text().encode(self.encoding)


def read_asc(path: str | Path) -> AscFile:
    """Read a `.asc`, detecting its encoding and keeping line terminators intact."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError as exc:
        raise AscError(f"No such schematic: {p}") from exc

    encoding = "utf-8"
    for marker, candidate in _BOMS:
        if data.startswith(marker):
            encoding = candidate
            break
    else:
        # No BOM. LTspice 26 writes UTF-8; older builds wrote cp1252, and Ω/µ/° in a
        # component value is exactly where the two disagree.
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "cp1252"

    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise AscError(f"Could not decode {p.name} as {encoding}: {exc}") from exc

    # keepends so CRLF, LF and a missing final newline all survive a round trip.
    return AscFile(path=p, lines=text.splitlines(keepends=True), encoding=encoding)


def _attr_name(line: str) -> str | None:
    """Return the attribute name of a SYMATTR line, or None if it is not one.

    Tokenised on purpose: `SYMATTR Value2 ...` must not be mistaken for `Value`.
    """
    stripped = line.strip()
    if not stripped.upper().startswith("SYMATTR"):
        return None
    parts = stripped.split(None, 2)
    return parts[1] if len(parts) >= 2 else None


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""  # final line with no terminator


@dataclass
class ValueLocation:
    """Where a component's value lives in the file."""

    ref: str
    instname_index: int
    value_index: int | None  # None when the component has no SYMATTR Value line
    current_value: str | None
    symbol_index: int


def locate_value(asc: AscFile, ref: str) -> ValueLocation:
    """Find the `SYMATTR Value` line belonging to `ref`.

    `ref` is matched case-insensitively because LTspice designators are, but the file's
    own spelling is what gets reported back.
    """
    target = ref.strip().upper()
    if not target:
        raise AscError("A reference designator is required.")

    symbol_index = -1
    instname_index: int | None = None
    value_index: int | None = None
    current_value: str | None = None
    found_symbol_index = -1
    in_target_block = False
    seen: list[str] = []

    for index, line in enumerate(asc.lines):
        head = line.strip().split(None, 1)[0].upper() if line.strip() else ""

        if head in _BLOCK_ENDERS:
            if in_target_block:
                break  # the target block just ended; stop before another Value line
            in_target_block = False
            symbol_index = index if head == "SYMBOL" else -1
            continue

        name = _attr_name(line)
        if name is None:
            continue

        if name.upper() == "INSTNAME":
            parts = line.strip().split(None, 2)
            instance = parts[2].strip() if len(parts) >= 3 else ""
            seen.append(instance)
            in_target_block = instance.upper() == target
            if in_target_block:
                instname_index = index
                found_symbol_index = symbol_index
            continue

        if in_target_block and name.upper() == "VALUE":
            parts = line.strip().split(None, 2)
            value_index = index
            current_value = parts[2].strip() if len(parts) >= 3 else ""

    if instname_index is None:
        known = ", ".join(sorted(seen)) or "none"
        raise AscError(
            f"No component named {ref} in {asc.path.name}.\n"
            f"Components present: {known}"
        )

    return ValueLocation(
        ref=ref,
        instname_index=instname_index,
        value_index=value_index,
        current_value=current_value,
        symbol_index=found_symbol_index,
    )


def unified_diff(before: str, after: str, name: str) -> str:
    """A reviewable diff of the two file states."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{name} (before)",
            tofile=f"{name} (after)",
            n=3,
        )
    )


@dataclass
class PatchOutcome:
    ref: str
    old_value: str | None
    new_value: str
    line_no: int  # 1-indexed
    inserted: bool
    applied: bool
    diff: str
    before_line: str | None
    after_line: str
    encoding: str


def patch_component_value(
    asc_path: str | Path,
    ref: str,
    new_value: str,
    *,
    apply: bool = False,
) -> PatchOutcome:
    """Change one component's value in a `.asc`, preserving every other byte.

    With `apply=False` (the default) nothing is written and the diff is returned for
    review. That default is deliberate: the destructive form has to be asked for.

    A component with no `SYMATTR Value` line at all - LTspice writes none when the
    value was left blank - gets one inserted directly after its `SYMATTR InstName`,
    which is where LTspice itself puts it.
    """
    new_value = new_value.strip()
    if not new_value:
        raise AscError(
            "A new value is required. To blank a value, pass the two-character "
            'literal "" as LTspice itself writes it.'
        )

    asc = read_asc(asc_path)
    if not asc.lines:
        raise AscError(f"{asc.path.name} is empty.")

    location = locate_value(asc, ref)
    before_text = asc.text()
    patched = list(asc.lines)

    if location.value_index is not None:
        index = location.value_index
        original = patched[index]
        ending = _line_ending(original)
        # Rebuild from the file's own prefix so indentation and the SYMATTR/Value
        # spelling are carried over rather than normalised.
        stripped = original[: len(original) - len(ending)]
        parts = stripped.split(None, 2)
        prefix_len = len(stripped) - len(stripped.lstrip())
        indent = stripped[:prefix_len]
        patched[index] = f"{indent}{parts[0]} {parts[1]} {new_value}{ending}"
        before_line: str | None = original
        inserted = False
    else:
        # No Value line: insert one, borrowing the InstName line's terminator so a
        # CRLF file does not acquire a lone LF.
        index = location.instname_index + 1
        anchor = patched[location.instname_index]
        ending = _line_ending(anchor)
        if not ending:
            # The InstName line was last and unterminated; terminate it before
            # appending, or the two lines would merge.
            patched[location.instname_index] = anchor + "\n"
            ending = "\n"
        stripped = anchor[: len(anchor) - len(_line_ending(anchor))]
        indent = stripped[: len(stripped) - len(stripped.lstrip())]
        patched.insert(index, f"{indent}SYMATTR Value {new_value}{ending}")
        before_line = None
        inserted = True

    after_text = "".join(patched)
    outcome = PatchOutcome(
        ref=ref,
        old_value=location.current_value,
        new_value=new_value,
        line_no=index + 1,
        inserted=inserted,
        applied=False,
        diff=unified_diff(before_text, after_text, asc.path.name),
        before_line=before_line.rstrip("\r\n") if before_line else None,
        after_line=patched[index].rstrip("\r\n"),
        encoding=asc.encoding,
    )

    if not apply:
        return outcome

    if after_text == before_text:
        log.info("patch_component_value: %s already %s, nothing written", ref, new_value)
        outcome.applied = True
        return outcome

    # write_bytes so the encoding and every line terminator go back exactly as read.
    Path(asc_path).write_bytes(after_text.encode(asc.encoding))
    outcome.applied = True
    log.info(
        "patched %s: %s %r -> %r (line %d, %s)",
        asc.path.name,
        ref,
        location.current_value,
        new_value,
        outcome.line_no,
        asc.encoding,
    )
    return outcome
