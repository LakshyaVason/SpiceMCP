"""MCP tool definitions for LTspice circuit debugging.

Nothing in this module knows about LLMs. Tools are declared with @mcp.tool(); the
return type annotation is the output schema (mcp v2 derives it automatically), so the
pydantic models in models.py are the contract the model sees.

Two conventions matter here:

  * **Never write to stdout.** stdout is the MCP wire. Use the `log` logger, which the
    SDK flushes to stderr.
  * **Raise ToolError for anticipated failures.** Any other exception is treated as a
    crash and the model receives only "Error executing tool <name>", discarding the
    message. Since our error text is often the actual diagnosis (for example "this is
    an ExpressPCB netlist, point me at the .asc instead"), losing it would defeat the
    purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server import MCPServer

# Only importable from this deep path in mcp 2.1; it is not re-exported by mcp.server
# or mcp.server.mcpserver. Revisit if a future version surfaces it more conveniently.
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .asc import AscError
from .asc import patch_component_value as _patch_component_value
from .checks import run_static_checks
from .diff import diff_netlists
from .logparse import read_sim_log as _read_sim_log
from .ltspice import LTSpiceError, run_batch
from .models import (
    ExportResult,
    Netlist,
    NetlistDiff,
    PatchResult,
    SimLog,
    SimulationResult,
    StaticCheckResult,
)
from .netlist import NetlistFormatError, load_netlist

log = logging.getLogger(__name__)

mcp = MCPServer(
    "spice-mcp",
    title="SPICE MCP client",
    version=__version__,
    instructions=(
        "Tools for debugging LTspice circuits as text rather than screenshots.\n\n"
        "Recommended order when diagnosing a circuit:\n"
        "1. read_netlist to get the structured circuit.\n"
        "2. check_netlist_static for the cheap, simulation-free pass. Most drafting "
        "mistakes are caught here.\n"
        "3. Only then run a simulation, which is comparatively slow.\n\n"
        "Prefer the .asc schematic as the path argument: it is the source of truth and "
        "is converted to a real SPICE netlist automatically. A .net file sitting next to "
        "a .asc is often an ExpressPCB export, which is not SPICE."
    ),
)


def _resolve(path: str) -> Path:
    """Validate a caller-supplied path and return it resolved."""
    if not path or not path.strip():
        raise ToolError("A file path is required.")

    candidate = Path(path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    if not candidate.exists():
        raise ToolError(
            f"No such file: {candidate}\n\n"
            "Pass an absolute path, or a path relative to the directory the server was "
            "started in."
        )
    if not candidate.is_file():
        raise ToolError(f"{candidate} is a directory, not a circuit file.")
    return candidate


def _load(path: str) -> Netlist:
    """Shared loader that maps internal exceptions onto ToolError."""
    resolved = _resolve(path)
    try:
        return load_netlist(resolved)
    except NetlistFormatError as exc:
        # The message here is genuinely diagnostic; surface it verbatim.
        raise ToolError(str(exc)) from exc
    except LTSpiceError as exc:
        raise ToolError(
            f"Could not convert {resolved.name} to a netlist.\n\n{exc}"
        ) from exc
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def read_netlist(path: str, include_raw_text: bool = True) -> Netlist:
    """Parse an LTspice circuit into structured JSON.

    Accepts a .asc schematic (converted with `LTspice -netlist` into a scratch
    directory, leaving the caller's files untouched) or an existing .net/.cir SPICE
    netlist. Returns components with their reference designators, element types, nets
    and values; all dot-directives; subcircuit definitions; and per-net connection
    counts.

    This structured form is the point of the tool: it replaces sending a screenshot of
    the schematic, and a connection count of 1 on any net is an immediate red flag.

    Args:
        path: Path to a .asc, .net or .cir file.
        include_raw_text: Include the full netlist text alongside the structured data.
            Set False to save input tokens when the structured view is sufficient.
    """
    netlist = _load(path)
    if not include_raw_text:
        netlist = netlist.model_copy(update={"raw_text": ""})
    log.info(
        "read_netlist %s -> %d components, %d nets",
        path,
        len(netlist.components),
        len(netlist.nets),
    )
    return netlist


@mcp.tool()
def check_netlist_static(path: str) -> StaticCheckResult:
    """Check a circuit for common mistakes without running a simulation.

    Fast and free compared to a simulation, so run this before simulating. Detects:
    missing ground (node 0), floating pins and single-connection nets, sections of
    circuit isolated from ground, nodes with no DC path to ground (the classic
    non-convergence cause), duplicate reference designators, missing or malformed
    component values, a missing analysis directive, and the SPICE 'M means milli, not
    mega' trap that silently produces wrong answers.

    Findings are ordered most severe first. `ok` is True only when there are no
    error-severity findings.

    Args:
        path: Path to a .asc, .net or .cir file.
    """
    netlist = _load(path)
    result = run_static_checks(netlist)
    log.info("check_netlist_static %s -> %s", path, result.summary)
    return result


@mcp.tool()
def run_simulation(path: str, timeout_s: float = 120.0) -> SimulationResult:
    """Run an LTspice simulation in batch mode and return the parsed log.

    The circuit is copied into a scratch directory first, so the .raw and .log land
    there and nothing is written next to the caller's file.

    Read `succeeded` to find out whether it worked. Do not infer success from
    `returncode` or from the presence of `raw_path`: LTspice exits 0 for several
    genuinely failed runs, and writes a .raw file even when the analysis failed. Both
    facts are verified behaviours of LTspice 26, not theoretical concerns.

    If the run times out, the process tree is killed. A timeout usually means the
    circuit is oscillating or the timestep has collapsed, which is itself the
    diagnosis - not a reason to retry with a longer timeout.

    Args:
        path: Path to a .asc, .net or .cir file.
        timeout_s: Seconds to allow before killing the simulation.
    """
    resolved = _resolve(path)
    if timeout_s <= 0:
        raise ToolError("timeout_s must be greater than zero.")

    try:
        batch = run_batch(resolved, timeout_s=timeout_s)
    except LTSpiceError as exc:
        raise ToolError(f"Could not run LTspice on {resolved.name}.\n\n{exc}") from exc
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc

    parsed: SimLog | None = None
    if batch.log_path:
        try:
            parsed = _read_sim_log(batch.log_path)
        except Exception as exc:  # a truncated log must not sink the whole call
            log.warning("could not parse %s: %s", batch.log_path, exc)

    if batch.timed_out:
        succeeded = False
        summary = (
            f"Simulation timed out after {timeout_s:.0f}s and was killed. The circuit is "
            "most likely oscillating or the timestep has collapsed."
        )
    elif parsed is not None:
        succeeded = parsed.succeeded
        summary = parsed.summary
    else:
        succeeded = False
        summary = (
            "LTspice produced no log file, so the run cannot be assessed. "
            f"Exit code was {batch.returncode}."
        )

    log.info("run_simulation %s -> succeeded=%s", path, succeeded)
    return SimulationResult(
        circuit_path=str(resolved),
        simulated_path=batch.circuit_path,
        succeeded=succeeded,
        timed_out=batch.timed_out,
        returncode=batch.returncode,
        log_path=batch.log_path,
        raw_path=batch.raw_path,
        log=parsed,
        summary=summary,
    )


@mcp.tool()
def read_sim_log(path: str, include_raw_text: bool = False) -> SimLog:
    """Parse an existing LTspice .log file into structured findings.

    Use this to re-read the log from an earlier run_simulation call, or to inspect a
    log produced by the LTspice GUI. This is usually where the real diagnosis lives: a
    circuit can be topologically perfect and still fail to solve, and LTspice only
    explains why in the log.

    Recognises singular and over-defined matrices, undefined models (with the offending
    netlist line number), convergence and timestep failures, floating nodes, missing
    .include/.lib files, and .measure results.

    Args:
        path: Path to a .log file.
        include_raw_text: Include the full log text. Off by default because logs are
            mostly boilerplate and the structured findings carry the diagnosis.
    """
    resolved = _resolve(path)
    try:
        parsed = _read_sim_log(resolved)
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc

    if not include_raw_text:
        parsed = parsed.model_copy(update={"raw_text": ""})
    log.info("read_sim_log %s -> %s", path, parsed.summary)
    return parsed


@mcp.tool()
def patch_component_value(
    asc_path: str, ref: str, new_value: str, apply: bool = False
) -> PatchResult:
    """Change one component's value in a .asc schematic, preserving every other byte.

    **Defaults to a preview.** With apply=False nothing is written: you get back the
    unified diff so the user can review the change. Propose the fix that way first,
    show them the diff, and only call again with apply=True once they have agreed. Do
    not set apply=True on your own initiative.

    The edit rewrites only the `SYMATTR Value` line belonging to the matching
    `SYMATTR InstName`, keeping the file's original encoding and line endings exactly.
    That is what lets the patched schematic still open in the LTspice GUI, and what
    keeps the diff reviewable.

    Only component values can be changed here. Rewiring needs a different fix - report
    it in prose instead.

    Args:
        asc_path: Path to the .asc schematic. Not a .net or .cir: the schematic is the
            source of truth and the only form that stays editable in the GUI.
        ref: Reference designator, e.g. 'C1'. Case-insensitive.
        new_value: The replacement value, e.g. '100n'. Remember that in SPICE 'M' means
            milli, not mega - use 'MEG' for mega.
        apply: Write the file. Leave False to preview.
    """
    resolved = _resolve(asc_path)
    if resolved.suffix.lower() != ".asc":
        raise ToolError(
            f"{resolved.name} is not a .asc schematic. Values can only be patched in "
            "the schematic, because that is the file that stays usable in the LTspice "
            "GUI. Point me at the .asc instead."
        )

    try:
        outcome = _patch_component_value(resolved, ref, new_value, apply=apply)
    except AscError as exc:
        # Genuinely diagnostic - it lists the components that do exist.
        raise ToolError(str(exc)) from exc

    if outcome.applied:
        summary = (
            f"{ref} set to {outcome.new_value} in {resolved.name} "
            f"(line {outcome.line_no}, was {outcome.old_value!r}). File written."
        )
    else:
        summary = (
            f"Proposed: {ref} {outcome.old_value!r} -> {outcome.new_value!r} at line "
            f"{outcome.line_no}. Nothing written yet - show the diff and get approval, "
            "then call again with apply=True."
        )

    log.info("patch_component_value %s %s -> applied=%s", asc_path, ref, outcome.applied)
    return PatchResult(
        asc_path=str(resolved),
        ref=outcome.ref,
        old_value=outcome.old_value,
        new_value=outcome.new_value,
        line_no=outcome.line_no,
        inserted=outcome.inserted,
        applied=outcome.applied,
        diff=outcome.diff,
        before_line=outcome.before_line,
        after_line=outcome.after_line,
        encoding=outcome.encoding,
        summary=summary,
    )


@mcp.tool()
def diff_netlist(before_path: str, after_path: str) -> NetlistDiff:
    """Compare two circuits by component and connectivity.

    Components are matched by reference designator, so a value change is reported
    separately from a rewiring - the two mean different things. Use this to confirm a
    fix changed what you intended and nothing else, for instance after patching a value
    or against a backup copy of the schematic.

    Args:
        before_path: The original .asc, .net or .cir.
        after_path: The circuit to compare against it.
    """
    result = diff_netlists(_load(before_path), _load(after_path))
    log.info("diff_netlist %s vs %s -> %s", before_path, after_path, result.summary)
    return result


@mcp.tool()
def export_netlist(path: str, out_path: str) -> ExportResult:
    """Write a circuit's flattened SPICE netlist to a file.

    For fixes that cannot be expressed as a value change - rewiring, adding a component -
    where the user wants an editable SPICE deck. Note that a .net is not a schematic:
    it has no coordinates, so it does not open in the LTspice schematic editor. Prefer
    patch_component_value when the fix is only a value.

    Args:
        path: Circuit to export.
        out_path: File to write. Must not already exist, so nothing is overwritten.
    """
    netlist = _load(path)

    target = Path(out_path.strip()).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.exists():
        raise ToolError(
            f"{target} already exists. Choose a new path - this tool will not overwrite "
            "a file."
        )
    if not target.parent.is_dir():
        raise ToolError(f"The folder {target.parent} does not exist.")

    text = netlist.raw_text
    if not text.endswith("\n"):
        text += "\n"
    # CRLF because that is what LTspice itself writes into a .net.
    data = text.encode("utf-8").replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    target.write_bytes(data)

    log.info("export_netlist %s -> %s (%d bytes)", path, target, len(data))
    return ExportResult(
        source_path=netlist.source_path,
        out_path=str(target),
        line_count=text.count("\n"),
        byte_count=len(data),
        summary=f"Wrote {len(data)} bytes of SPICE netlist to {target.name}.",
    )
