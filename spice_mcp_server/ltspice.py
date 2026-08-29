"""Locating and invoking the LTspice executable.

Windows only for v1, by decision. Other platforms raise a clear error instead of
guessing at a Wine setup that cannot be tested.

Command line switches used here are taken from LTspice's own
LTspiceHelp/commandlineswitches.htm (verified against LTspice 26.0.2), not from
third-party blog posts:

    -b          run in batch mode; leaves results in <file>.raw and <file>.log
    -netlist    batch-convert a schematic to a SPICE netlist
    -ascii      write ASCII .raw files (slower, but human-readable)
    -I<path>    add a directory to the symbol/library search path.
                Must be the last option, and takes no space before <path>.
    -version    print version

Note that -PCBnetlist produces an *ExpressPCB* netlist, not a SPICE one. That is what
the GUI's "Export Netlist" menu item writes, and it is not parseable as SPICE - see
netlist.py for the detection of that format.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# LTspice can sit in several places depending on installer vintage. The per-user
# AppData location is where the modern (24+) installer puts it; the Program Files
# paths cover older XVII/IV installs.
_CANDIDATE_PATHS = (
    r"%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe",
    r"%PROGRAMFILES%\ADI\LTspice\LTspice.exe",
    r"%PROGRAMFILES%\LTC\LTspiceXVII\XVIIx64.exe",
    r"%PROGRAMFILES(X86)%\LTC\LTspiceIV\scad3.exe",
    r"%PROGRAMFILES%\LTC\LTspiceIV\scad3.exe",
)


class LTSpiceError(RuntimeError):
    """Base class for LTspice invocation problems."""


class UnsupportedPlatform(LTSpiceError):
    """Raised on non-Windows platforms."""


class LTSpiceNotFound(LTSpiceError):
    """Raised when no LTspice executable could be located."""


@dataclass
class BatchResult:
    """Outcome of an LTspice batch run.

    LTspice exits 0 even for some simulations that failed to converge, so callers
    must not treat `returncode == 0` as success. Parse the log instead - that is
    what read_sim_log is for.
    """

    circuit_path: str
    returncode: int
    timed_out: bool
    log_path: str | None = None
    raw_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    extra_outputs: list[str] = field(default_factory=list)


def require_windows() -> None:
    if sys.platform != "win32":
        raise UnsupportedPlatform(
            f"LTspice automation is Windows-only in this version (detected {sys.platform!r}). "
            "There is no native Linux build; macOS and Wine support are out of scope for v1. "
            "Reading and statically checking an existing .net/.cir file still works on any "
            "platform - only .asc conversion and simulation need the executable."
        )


def find_ltspice_exe() -> Path:
    """Locate LTspice.exe.

    Order: LTSPICE_EXE override, then known install locations, then whatever
    spicelib's own detection found. Raises LTSpiceNotFound with actionable text.
    """
    override = os.environ.get("LTSPICE_EXE")
    if override:
        candidate = Path(os.path.expandvars(override)).expanduser()
        if candidate.is_file():
            return candidate
        raise LTSpiceNotFound(
            f"LTSPICE_EXE is set to {override!r} but that file does not exist."
        )

    require_windows()

    for raw in _CANDIDATE_PATHS:
        expanded = os.path.expandvars(raw)
        # expandvars leaves unknown %VARS% untouched; skip those.
        if "%" in expanded:
            continue
        candidate = Path(expanded)
        if candidate.is_file():
            return candidate

    # spicelib does its own search and may know a location we don't.
    try:
        from spicelib.simulators.ltspice_simulator import LTspice

        for entry in LTspice.spice_exe or []:
            candidate = Path(entry)
            if candidate.is_file():
                return candidate
    except Exception:  # pragma: no cover - spicelib import/detection is best-effort
        log.debug("spicelib LTspice detection unavailable", exc_info=True)

    raise LTSpiceNotFound(
        "Could not find LTspice.exe. Searched:\n  "
        + "\n  ".join(_CANDIDATE_PATHS)
        + "\n\nSet the LTSPICE_EXE environment variable to the full path of LTspice.exe."
    )


def ltspice_is_running() -> bool:
    """True if an LTspice GUI process is alive right now.

    Used only to warn the user: LTspice reads a .asc once, at open, and writes its own
    in-memory copy back on save. It will not notice that we patched the file underneath
    it, so a save from the GUI silently reverts the fix. Knowing whether the GUI is up
    is what lets the app say so instead of leaving the user to discover it.

    Deliberately not an MCP tool - the model has no use for it, and an eighth tool
    schema would be paid for in every request. Best-effort by design: any psutil
    problem returns False, because a missing warning is better than a false one.
    """
    names = {"ltspice.exe", "xviix64.exe", "scad3.exe"}
    try:
        import psutil

        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if name in names:
                return True
    except Exception:
        log.debug("could not enumerate processes to check for LTspice", exc_info=True)
    return False


def ltspice_version() -> str:
    """Return the LTspice version string, for diagnostics."""
    exe = find_ltspice_exe()
    proc = subprocess.run(
        [str(exe), "-version"], capture_output=True, text=True, timeout=30
    )
    return (proc.stdout or proc.stderr).strip() or f"(no output from {exe})"


def work_dir_for(source: Path) -> Path:
    """Return a private scratch directory for a given source file.

    LTspice writes -netlist and -b output *next to the input file*. Running it
    directly on the user's schematic would overwrite their own .net file - which in
    this repo is a hand-exported ExpressPCB netlist that we must not destroy. So we
    always copy the input into a scratch dir first.

    Keyed by a hash of the absolute path so repeated calls reuse one directory
    rather than filling the temp folder.
    """
    override = os.environ.get("SPICE_MCP_WORKDIR")
    base = Path(override) if override else Path(tempfile.gettempdir()) / "spice_mcp_work"
    # gettempdir() can hand back an 8.3 short path (C:\Users\EXPERT~1\...). Those paths
    # end up in tool output the model reads and quotes back to the user, so normalise.
    base = base.resolve()
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    target = base / f"{source.stem}_{digest}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its children.

    A non-converging LTspice run can ignore a plain terminate and linger, holding a
    lock on the .raw file. psutil arrives as a spicelib dependency; fall back to
    kill() if it is somehow missing.
    """
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
        parent.kill()
    except Exception:
        log.debug("psutil tree-kill failed; falling back to kill()", exc_info=True)
        try:
            proc.kill()
        except OSError:
            pass


def _invoke(args: list[str], timeout_s: float, cwd: Path) -> tuple[int, str, str, bool]:
    """Run LTspice, enforcing a timeout and cleaning up orphans.

    Returns (returncode, stdout, stderr, timed_out).
    """
    log.info("running: %s", " ".join(args))
    proc = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        log.warning("LTspice exceeded %.1fs timeout; killing process tree", timeout_s)
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            stdout, stderr = "", ""
        return proc.returncode if proc.returncode is not None else -1, stdout or "", stderr or "", True


def _stage(source: Path) -> tuple[Path, Path]:
    """Copy `source` into its scratch dir. Returns (work_dir, staged_file)."""
    work = work_dir_for(source)
    staged = work / source.name
    shutil.copy2(source, staged)
    return work, staged


def generate_netlist(asc_path: Path, timeout_s: float = 60.0) -> Path:
    """Convert a .asc schematic to a SPICE netlist via `LTspice.exe -netlist`.

    The schematic is staged into a scratch directory so the user's own .net file is
    never overwritten. The original directory is added to the symbol search path with
    -I so that custom .asy symbols and relative .include/.lib references still
    resolve from the staged copy.
    """
    asc_path = asc_path.resolve()
    if not asc_path.is_file():
        raise FileNotFoundError(f"Schematic not found: {asc_path}")

    exe = find_ltspice_exe()
    work, staged = _stage(asc_path)

    # -I goes LAST, after the filename, and takes no space before the path. This is
    # not a style preference: with -I placed before the file, LTspice hangs forever
    # instead of converting (verified against 26.0.2).
    args = [str(exe), "-netlist", str(staged), f"-I{asc_path.parent}"]
    returncode, stdout, stderr, timed_out = _invoke(args, timeout_s, work)

    produced = staged.with_suffix(".net")
    if timed_out:
        raise LTSpiceError(
            f"LTspice -netlist timed out after {timeout_s:.0f}s on {asc_path.name}."
        )
    if not produced.is_file():
        raise LTSpiceError(
            f"LTspice -netlist produced no .net file for {asc_path.name} "
            f"(exit {returncode}).\nstdout: {stdout.strip()[:400]}\n"
            f"stderr: {stderr.strip()[:400]}"
        )
    return produced


def run_batch(circuit_path: Path, timeout_s: float = 120.0, ascii_raw: bool = False) -> BatchResult:
    """Run a simulation in batch mode (`-b`) and return the output paths.

    Accepts a .asc, .net or .cir. The file is staged into a scratch directory, so the
    .raw/.log land there rather than beside the user's source.
    """
    circuit_path = circuit_path.resolve()
    if not circuit_path.is_file():
        raise FileNotFoundError(f"Circuit not found: {circuit_path}")

    exe = find_ltspice_exe()
    work, staged = _stage(circuit_path)

    # Remove stale artifacts so we never report a previous run's results.
    for suffix in (".raw", ".log", ".op.raw", ".fra"):
        stale = staged.with_suffix(suffix)
        if stale.exists():
            stale.unlink()

    args = [str(exe), "-b"]
    if ascii_raw:
        args.append("-ascii")
    # -I must be the final argument, after the filename. See generate_netlist.
    args += [str(staged), f"-I{circuit_path.parent}"]

    returncode, stdout, stderr, timed_out = _invoke(args, timeout_s, work)

    log_path = staged.with_suffix(".log")
    raw_path = staged.with_suffix(".raw")
    extra = [
        str(p)
        for p in sorted(work.glob(f"{staged.stem}.*"))
        if p.suffix not in {".asc", ".net", ".cir", ".log", ".raw"}
    ]

    return BatchResult(
        circuit_path=str(circuit_path),
        returncode=returncode,
        timed_out=timed_out,
        log_path=str(log_path) if log_path.is_file() else None,
        raw_path=str(raw_path) if raw_path.is_file() else None,
        stdout=stdout,
        stderr=stderr,
        extra_outputs=extra,
    )
