"""Open a schematic in LTspice and in this client, together.

This is what the Explorer right-click verb runs (see `scripts/install_context_menu.py`):

    pythonw.exe <repo>\\spice_mcp_launch.py "C:\\path\\to\\circuit.asc"

It starts LTspice on the file, detached so it outlives us, then opens the app window with
that circuit already selected and its static checks already rendered. The binding is made
once, here, at launch - nothing polls LTspice afterwards to see what the user has open.

Running under `pythonw.exe` shapes the code below, and more than it first appears.
Verified by launching a console-less detached `pythonw`: `sys.stdout`, `sys.stderr` **and**
`sys.stdin` are all `None`. So `logging.StreamHandler(sys.stderr)` builds a handler whose
stream is `None` and every record it is given fails silently - which is why logging here is
configured against a *file* and a stream handler is added only when a stream actually
exists. A fatal failure before the window exists gets a message box, because there is no
console for a traceback to land in and no shell to see a non-zero exit.

A failure that still leaves a usable app - LTspice missing, most likely - is passed to the
window as a `startup_note` rather than stopping the launch.

The MCP handshake was the real worry here, since `_server_params()` spawns `sys.executable`,
which under this launcher is `pythonw.exe`. Verified working: the stdio client hands the
child explicit pipes, so a console-less parent makes no difference. All seven tools connect
and a `.asc` static check returns findings.

**On the "never write into the user's source directory" rule:** handing the user's own path
to the LTspice *GUI* is not a violation of it. That rule exists because our automated
`-netlist`/`-b` invocations drop artifacts next to their input, which is why those stage
into `%TEMP%` first. Opening a schematic for editing is the user's normal workflow and is
exactly what double-clicking the file already does. Do not "fix" this into a staged copy:
the user would then be editing a temp file that we later overwrite.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import REPO_ROOT

log = logging.getLogger(__name__)

LOG_PATH = REPO_ROOT / "launch.log"


def message_box(text: str, title: str = "SPICE MCP client") -> None:
    """Show a modal message box, for failures that happen before the window exists.

    ctypes rather than pywin32 or tkinter: no new dependency, and it works even when the
    failure is that the app's own imports are broken. Silently gives up if the call itself
    fails - there is nowhere left to report to at that point.
    """
    try:
        import ctypes

        # 0x10 = MB_ICONERROR, 0x40000 = MB_TOPMOST.
        ctypes.windll.user32.MessageBoxW(None, str(text), str(title), 0x10 | 0x40000)
    except Exception:  # pragma: no cover - non-Windows, or no window station
        log.error("%s: %s", title, text)


def _configure_logging(debug: bool) -> None:
    """Log to a file, and to stderr only if there is one.

    Under `pythonw.exe` with no console, `sys.stderr` is None, and a StreamHandler built on
    it swallows every record without complaint. The file is therefore the primary sink, not
    a backup. `force=True` claims the root logger before `__main__.main` runs, whose own
    `basicConfig` then becomes the documented no-op - so its `stream=sys.stderr` never gets
    the chance to install a broken handler.
    """
    handlers: list[logging.Handler] = []
    try:
        handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
    except OSError:  # pragma: no cover - read-only checkout
        pass
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers or [logging.NullHandler()],
        force=True,
    )


def validate(raw: str) -> Path:
    """Resolve the argument to an existing schematic, or raise ValueError."""
    circuit = Path(raw).expanduser()
    try:
        circuit = circuit.resolve()
    except OSError as exc:
        raise ValueError(f"Could not resolve {raw!r}: {exc}") from exc

    if not circuit.exists():
        raise ValueError(f"No such file:\n\n{circuit}")
    if not circuit.is_file():
        raise ValueError(f"Not a file:\n\n{circuit}")
    # Case-insensitive: Explorer will happily hand back RCLP.ASC.
    if circuit.suffix.lower() != ".asc":
        raise ValueError(
            f"This opens LTspice schematics (.asc), and that is a "
            f"{circuit.suffix or 'extensionless'} file:\n\n{circuit}"
        )
    return circuit


def open_in_ltspice(circuit: Path) -> None:
    """Start the LTspice GUI on `circuit`, detached from this process.

    Detached, in its own process group, and with its stdio on DEVNULL. Each of those is
    load-bearing: without a new process group LTspice would die with us when the app window
    closes; a pipe nobody drains could block it once the buffer filled; and DEVNULL rather
    than inheritance because under `pythonw` our own handles are None, which is not
    something to hand a child. That is also why this does not reuse `ltspice._invoke`, which
    pipes and then blocks in `communicate()` - right for a batch run, wrong for a GUI.

    Note the argv is exactly `[exe, circuit]`. With no batch switch LTspice writes nothing
    until the user asks it to, which is what makes passing the user's own path safe here.
    """
    from spice_mcp_server.ltspice import find_ltspice_exe

    exe = find_ltspice_exe()
    log.info("opening %s in %s", circuit.name, exe)
    # getattr because these constants are Windows-only and this module must stay importable
    # everywhere the test suite can run.
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    subprocess.Popen(
        [str(exe), str(circuit)],
        cwd=str(circuit.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spice_mcp_launch",
        description="Open a .asc in LTspice and in the SPICE MCP client together.",
    )
    parser.add_argument("circuit", help="Path to the .asc schematic.")
    parser.add_argument("--debug", action="store_true", help="Verbose log + web inspector.")
    parser.add_argument(
        "--no-ltspice",
        action="store_true",
        help="Open only the client. Useful when LTspice is already showing the file.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.debug)

    try:
        circuit = validate(args.circuit)
    except ValueError as exc:
        log.error("%s", exc)
        message_box(str(exc))
        return 2

    # Explorer starts us with the cwd set to the user's circuit folder. Holding it locks
    # that directory against rename, and makes every relative path in the app resolve into
    # the user's source tree - exactly what this project promises not to write to. Must come
    # after the resolve() above, or a relative argument from a shell would resolve wrongly.
    try:
        os.chdir(REPO_ROOT)
    except OSError:  # pragma: no cover
        log.warning("could not change directory to %s", REPO_ROOT)

    note: str | None = None
    opened_gui = False
    if not args.no_ltspice:
        try:
            open_in_ltspice(circuit)
            opened_gui = True
        except Exception as exc:
            # Not fatal. Reading, checking and patching the schematic all work without the
            # GUI, so report it in the window rather than refusing to start.
            log.warning("could not start LTspice: %s", exc)
            note = f"Could not start LTspice, so only this client opened. {exc}"

    from .__main__ import main as app_main

    app_argv = ["--file", str(circuit)]
    if args.debug:
        app_argv.append("--debug")
    return app_main(app_argv, startup_note=note, opened_in_ltspice=opened_gui)


if __name__ == "__main__":
    raise SystemExit(main())
