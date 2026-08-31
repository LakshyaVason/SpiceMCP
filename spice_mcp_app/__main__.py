"""Entry point for the desktop app.

    python -m spice_mcp_app

Opens a pywebview window backed by `web/`, with `api.Api` exposed to JavaScript. The
window runs on the main thread (a platform requirement on Windows); the MCP client keeps
its own asyncio loop on a background thread, and pywebview dispatches `js_api` calls on
worker threads, so a long agent turn does not freeze the UI.

Logging goes to stderr. Unlike the server there is no protocol on stdout here, but the
convention is kept so the two halves behave the same way.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"


def build_parser() -> argparse.ArgumentParser:
    """Separate from main() so the flags can be asserted without opening a window."""
    parser = argparse.ArgumentParser(prog="python -m spice_mcp_app")
    parser.add_argument(
        "--folder",
        help="Pre-select a folder of circuits, skipping the picker.",
    )
    parser.add_argument(
        "--file",
        help="Open one circuit directly: selects it and runs the static checks on load. "
        "Its folder populates the sidebar.",
    )
    parser.add_argument("--debug", action="store_true", help="Open the web inspector.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    startup_note: str | None = None,
    opened_in_ltspice: bool = False,
) -> int:
    """Open the window.

    `startup_note` carries a non-fatal problem from before the window existed - the
    Explorer launcher uses it to report that LTspice could not be started. It goes into
    the UI banner, which is the only place the user will see it when launched by
    pythonw.exe with no console attached.

    `opened_in_ltspice` says we just opened `--file` in the GUI ourselves, so the file on
    disk and the file on screen are the same and the "save before asking" warning would be
    false. See `Api._ltspice_open_warning`.
    """
    args = build_parser().parse_args(argv)

    # No-op when launch.py already configured the root logger, which is deliberate - it has
    # a file handler this would otherwise not replace. `handlers=` rather than `stream=`
    # because under pythonw.exe sys.stderr is None, and a StreamHandler built on None
    # discards every record without a word.
    logging.basicConfig(
        level=os.environ.get("SPICE_MCP_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr)
            if sys.stderr is not None
            else logging.NullHandler()
        ],
    )

    import webview

    from .api import Api

    api = Api()
    window = webview.create_window(
        "SPICE MCP client",
        url=str(WEB_DIR / "index.html"),
        js_api=api,
        width=1180,
        height=820,
        min_size=(900, 600),
        background_color="#11131a",
        text_select=True,
    )
    # Through the setter, and onto a private attribute. Assigning the Window to a public
    # attribute of `api` makes pywebview's bridge builder recurse into the native WinForms
    # form and never finish, which leaves the window open but completely dead. See
    # Api._attach_window.
    api._attach_window(window)
    api.startup_note = startup_note

    # Both are read by the UI on load, so --folder behaves as if it had been picked and
    # --file additionally as if the circuit had been clicked.
    folder = args.folder
    if args.file:
        circuit = Path(args.file).resolve()
        api.initial_circuit = str(circuit)
        api.opened_in_ltspice = opened_in_ltspice
        # Populate the sidebar from the circuit's own folder, so the user can still switch
        # to a sibling schematic without reaching for the picker.
        folder = folder or str(circuit.parent)
    if folder:
        api.initial_folder = str(Path(folder).resolve())

    try:
        webview.start(debug=args.debug)
    finally:
        api.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
