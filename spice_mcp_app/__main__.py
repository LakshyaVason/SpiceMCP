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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spice_mcp_app")
    parser.add_argument(
        "--folder",
        help="Pre-select a folder of circuits, skipping the picker.",
    )
    parser.add_argument("--debug", action="store_true", help="Open the web inspector.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("SPICE_MCP_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
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
    api.window = window
    if args.folder:
        # Read by the UI on load so --folder behaves as if it had been picked.
        api.initial_folder = str(Path(args.folder).resolve())  # type: ignore[attr-defined]

    try:
        webview.start(debug=args.debug)
    finally:
        api.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
