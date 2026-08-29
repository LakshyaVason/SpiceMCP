"""Entry point: run the SPICE MCP server over stdio.

    python -m spice_mcp_server

Blocks for the life of the server, reading MCP traffic on stdin. Producing no output
and not exiting is the correct behaviour - it is waiting for a host to connect.

Logging goes to stderr, never stdout, because stdout carries the protocol.
"""

from __future__ import annotations

import logging
import os
import sys

from .server import mcp


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("SPICE_MCP_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # No argument means stdio transport.
    mcp.run()


if __name__ == "__main__":
    main()
