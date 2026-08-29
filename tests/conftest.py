from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"


def _ltspice_available() -> bool:
    try:
        from spice_mcp_server.ltspice import find_ltspice_exe

        find_ltspice_exe()
        return True
    except Exception:
        return False


#

# Tests that need a .asc converted to a netlist require the real executable. Text-level
# parser and check tests do not, and must keep passing without LTspice installed.
needs_ltspice = pytest.mark.skipif(
    not _ltspice_available(),
    reason="LTspice executable not found; set LTSPICE_EXE to run schematic tests.",
)


def parse(text: str):
    """Parse netlist text directly, bypassing LTspice."""
    from spice_mcp_server.netlist import parse_netlist_text

    return parse_netlist_text(text, Path("test.net"), Path("test.net"))


def check_ids(text: str) -> set[str]:
    """Return the set of check ids that fire on the given netlist text."""
    from spice_mcp_server.checks import run_static_checks

    return {f.check for f in run_static_checks(parse(text)).findings}
