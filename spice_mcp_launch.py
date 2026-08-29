"""Explorer entry point. Registered as the "Debug with SPICE MCP" right-click verb.

    pythonw.exe spice_mcp_launch.py "C:\\path\\to\\circuit.asc"

A script at the repo root rather than `-m spice_mcp_app.launch`, because the registry gives
us no way to set a working directory and `-m` would need the repo on PYTHONPATH. Python puts
a script's own directory on sys.path automatically, so running this file by absolute path
makes both packages importable wherever Explorer happens to start us.

The real work is in `spice_mcp_app/launch.py`; keeping this to one line means the registry
never has to be rewritten when that changes.
"""

from spice_mcp_app.launch import main

if __name__ == "__main__":
    raise SystemExit(main())
