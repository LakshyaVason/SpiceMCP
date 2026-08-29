"""SPICE MCP server: read, check, simulate and patch LTspice files.

This package deliberately knows nothing about LLMs. It exposes MCP tools over stdio
and is reusable by any MCP host (Claude Code, Claude Desktop, or the sibling
spice_mcp_app). Keeping the LLM out of here is what makes the approach portable to
other EDA tools later.
"""

__version__ = "0.1.0"
