"""Sync wrapper around the async MCP stdio client.

`mcp.Client` is async and is designed to be held open as a context manager for the life
of the connection. pywebview's JS bridge calls us on plain worker threads, so this
module runs one asyncio loop on a dedicated background thread, keeps the client open on
it, and hands the UI thread ordinary blocking methods.

Holding the connection open matters: it keeps a single server subprocess alive across
tool calls instead of paying process startup - and LTspice's scratch directory setup -
on every call.

Note the mcp 2.1 shapes, which differ from most examples online:
  * `list_tools()` returns a `ListToolsResult`; the list is on `.tools`.
  * `Tool.input_schema` is snake_case on the Python object.
  * `call_tool()` returns a `CallToolResult` with `.content`, `.structured_content` and
    `.is_error`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .config import REPO_ROOT, TOKEN_ENV_VAR

log = logging.getLogger(__name__)

# Structured content is what the tools are for, but a huge netlist dump costs real
# tokens. Truncate defensively; the model can ask for a narrower view.
MAX_RESULT_CHARS = 60_000

# Withheld from the server subprocess. These are the app half's business only, and the
# gateway token in particular has no reason to exist in a process that knows nothing about
# LLMs - forwarding it would widen the credential's exposure for no benefit.
#
# TOKEN_ENV_VAR is *imported* rather than spelled out again: the same name has to appear in
# config.py, and a rename that updated one copy and not the other would leak the token
# while every test still passed.
#
# The AWS_ prefix rule is kept even though this app no longer uses AWS. It costs nothing,
# and if a user's shell happens to carry AWS credentials for unrelated work there is still
# no reason to hand them to the netlist parser. It is belt-and-braces, not the load-bearing
# rule - a period when it *was* the only rule is exactly when TOKEN_ENV_VAR went missing
# from this set and the token started reaching the server.
_LLM_ONLY_ENV = frozenset(
    {
        TOKEN_ENV_VAR,
        "SPICE_MCP_MODEL",
        "SPICE_MCP_BASE_URL",
        "SPICE_MCP_TOOL_MODE",
    }
)
_LLM_ONLY_PREFIXES = ("AWS_",)


def _is_llm_only(name: str) -> bool:
    return name in _LLM_ONLY_ENV or name.startswith(_LLM_ONLY_PREFIXES)


class MCPClientError(RuntimeError):
    """The server could not be started or a call failed at the transport level."""


def _server_params() -> StdioServerParameters:
    """Launch the sibling server package with this same interpreter.

    sys.executable rather than a hardcoded `.venv\\Scripts\\python.exe` so the app and
    the server can never end up on different interpreters, and cwd is pinned to the
    repo root so relative circuit paths resolve the way the user expects.

    The environment is *extended*, not replaced. Passing a bare two-key dict silently
    dropped LTSPICE_EXE - and PATH, and SPICE_MCP_WORKDIR - so a user who overrode their
    LTspice location in `.env` (which `.env.example` says works) got the default search
    inside the server anyway.

    Extended minus the LLM variables, though. The server must never learn what an LLM is,
    and that is not only an architectural line: forwarding the gateway token would hand the
    credential to a process with no use for it, widening its exposure for nothing.

    Ordering note: LTSPICE_EXE only reaches os.environ once `load_dotenv` has run inside
    `load_config()`. `Api.__init__` does that before `start()` gets here, so the app path is
    fine; a bare `SpiceMCP()` with no prior `load_config()` would not be.
    """
    env = {k: v for k, v in os.environ.items() if not _is_llm_only(k)}
    env["SPICE_MCP_LOG_LEVEL"] = os.environ.get("SPICE_MCP_LOG_LEVEL", "INFO")
    # The server's stdout is the MCP wire and its stderr carries LTspice's Ω/µ/°.
    env["PYTHONIOENCODING"] = "utf-8"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "spice_mcp_server"],
        cwd=str(REPO_ROOT),
        env=env,
    )


def result_to_text(result: Any) -> str:
    """Flatten a CallToolResult into text for the model.

    Prefers `structured_content`, which is the whole point of the typed output schemas -
    the model gets JSON it can reason over rather than a prose summary.
    """
    structured = getattr(result, "structured_content", None)
    if structured:
        text = json.dumps(structured, indent=2, ensure_ascii=False, default=str)
    else:
        parts: list[str] = []
        for block in getattr(result, "content", None) or []:
            block_text = getattr(block, "text", None)
            if block_text:
                parts.append(block_text)
            else:
                parts.append(str(block))
        text = "\n".join(parts)

    if len(text) > MAX_RESULT_CHARS:
        text = (
            text[:MAX_RESULT_CHARS]
            + f"\n\n[truncated: result was {len(text)} characters]"
        )
    return text or "(the tool returned no content)"


class SpiceMCP:
    """A running spice-mcp server, driven synchronously."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Client | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._tools: list[Any] = []

    # --- lifecycle -------------------------------------------------------------------

    def start(self, timeout: float = 60.0) -> list[Any]:
        """Spawn the server, complete the handshake, and return its tool list."""
        if self._thread is not None:
            return self._tools

        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-client", daemon=True
        )
        self._thread.start()

        if not self._ready.wait(timeout):
            raise MCPClientError(
                f"The MCP server did not become ready within {timeout:.0f}s."
            )
        if self._error is not None:
            raise MCPClientError(f"Could not start the MCP server: {self._error}")

        log.info(
            "MCP server ready with %d tools: %s",
            len(self._tools),
            ", ".join(getattr(t, "name", "?") for t in self._tools),
        )
        return self._tools

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 - reported to start()
            self._error = exc
            log.exception("MCP client thread died")
        finally:
            # Signal even on failure, so start() reports the error instead of timing out.
            self._ready.set()
            try:
                loop.close()
            finally:
                self._loop = None

    async def _serve(self) -> None:
        # raise_exceptions=False so a ToolError comes back as a result with is_error
        # set, carrying the server's message. That message is often the diagnosis
        # itself, and we want the model to read it rather than see a transport crash.
        async with Client(_server_params(), raise_exceptions=False) as client:
            self._client = client
            self._tools = list((await client.list_tools()).tools)
            self._ready.set()
            # Park until close() is called. The client must stay inside this `async
            # with` for the subprocess to stay alive.
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        self._client = None

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10)
        self._thread = None
        self._ready.clear()

    # --- calls -----------------------------------------------------------------------

    @property
    def tools(self) -> list[Any]:
        return self._tools

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 300.0) -> str:
        """Call a tool and return its result as text."""
        loop = self._loop
        client = self._client
        if loop is None or client is None:
            raise MCPClientError("The MCP server is not running.")

        future: Future = asyncio.run_coroutine_threadsafe(
            client.call_tool(name, arguments), loop
        )
        try:
            result = future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise MCPClientError(
                f"{name} did not return within {timeout:.0f}s."
            ) from exc

        text = result_to_text(result)
        if getattr(result, "is_error", False):
            # Surfaced as text, not raised: the agent loop passes it to the model,
            # which can often recover by calling the tool differently.
            log.info("tool %s reported an error: %s", name, text[:200])
        return text

    def __enter__(self) -> SpiceMCP:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def find_circuits(folder: str | Path) -> list[dict[str, Any]]:
    """List circuit files in a folder, for the app's folder picker.

    `.asc` first because it is the source of truth; `.net`/`.cir` are included but a
    `.net` sitting beside a `.asc` is usually an ExpressPCB export, so the schematic
    should be preferred.
    """
    root = Path(folder)
    if not root.is_dir():
        raise MCPClientError(f"{root} is not a folder.")

    found: list[dict[str, Any]] = []
    for suffix in (".asc", ".net", ".cir"):
        for path in sorted(root.glob(f"*{suffix}")):
            found.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "suffix": suffix,
                    "size": path.stat().st_size,
                }
            )
    return found
