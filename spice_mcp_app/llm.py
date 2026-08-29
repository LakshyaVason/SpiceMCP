"""TAMU AI Chat client, MCP->OpenAI tool translation, and the agent loop.

The proxy is OpenAI-compatible, with two deviations found by probing it (see
`scripts/probe_tool_calling.py`, which is the reproducer for both):

  * **`"stream": false` must be sent explicitly.** Omit it and the proxy replies with
    `text/event-stream` *and* drops the `usage` block entirely. Since per-turn token
    counts are the reason the session log exists, every request forces it off.
  * `/openai/v1/chat/completions` returns 403 ("Direct API passthrough is disabled").
    The correct path has no `/v1`.

Tool calling itself works fully on `protected.Claude Opus 4.8`: `tool_calls` come back
with parseable JSON arguments, `role: "tool"` results are accepted, and `usage` is
present on every turn. The prompted-JSON fallback the plan held in reserve is not
needed.

This module never imports `mcp`. It takes a `tool_executor` callable, which keeps the
LLM side testable with a fake and the MCP side replaceable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import requests

from .config import Config
from .session import Session, Turn

log = logging.getLogger(__name__)

# Guardrail on a runaway loop. Diagnosing a circuit takes a handful of calls
# (read_netlist -> check_netlist_static -> maybe run_simulation); a model that wants 12
# rounds is stuck, and each round costs tokens.
MAX_TOOL_ROUNDS = 12

SYSTEM_PROMPT = """\
You are an expert analog circuit engineer helping debug LTspice circuits.

You have tools that read the circuit as structured text - components, nets, values,
directives - plus a static checker and a simulator. Use them; do not ask the user to
paste a netlist or a screenshot, and never guess at a value you could look up.

An efficient order of work:
  1. read_netlist to see the topology.
  2. check_netlist_static for the cheap pass that catches most drafting mistakes.
  3. run_simulation only when you need the simulator's verdict - it is slow.

Things about LTspice that matter for a correct diagnosis:
  * A simulation that "succeeds" can still be wrong. Exit codes and the presence of a
    .raw file are unreliable; trust the parsed log and your own reading of the circuit.
  * The static checks and the simulation catch different faults. Neither subsumes the
    other, so a clean static pass is not proof the circuit works, and a clean
    simulation is not proof the circuit does what it looks like it does.
  * In SPICE, M means milli, not mega. MEG is mega. This silently produces answers that
    are wrong by 10^9.
  * A component with a correct topology can still have a wrong value. If the user
    states an intended spec (a cutoff frequency, a gain), check the values against it
    arithmetically and show the arithmetic.

When you find a fault, say plainly what is wrong, why it produces the observed
behaviour, and what the fix is - including the specific component and value. Be
concise and concrete. Show the numbers you relied on."""


class ToolExecutor(Protocol):
    """Runs a tool by name and returns its result as text for the model."""

    def __call__(self, name: str, arguments: dict[str, Any]) -> str: ...


class LLMError(RuntimeError):
    """A call to the chat API failed in a way worth showing the user."""


def mcp_tools_to_openai(mcp_tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Translate MCP tool definitions into the OpenAI `tools` array.

    MCP Python objects are snake_case (`input_schema`), unlike the wire format. Tools
    arrive from `client.list_tools()` as objects, but dicts are accepted too so tests
    can pass literals.
    """
    translated: list[dict[str, Any]] = []
    for tool in mcp_tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description") or ""
            schema = tool.get("input_schema") or tool.get("inputSchema")
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "") or ""
            schema = getattr(tool, "input_schema", None) or getattr(
                tool, "inputSchema", None
            )

        if not name:
            log.warning("skipping a tool with no name: %r", tool)
            continue

        translated.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    # An empty-but-valid object schema, because some providers reject
                    # a function whose parameters are null.
                    "parameters": schema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return translated


@dataclass
class ToolCallRecord:
    """What happened on one tool call, for the session log and the UI."""

    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "is_error": self.is_error,
        }


@dataclass
class AgentResult:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class TamuClient:
    """Thin HTTP client for the TAMU chat completions endpoint."""

    def __init__(self, config: Config, *, timeout: float = 180.0) -> None:
        self._config = config
        self._timeout = timeout
        self._http = requests.Session()

    @property
    def model(self) -> str:
        return self._config.model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """One chat completion. Returns the raw OpenAI-shaped response body."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            # Not a default on this proxy - see the module docstring.
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = self._http.post(
                self._config.chat_completions_url,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._config.api_key}",
                },
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the chat API: {exc}") from exc

        if response.status_code >= 400:
            # The body carries the proxy's own explanation, which is usually the
            # actual diagnosis (a rejected key, an unknown model, tools unsupported).
            detail = response.text[:1000]
            raise LLMError(
                f"Chat API returned HTTP {response.status_code}.\n{detail}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            content_type = response.headers.get("content-type", "?")
            raise LLMError(
                f"Chat API returned {content_type} rather than JSON, so there is no "
                f"usage block to log.\n{response.text[:500]}"
            ) from exc

        if not body.get("choices"):
            raise LLMError(f"Chat API returned no choices: {json.dumps(body)[:500]}")
        return body


def _message_text(message: dict[str, Any]) -> str:
    """Extract text from a message, tolerating content-part lists."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(parts)
    return ""


def run_agent_turn(
    client: TamuClient,
    session: Session,
    user_text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: ToolExecutor | None = None,
    history: list[dict[str, Any]] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> AgentResult:
    """Run one user turn to completion, executing tool calls as the model asks.

    `history` is the running OpenAI-format message list and is mutated in place, so the
    caller keeps conversational context across turns. The session log is updated as
    each turn completes rather than at the end, so an interrupted session still leaves
    complete token counts behind.
    """
    if history is None:
        history = []
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_text})
    session.add_turn(Turn(role="user", text=user_text))

    records: list[ToolCallRecord] = []
    total_in = 0
    total_out = 0

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        body = client.complete(history, tools=tools)
        message = body["choices"][0]["message"]
        usage = body.get("usage") or {}
        total_in += int(usage.get("prompt_tokens") or 0)
        total_out += int(usage.get("completion_tokens") or 0)

        text = _message_text(message)
        tool_calls = message.get("tool_calls") or []

        # Append the assistant message verbatim: the provider needs its own tool_calls
        # structure back unchanged to match the tool results to it.
        history.append(message)

        round_records: list[ToolCallRecord] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or "?"
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                arguments = {}
                result = f"Your arguments were not valid JSON ({exc}). Raw: {raw_args}"
                is_error = True
            else:
                if tool_executor is None:
                    result = "No tools are available in this session."
                    is_error = True
                else:
                    if on_progress:
                        on_progress(f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})")
                    try:
                        result = tool_executor(name, arguments)
                        is_error = False
                    except Exception as exc:
                        # Hand the failure to the model rather than aborting: a bad
                        # path or an unsupported file is something it can recover from
                        # by calling the tool differently.
                        log.warning("tool %s failed: %s", name, exc)
                        result = f"Tool {name} failed: {exc}"
                        is_error = True

            record = ToolCallRecord(name, arguments, result, is_error)
            round_records.append(record)
            records.append(record)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result,
                }
            )

        session.add_turn(
            Turn.from_usage(
                "assistant",
                text,
                usage,
                tool_calls=[r.to_dict() for r in round_records] or None,
            )
        )
        for record in round_records:
            session.add_turn(Turn(role="tool", text=record.result))

        if not tool_calls:
            return AgentResult(
                text=text,
                tool_calls=records,
                rounds=round_index,
                input_tokens=total_in,
                output_tokens=total_out,
            )

    # Ran out of rounds. Return what we have with a note, rather than raising - the
    # partial diagnosis and its token counts are still worth keeping.
    return AgentResult(
        text=(
            f"Stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer. "
            "The last tool results are above; try asking a narrower question."
        ),
        tool_calls=records,
        rounds=MAX_TOOL_ROUNDS,
        input_tokens=total_in,
        output_tokens=total_out,
    )
