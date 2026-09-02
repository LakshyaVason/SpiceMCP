"""Bedrock client, MCP->Anthropic tool mapping, and the agent loop.

The transport is AWS Bedrock through `anthropic.AnthropicBedrock`, which speaks the
native Messages API. That means tool calling needs no translation layer: an MCP tool
definition is already `{name, description, input_schema}`, which is exactly what
Anthropic's `tools` parameter wants.

Facts about this transport that the code depends on:

  * **`max_tokens` is required on every request.** There is no default.
  * **Never pass `thinking`.** Adaptive thinking is on by default for Opus 5, and
    `thinking: {"type": "disabled"}` makes the model occasionally write a tool call into
    its *visible text* instead of a `tool_use` block - the turn succeeds, the call never
    runs, and nothing raises. That is precisely the failure this module was rewritten to
    fix, so leaving thinking alone is load-bearing. `budget_tokens` is rejected with a
    400 on Opus 5 and must not be reintroduced either.
  * **Thinking blocks come back in `content` and must be handed back unchanged** when the
    same turn carries a `tool_use`. The loop appends `response.content` verbatim, which
    covers it; `_response_text` filters them out of what the user sees.
  * **All results from one round go back in a single user message.** The model may emit
    several `tool_use` blocks at once, and splitting their `tool_result` blocks across
    messages is rejected.
  * **Consecutive same-role turns are rejected.** `append_user_note` exists for that -
    see its docstring.

This module never imports `mcp`. It takes a `tool_executor` callable, which keeps the
LLM side testable with a fake and the MCP side replaceable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import anthropic
from anthropic import AnthropicBedrock

from .config import Config
from .session import Session, Turn, usage_value

log = logging.getLogger(__name__)

# Guardrail on a runaway loop. Diagnosing a circuit takes a handful of calls
# (read_netlist -> check_netlist_static -> maybe run_simulation); a model that wants 12
# rounds is stuck, and each round costs tokens.
MAX_TOOL_ROUNDS = 12

# Required by the API, and it has to leave room for thinking tokens as well as the answer.
MAX_TOKENS = 16000

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
    """A call to the model failed in a way worth showing the user."""


def mcp_tools_to_anthropic(mcp_tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Map MCP tool definitions onto Anthropic's `tools` parameter.

    Barely a translation: MCP Python objects are already snake_case (`input_schema`),
    which is the name Anthropic uses too. Tools arrive from `client.list_tools()` as
    objects, but dicts are accepted so tests can pass literals.
    """
    mapped: list[dict[str, Any]] = []
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

        mapped.append(
            {
                "name": name,
                "description": description,
                # input_schema is required, not optional - a tool without one is
                # rejected outright, so the empty-but-valid object is load-bearing.
                "input_schema": schema or {"type": "object", "properties": {}},
            }
        )
    return mapped


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


class BedrockClient:
    """Thin wrapper over AnthropicBedrock, and the one place errors become LLMError."""

    def __init__(self, config: Config, *, timeout: float = 180.0) -> None:
        self._config = config
        # Credentials are left to the SDK's default chain on purpose - see config.py.
        self._client = AnthropicBedrock(
            aws_region=config.aws_region, timeout=timeout
        )

    @property
    def model(self) -> str:
        return self._config.model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str = SYSTEM_PROMPT,
        max_tokens: int = MAX_TOKENS,
    ) -> Any:
        """One turn. Returns the SDK's Message object."""
        try:
            return self._client.messages.create(
                model=self._config.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools or anthropic.NOT_GIVEN,
            )
        except (anthropic.NotFoundError, anthropic.PermissionDeniedError) as exc:
            # By far the most likely first-run failure, and the SDK's own message does not
            # say what to do about it. Both shapes mean the same thing in practice.
            raise LLMError(
                f"Bedrock will not serve {self._config.model} in "
                f"{self._config.aws_region}.\n\n"
                "Either the inference profile does not exist in that region or the "
                "account has not been granted access to it. Check with "
                "`python scripts/list_bedrock_models.py`, then enable the model in the "
                "Bedrock console or set SPICE_MCP_AWS_REGION to a region where it is "
                f"enabled.\n\n{exc}"
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Bedrock rejected the credentials "
                f"({self._config.credentials_source}).\n\n{exc}"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError(f"The Bedrock call failed: {exc}") from exc
        except Exception as exc:
            # botocore raises its own exceptions during credential resolution and SigV4
            # signing, before the anthropic layer sees anything.
            raise LLMError(f"Could not reach Bedrock: {exc}") from exc


def _block_field(block: Any, name: str) -> Any:
    """Read a field from a content block, tolerating dicts as well as SDK objects."""
    if isinstance(block, Mapping):
        return block.get(name)
    return getattr(block, name, None)


def _response_text(response: Any) -> str:
    """Join the visible text of a response, skipping thinking and tool_use blocks."""
    content = _block_field(response, "content")
    if isinstance(content, str):
        return content
    if not content:
        return ""
    parts = [
        _block_field(block, "text") or ""
        for block in content
        if _block_field(block, "type") == "text"
    ]
    return "".join(parts)


def append_user_note(history: list[dict[str, Any]], text: str) -> None:
    """Add user-role text to the history without creating two consecutive user turns.

    The Messages API rejects consecutive same-role messages, so a note the app wants the
    model to see - "the user opened this circuit", "the user approved your fix" - cannot
    simply be appended: the next question would append a second user turn behind it and
    the request would be refused. Folding it into the pending turn keeps the note intact
    and the history valid, and means callers never have to think about ordering.
    """
    if history and history[-1].get("role") == "user":
        content = history[-1].get("content")
        if isinstance(content, str):
            history[-1]["content"] = f"{content}\n\n{text}"
            return
        if isinstance(content, list):
            content.append({"type": "text", "text": text})
            return
    history.append({"role": "user", "content": text})


def run_agent_turn(
    client: BedrockClient,
    session: Session,
    user_text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: ToolExecutor | None = None,
    history: list[dict[str, Any]] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> AgentResult:
    """Run one user turn to completion, executing tool calls as the model asks.

    `history` is the running Messages-API message list and is mutated in place, so the
    caller keeps conversational context across turns. The system prompt is *not* in
    there - it is a top-level parameter on every request. The session log is updated as
    each turn completes rather than at the end, so an interrupted session still leaves
    complete token counts behind.
    """
    if history is None:
        history = []

    append_user_note(history, user_text)
    session.add_turn(Turn(role="user", text=user_text))

    records: list[ToolCallRecord] = []
    total_in = 0
    total_out = 0

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.complete(history, tools=tools)
        usage = _block_field(response, "usage")
        total_in += int(usage_value(usage, "input_tokens") or 0)
        total_out += int(usage_value(usage, "output_tokens") or 0)

        text = _response_text(response)
        stop_reason = _block_field(response, "stop_reason")
        content = _block_field(response, "content") or []
        tool_uses = [b for b in content if _block_field(b, "type") == "tool_use"]

        # Append the content verbatim: the tool_use blocks have to come back unchanged for
        # the results to match up, and so do any thinking blocks alongside them.
        history.append({"role": "assistant", "content": content})

        round_records: list[ToolCallRecord] = []
        results: list[dict[str, Any]] = []
        for block in tool_uses:
            name = _block_field(block, "name") or "?"
            # The API parses tool input for us, so this is a dict in practice. The check is
            # not decoration: the approval gate in api.py reads arguments.get("apply"), and
            # a non-dict would make that check silently pass, so never hand one onward.
            raw_input = _block_field(block, "input")
            valid_input = isinstance(raw_input, Mapping)
            arguments = dict(raw_input) if valid_input else {}

            if not valid_input:
                log.warning("tool %s sent non-dict input: %r", name, raw_input)
                result = (
                    f"Your input for {name} was not an object. Send the arguments as "
                    "a JSON object and try again."
                )
                is_error = True
            elif tool_executor is None:
                result = "No tools are available in this session."
                is_error = True
            else:
                if on_progress:
                    on_progress(
                        f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
                    )
                try:
                    result = tool_executor(name, arguments)
                    is_error = False
                except Exception as exc:
                    # Hand the failure to the model rather than aborting: a bad path or
                    # an unsupported file is something it can recover from by calling the
                    # tool differently.
                    log.warning("tool %s failed: %s", name, exc)
                    result = f"Tool {name} failed: {exc}"
                    is_error = True

            record = ToolCallRecord(name, arguments, result, is_error)
            round_records.append(record)
            records.append(record)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": _block_field(block, "id"),
                    "content": result,
                    "is_error": is_error,
                }
            )

        if results:
            # One message carrying every result from this round. Splitting them is rejected.
            history.append({"role": "user", "content": results})

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

        if stop_reason != "tool_use":
            if stop_reason == "max_tokens":
                # Say so rather than presenting a truncated diagnosis as a finished one.
                text = (
                    f"{text}\n\n[Cut off at the {MAX_TOKENS}-token limit, so this answer "
                    "is incomplete. Try asking a narrower question.]"
                ).strip()
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
