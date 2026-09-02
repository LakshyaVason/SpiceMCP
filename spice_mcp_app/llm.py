"""TAMU AI Chat client, MCP->OpenAI tool translation, and the agent loop.

The proxy is OpenAI-compatible, with two deviations found by probing it (see
`scripts/probe_tool_calling.py`, which is the reproducer for both):

  * **`"stream": false` must be sent explicitly.** Omit it and the proxy replies with
    `text/event-stream` *and* drops the `usage` block entirely. Since per-turn token
    counts are the reason the session log exists, every request forces it off.
  * The TAMU gateway exposes the OpenAI-compatible API under `/v1` on the gateway host.
    The correct path is `{base_url}/v1/chat/completions`.

Native tool calling works fully on `protected.Claude Opus 4.8`: `tool_calls` come back
with parseable JSON arguments, `role: "tool"` results are accepted, and `usage` is
present on every turn.

**It does not work on every route the gateway exposes.** On
`us.anthropic.claude-opus-5` the `tools` array is accepted by the HTTP layer and then
ignored: the model answers as though it had no tools at all ("I don't have any way to
see your files..."), and because the reply carries no `tool_calls`, a loop that treats
"no tool_calls" as "finished" returns that non-answer to the user and never calls MCP.
So the fallback the plan held in reserve is needed after all, and is implemented here as
`tool_mode="prompted_json"`: the tool list moves into the system prompt and the model
replies with one JSON object per turn. See `scripts/probe_tool_calling.py`, which probes
the two capabilities separately.

Which mode to use is configuration (`SPICE_MCP_TOOL_MODE`), never inferred from the
model id - what the gateway does with `tools` is a property of the route.

This module never imports `mcp`. It takes a `tool_executor` callable, which keeps the
LLM side testable with a fake and the MCP side replaceable. Both modes converge on that
one callable, so the approval gate in `Api._tool_executor` covers both.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import requests

from .config import (
    TOOL_MODE_NATIVE,
    TOOL_MODE_PROMPTED_JSON,
    TOOL_MODES,
    Config,
)
from .session import Session, Turn

log = logging.getLogger(__name__)

# Guardrail on a runaway loop. Diagnosing a circuit takes a handful of calls
# (read_netlist -> check_netlist_static -> maybe run_simulation); a model that wants 12
# rounds is stuck, and each round costs tokens.
MAX_TOOL_ROUNDS = 12

# In prompted-JSON mode a reply that is not protocol JSON costs a round to correct. One
# correction is worth paying for (a stray code fence, a sentence of preamble); a model
# that cannot produce the protocol twice in a row will not produce it on the third try
# either, so the turn ends with its text rather than burning the whole round budget.
MAX_PROTOCOL_CORRECTIONS = 2

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


# --- the prompted-JSON protocol --------------------------------------------------------
#
# Deliberately terse. This text is prepended to the system prompt on every request in
# prompted mode, so every word is paid for on every round, and the project exists to
# compare token cost.
PROMPTED_PROTOCOL_HEADER = """\
TOOL PROTOCOL

This connection cannot deliver function calls, so tools are driven by JSON instead.
Every message you send must be exactly one JSON object and nothing else: no prose
around it, no code fence, no tag or XML markup.

To call a tool:
{"type":"tool_call","name":"<tool name>","arguments":{<arguments>}}

To answer the user, once you have what you need:
{"type":"answer","text":"<the whole answer, plain text>"}

Rules:
- Only the tools listed below exist. Use their exact names and argument schemas.
- Send one object, then stop. Do not continue the conversation past it.
- I run the tool and send back a message beginning "TOOL RESULT". Those messages come
  from me. Writing one yourself invents a result: it tells the user a value that was
  never read from their circuit. Wait for mine instead.
- Never state a value you have not seen in a TOOL RESULT. If something can be looked up
  with a tool, call the tool instead of guessing.
- Your reasoning belongs in the "text" of the final answer, not around the JSON.

TOOLS"""

# Sent as a stop sequence in prompted mode. Belt and braces with the rule above, and
# needed: on the first live probe of this mode, us.anthropic.claude-opus-5 emitted a
# correct tool_call and then role-played the rest of the exchange itself - a fabricated
# "TOOL RESULT" with invented component values, followed by an answer quoting them. The
# parser discards anything after the first object, so nothing fabricated was ever used,
# but a stop sequence means we do not pay for those tokens either.
PROMPTED_STOP = ["TOOL RESULT"]

# Omit `max_tokens` and the gateway caps the completion at 1024 - verified 2026-09-01 by
# asking for a long reply with and without it (1024 vs 4096 completion tokens, both with
# finish_reason "length"). 1024 cuts a full diagnosis off mid-sentence, and in prompted
# mode it lands mid-JSON, so the reply is unparseable and a correction round has to be
# paid for on top of the output already wasted. An explicit ceiling is the cheaper end of
# that trade: seen live, the truncated attempt burned 1024 output tokens and a 6.6k-token
# retry to say what fitted in 688.
DEFAULT_MAX_REPLY_TOKENS = 2048

PROMPTED_CORRECTION = (
    'PROTOCOL ERROR\n{reason}\nNothing was run. Reply with exactly one JSON object and '
    'nothing else: {{"type":"tool_call","name":"<tool>","arguments":{{...}}}} or '
    '{{"type":"answer","text":"..."}}'
)

# A truncated reply is unparseable for a reason the generic correction misdiagnoses: the
# JSON was well-formed until the output cap cut it off mid-string. Observed live - a long
# answer stopped at exactly 1024 completion tokens, and telling the model its JSON was
# invalid would have it hunt for a syntax error it never made. Verified 2026-09-01.
PROMPTED_TRUNCATED = (
    "PROTOCOL ERROR\nYour reply hit the output length limit part-way through the JSON, "
    "so it could not be parsed and nothing was run. Send the same object again but "
    "shorter - keep the answer text tight and put nothing outside the JSON."
)


class ToolExecutor(Protocol):
    """Runs a tool by name and returns its result as text for the model."""

    def __call__(self, name: str, arguments: dict[str, Any]) -> str: ...


class LLMError(RuntimeError):
    """A call to the chat API failed in a way worth showing the user."""


class ProtocolError(ValueError):
    """A prompted-mode reply was not a valid protocol message.

    `requested_name` is set when the reply did name a tool - an unknown name, say - so
    the caller can record the rejected attempt without executing anything.
    """

    def __init__(self, message: str, *, requested_name: str | None = None) -> None:
        super().__init__(message)
        self.requested_name = requested_name


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


def _tool_spec(tool: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Pull (name, description, schema) out of one OpenAI-shaped tool definition."""
    function = tool.get("function") if isinstance(tool, dict) else None
    source = function if isinstance(function, dict) else (tool if isinstance(tool, dict) else {})
    schema = source.get("parameters") or {"type": "object", "properties": {}}
    return (
        str(source.get("name") or ""),
        str(source.get("description") or ""),
        schema,
    )


def tool_names(tools: Iterable[dict[str, Any]] | None) -> list[str]:
    """The tool names in an OpenAI `tools` array, in order."""
    return [name for name, _, _ in (_tool_spec(t) for t in tools or []) if name]


def prompted_protocol_prompt(tools: Iterable[dict[str, Any]] | None) -> str:
    """Render the protocol and the tool catalogue for the system prompt.

    Takes the same OpenAI-shaped `tools` array that native mode sends over the wire, so
    there is one source of truth for what the model is told a tool takes - and so the
    two modes stay comparable on tokens: the descriptions and schemas are identical, only
    the transport differs.
    """
    lines = [PROMPTED_PROTOCOL_HEADER]
    for tool in tools or []:
        name, description, schema = _tool_spec(tool)
        if not name:
            continue
        lines.append(f"\n{name}")
        if description:
            lines.append(description.strip())
        # Compact separators: the schema is machine-readable input for the model, and
        # pretty-printing it would cost tokens for whitespace on every round.
        lines.append(
            "arguments: " + json.dumps(schema, separators=(",", ":"), sort_keys=False)
        )
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    """Remove one wrapping ``` fence, if that is all that is wrong with the reply.

    Narrow on purpose. A fence is the one formatting habit worth tolerating, because
    models add it reflexively to anything that looks like JSON. Anything beyond it -
    prose, several objects, tag markup - is left to fail parsing and earn a correction,
    because guessing at which brace in a paragraph was meant to be a tool call is how
    you end up executing something the model never asked for.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline == -1:
        return stripped
    # Only the language tag may sit on the opening line ("```json").
    if body[:newline].strip().isalpha() or not body[:newline].strip():
        body = body[newline + 1 :]
    else:
        return stripped
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def parse_prompted_reply(text: str, *, known_tools: Iterable[str]) -> dict[str, Any]:
    """Validate one prompted-mode reply and return a normalized protocol message.

    Returns either `{"type": "tool_call", "name": str, "arguments": dict}` or
    `{"type": "answer", "text": str}`. Raises `ProtocolError` for anything else - and
    raising is the point: nothing is executed unless the request parsed *and* named a
    tool the app actually supplied.
    """
    candidate = _strip_code_fence(text or "")
    if not candidate:
        raise ProtocolError("Your reply was empty.")

    # Decoded from position 0, so the reply must *begin* with the JSON object - no
    # searching the text for something brace-shaped. Anything after the first object is
    # discarded, which is the safe direction: a model that role-plays the rest of the
    # exchange (a fabricated TOOL RESULT and an answer quoting it - observed live) gets
    # its real first request executed and its invented remainder thrown away.
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Your reply was not valid JSON ({exc.msg}).") from exc

    trailing = candidate[end:].strip()
    if trailing:
        log.warning(
            "discarded %d characters after the protocol object: %.200s",
            len(trailing),
            trailing,
        )

    if not isinstance(payload, dict):
        raise ProtocolError("The JSON must be an object, not a list or a bare value.")

    kind = payload.get("type")
    if kind == "answer":
        answer = payload.get("text")
        if not isinstance(answer, str):
            raise ProtocolError('An "answer" needs a "text" string.')
        return {"type": "answer", "text": answer}

    if kind != "tool_call":
        raise ProtocolError('"type" must be "tool_call" or "answer".')

    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError('A "tool_call" needs a "name" string.')

    arguments = payload.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ProtocolError(
            '"arguments" must be an object of named arguments.', requested_name=name
        )

    allowed = list(known_tools)
    if name not in allowed:
        raise ProtocolError(
            f"There is no tool called {name!r}. The tools you have are: "
            f"{', '.join(allowed) or '(none)'}.",
            requested_name=name,
        )

    return {"type": "tool_call", "name": name, "arguments": arguments}


def prompted_tool_result_message(name: str, result: str) -> dict[str, Any]:
    """Frame a real tool result so the model cannot mistake its own text for one.

    Sent as a user-role message because prompted mode has no `tool_call_id` to attach a
    `role: "tool"` message to. The "TOOL RESULT" label is what the system prompt tells
    the model is the only genuine tool output.
    """
    return {"role": "user", "content": f"TOOL RESULT\nname: {name}\nresult:\n{result}"}


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
    # Prompted mode only: how many replies failed to parse as protocol messages. Worth
    # surfacing rather than swallowing - a mode that needs correcting every round is
    # costing tokens for nothing and the user should be able to see it.
    protocol_errors: int = 0


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
        max_tokens: int | None = DEFAULT_MAX_REPLY_TOKENS,
        stop: list[str] | None = None,
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
        if stop:
            # Honoured by the gateway: verified on us.anthropic.claude-opus-5, which
            # returns finish_reason "stop" with the text cut before the sequence.
            payload["stop"] = stop

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


def _execute_tool(
    tool_executor: ToolExecutor | None,
    name: str,
    arguments: dict[str, Any],
    on_progress: Callable[[str], None] | None,
) -> ToolCallRecord:
    """Run one tool through the caller's executor and record what came back.

    The single place either mode reaches MCP. Both loops funnel through here, so the
    approval gate the executor implements cannot be sidestepped by picking a mode.
    """
    if tool_executor is None:
        return ToolCallRecord(name, arguments, "No tools are available in this session.", True)

    if on_progress:
        on_progress(f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})")
    try:
        return ToolCallRecord(name, arguments, tool_executor(name, arguments), False)
    except Exception as exc:
        # Hand the failure to the model rather than aborting: a bad path or an
        # unsupported file is something it can recover from by calling the tool
        # differently.
        log.warning("tool %s failed: %s", name, exc)
        return ToolCallRecord(name, arguments, f"Tool {name} failed: {exc}", True)


def _usage_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _log_round(
    session: Session,
    text: str,
    usage: dict[str, Any],
    round_records: list[ToolCallRecord],
) -> None:
    """Write one model round to the session log: the assistant turn, then its results.

    Called for every API response in both modes, including ones that failed to parse -
    the request was billed either way, and a log that quietly omits the expensive rounds
    would misreport the cost comparison this project exists to make.
    """
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


def _exhausted(
    records: list[ToolCallRecord], total_in: int, total_out: int, protocol_errors: int = 0
) -> AgentResult:
    """Out of rounds. Return the partial work rather than raising - it was paid for."""
    return AgentResult(
        text=(
            f"Stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer. "
            "The last tool results are above; try asking a narrower question."
        ),
        tool_calls=records,
        rounds=MAX_TOOL_ROUNDS,
        input_tokens=total_in,
        output_tokens=total_out,
        protocol_errors=protocol_errors,
    )


def run_agent_turn(
    client: TamuClient,
    session: Session,
    user_text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: ToolExecutor | None = None,
    history: list[dict[str, Any]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    tool_mode: str = TOOL_MODE_NATIVE,
) -> AgentResult:
    """Run one user turn to completion, executing tool calls as the model asks.

    `history` is the running OpenAI-format message list and is mutated in place, so the
    caller keeps conversational context across turns. The session log is updated as
    each turn completes rather than at the end, so an interrupted session still leaves
    complete token counts behind.

    `tool_mode` selects how tools are offered - `native` sends the OpenAI `tools` array,
    `prompted_json` puts the catalogue in the system prompt instead. See the module
    docstring for why both exist.
    """
    if tool_mode not in TOOL_MODES:
        raise LLMError(
            f"Unknown tool mode {tool_mode!r}. Use one of: {', '.join(TOOL_MODES)}."
        )

    if history is None:
        history = []
    if not history or history[0].get("role") != "system":
        system_prompt = SYSTEM_PROMPT
        if tool_mode == TOOL_MODE_PROMPTED_JSON:
            system_prompt += "\n\n" + prompted_protocol_prompt(tools)
        history.insert(0, {"role": "system", "content": system_prompt})

    history.append({"role": "user", "content": user_text})
    session.add_turn(Turn(role="user", text=user_text))

    if tool_mode == TOOL_MODE_PROMPTED_JSON:
        return _run_prompted_turn(
            client, session, tools, tool_executor, history, on_progress
        )
    return _run_native_turn(client, session, tools, tool_executor, history, on_progress)


def _run_native_turn(
    client: TamuClient,
    session: Session,
    tools: list[dict[str, Any]] | None,
    tool_executor: ToolExecutor | None,
    history: list[dict[str, Any]],
    on_progress: Callable[[str], None] | None,
) -> AgentResult:
    """The OpenAI path: `tools` on the request, `tool_calls` on the reply."""
    records: list[ToolCallRecord] = []
    total_in = 0
    total_out = 0

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        body = client.complete(history, tools=tools)
        message = body["choices"][0]["message"]
        usage = body.get("usage") or {}
        round_in, round_out = _usage_tokens(usage)
        total_in += round_in
        total_out += round_out

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
                record = ToolCallRecord(
                    name,
                    {},
                    f"Your arguments were not valid JSON ({exc}). Raw: {raw_args}",
                    True,
                )
            else:
                record = _execute_tool(tool_executor, name, arguments, on_progress)

            round_records.append(record)
            records.append(record)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": record.result,
                }
            )

        _log_round(session, text, usage, round_records)

        if not tool_calls:
            return AgentResult(
                text=text,
                tool_calls=records,
                rounds=round_index,
                input_tokens=total_in,
                output_tokens=total_out,
            )

    return _exhausted(records, total_in, total_out)


def _run_prompted_turn(
    client: TamuClient,
    session: Session,
    tools: list[dict[str, Any]] | None,
    tool_executor: ToolExecutor | None,
    history: list[dict[str, Any]],
    on_progress: Callable[[str], None] | None,
) -> AgentResult:
    """The prompted path, for routes that accept `tools` and then ignore it.

    `tools` is deliberately *not* sent on the request here: on such a route it is dead
    weight in the prompt budget, and sending it would make a failure ambiguous between
    "the model ignored it" and "the model chose not to use it".
    """
    known = tool_names(tools)
    records: list[ToolCallRecord] = []
    total_in = 0
    total_out = 0
    protocol_errors = 0
    consecutive_failures = 0

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        body = client.complete(history, stop=PROMPTED_STOP)
        choice = body["choices"][0]
        message = choice["message"]
        truncated = choice.get("finish_reason") == "length"
        usage = body.get("usage") or {}
        round_in, round_out = _usage_tokens(usage)
        total_in += round_in
        total_out += round_out

        text = _message_text(message)
        # Normalized rather than verbatim: there is no provider-side tool_call structure
        # to preserve in this mode, and echoing back a stray `tool_calls` field the model
        # never meant would confuse the next round.
        history.append({"role": "assistant", "content": text})

        try:
            parsed = parse_prompted_reply(text, known_tools=known)
        except ProtocolError as exc:
            protocol_errors += 1
            consecutive_failures += 1
            log.warning(
                "prompted reply failed protocol validation (%s%s): %.300s",
                exc,
                "; cut off by the output length limit" if truncated else "",
                text,
            )
            # A named-but-unknown tool is recorded so the rejection is visible in the UI
            # and the session log. Recorded, not executed.
            round_records: list[ToolCallRecord] = []
            if exc.requested_name:
                round_records.append(
                    ToolCallRecord(
                        exc.requested_name, {}, f"REJECTED: {exc}", True
                    )
                )
                records.extend(round_records)
            _log_round(session, text, usage, round_records)

            if consecutive_failures >= MAX_PROTOCOL_CORRECTIONS:
                log.error(
                    "giving up on the prompted protocol after %d consecutive failures",
                    consecutive_failures,
                )
                return AgentResult(
                    text=text
                    or (
                        "The model stopped following the tool protocol and returned "
                        "nothing usable. Try asking again."
                    ),
                    tool_calls=records,
                    rounds=round_index,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    protocol_errors=protocol_errors,
                )

            history.append(
                {
                    "role": "user",
                    "content": PROMPTED_TRUNCATED
                    if truncated
                    else PROMPTED_CORRECTION.format(reason=exc),
                }
            )
            continue

        consecutive_failures = 0

        if parsed["type"] == "answer":
            _log_round(session, text, usage, [])
            return AgentResult(
                text=parsed["text"],
                tool_calls=records,
                rounds=round_index,
                input_tokens=total_in,
                output_tokens=total_out,
                protocol_errors=protocol_errors,
            )

        record = _execute_tool(
            tool_executor, parsed["name"], parsed["arguments"], on_progress
        )
        records.append(record)
        _log_round(session, text, usage, [record])
        history.append(prompted_tool_result_message(record.name, record.result))

    return _exhausted(records, total_in, total_out, protocol_errors)
