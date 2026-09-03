"""Gateway client, MCP->Anthropic tool mapping, and the agent loop.

The transport is the **TAMU AI Gateway's** Anthropic-shaped endpoint, reached with
`anthropic.Anthropic(base_url=..., auth_token=...)`, which resolves to
`POST {base_url}/v1/messages` and authenticates with `Authorization: Bearer`. No AWS
credentials, no region, no boto3: the gateway holds the upstream provider relationship and
this client holds one token.

Because the endpoint speaks the native Messages API, tool calling needs no translation
layer: an MCP tool definition is already `{name, description, input_schema}`, which is
exactly what Anthropic's `tools` parameter wants.

**The endpoint matters more than the model.** The same gateway also exposes
`/v1/chat/completions`, and on that path the very same model accepts an OpenAI-shaped
`tools` array and then answers as though it had none - narrating the call in visible text
while the turn "succeeds". `/v1/messages` returns real `tool_use` blocks; that difference
is the whole reason this module targets the path it does.

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

There are two tool-calling modes, selected by `SPICE_MCP_TOOL_MODE`:

  * **`native`** (the default) sends the `tools` parameter and reads `tool_use` blocks
    back. Verified live against the gateway's `/v1/messages` by
    `scripts/probe_tool_calling.py`.
  * **`prompted_json`** puts the tool catalogue in the system prompt and asks for one
    JSON object per reply. It exists for routes that accept `tools` and then ignore it -
    not hypothetical, and not a retired concern: it is what this same gateway's
    `/v1/chat/completions` path does with this same model. The symptom is the model
    *narrating* a tool call in its visible text while the turn "succeeds". A session log
    with no `tool_calls` key on any turn is that failure.

The mode is explicit configuration rather than a guess from the model id, because the
behaviour belongs to a route (a model reached through an endpoint), not to a name.

This module never imports `mcp`. It takes a `tool_executor` callable, which keeps the
LLM side testable with a fake and the MCP side replaceable. Both modes converge on that
one callable - through `_execute_tool` - so the approval gate in `Api._tool_executor`
covers both, and picking a mode buys the model no extra reach over the user's files.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import anthropic
from anthropic import Anthropic

from .config import (
    TOKEN_ENV_VAR,
    TOOL_MODE_NATIVE,
    TOOL_MODE_PROMPTED_JSON,
    TOOL_MODES,
    Config,
    ConfigError,
)
from .session import Session, Turn, usage_value

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

PROMPTED_CORRECTION = (
    'PROTOCOL ERROR\n{reason}\nNothing was run. Reply with exactly one JSON object and '
    'nothing else: {{"type":"tool_call","name":"<tool>","arguments":{{...}}}} or '
    '{{"type":"answer","text":"..."}}'
)

# A truncated reply is unparseable for a reason the generic correction misdiagnoses: the
# JSON was well-formed until the output cap cut it off mid-string, and telling the model
# its JSON was invalid would have it hunt for a syntax error it never made. Observed live
# live on the gateway's OpenAI path, which capped a reply at 1024 tokens when `max_tokens`
# was omitted. The Messages API cannot do that silently - `max_tokens` is mandatory and
# MAX_TOKENS is generous - but a long enough answer can still hit the ceiling, so the
# diagnosis stays.
PROMPTED_TRUNCATED = (
    "PROTOCOL ERROR\nYour reply hit the output length limit part-way through the JSON, "
    "so it could not be parsed and nothing was run. Send the same object again but "
    "shorter - keep the answer text tight and put nothing outside the JSON."
)


class ToolExecutor(Protocol):
    """Runs a tool by name and returns its result as text for the model."""

    def __call__(self, name: str, arguments: dict[str, Any]) -> str: ...


class LLMError(RuntimeError):
    """A call to the model failed in a way worth showing the user."""


class ProtocolError(ValueError):
    """A prompted-mode reply was not a valid protocol message.

    `requested_name` is set when the reply did name a tool - an unknown name, say - so
    the caller can record the rejected attempt without executing anything.
    """

    def __init__(self, message: str, *, requested_name: str | None = None) -> None:
        super().__init__(message)
        self.requested_name = requested_name


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


def _tool_spec(tool: Any) -> tuple[str, str, dict[str, Any]]:
    """Pull (name, description, schema) out of one Anthropic-shaped tool definition."""
    source = tool if isinstance(tool, Mapping) else {}
    schema = source.get("input_schema") or {"type": "object", "properties": {}}
    return (
        str(source.get("name") or ""),
        str(source.get("description") or ""),
        schema,
    )


def tool_names(tools: Iterable[dict[str, Any]] | None) -> list[str]:
    """The tool names in an Anthropic `tools` array, in order."""
    return [name for name, _, _ in (_tool_spec(t) for t in tools or []) if name]


def prompted_protocol_prompt(tools: Iterable[dict[str, Any]] | None) -> str:
    """Render the protocol and the tool catalogue for the system prompt.

    Takes the same `tools` array native mode sends over the wire, so there is one source
    of truth for what the model is told a tool takes - and so the two modes stay
    comparable on tokens: the descriptions and schemas are identical, only the transport
    differs.
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


def prompted_system_prompt(tools: Iterable[dict[str, Any]] | None) -> str:
    """The full system prompt for prompted mode.

    Built per request rather than stored, because the Messages API has no system *role*:
    the system prompt is a top-level parameter on every call, so there is nowhere in the
    message history to keep it. A `{"role": "system"}` message is a 400.
    """
    return SYSTEM_PROMPT + "\n\n" + prompted_protocol_prompt(tools)


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

    Sent as a user-role message because prompted mode has no `tool_use_id` to attach a
    `tool_result` block to. The "TOOL RESULT" label is what the system prompt tells the
    model is the only genuine tool output.
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


class GatewayClient:
    """Thin wrapper over the gateway's Messages endpoint, and the one place errors
    become LLMError.

    The bearer token is read from the environment *here* rather than carried on `Config`,
    which is what keeps `Config` free of secrets - see config.py. `auth_token=` makes the
    SDK send `Authorization: Bearer <token>` and omit `x-api-key`, matching what the
    gateway expects. It is held only on the SDK client; this object exposes no attribute
    carrying it.
    """

    def __init__(self, config: Config, *, timeout: float = 180.0) -> None:
        self._config = config
        token = (os.environ.get(TOKEN_ENV_VAR) or "").strip()
        if not token:
            # Reachable when a caller built a Config with require_credentials=False and
            # then tried to talk to the model anyway. Better here than as a 401.
            raise ConfigError(
                f"{TOKEN_ENV_VAR} is not set, so {config.messages_url} cannot be "
                "reached."
            )
        self._client = Anthropic(
            base_url=config.base_url, auth_token=token, timeout=timeout
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
        stop_sequences: list[str] | None = None,
    ) -> Any:
        """One turn. Returns the SDK's Message object."""
        try:
            return self._client.messages.create(
                model=self._config.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools or anthropic.NOT_GIVEN,
                stop_sequences=stop_sequences or anthropic.NOT_GIVEN,
            )
        except anthropic.NotFoundError as exc:
            # A 404 from the gateway is about its routing table, not about this machine.
            # Worth saying so: the SDK's own message reads like a broken URL.
            raise LLMError(
                f"The gateway did not route {self._config.model!r}.\n\n"
                "Either it does not offer that model id or this token is not entitled "
                "to it. Set SPICE_MCP_MODEL in .env to an id the gateway serves.\n\n"
                f"Endpoint: {self._config.messages_url}\n\n{exc}"
            ) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            # Split from the 404 on purpose - the two used to share a branch because on
            # Bedrock both meant "no model access", but here they mean different things
            # and send the reader to different places.
            raise LLMError(
                f"The gateway rejected the credential "
                f"({self._config.credentials_source}).\n\n"
                f"Check that {TOKEN_ENV_VAR} in .env is current and entitled to "
                f"{self._config.model}.\n\n{exc}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(
                f"Could not reach {self._config.messages_url}.\n\n"
                "The request never got an HTTP status back, so this is a network or VPN "
                f"problem rather than a configuration one.\n\n{exc}"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError(f"The gateway call failed: {exc}") from exc


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
        return ToolCallRecord(
            name, arguments, "No tools are available in this session.", True
        )

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


def _usage_tokens(usage: Any) -> tuple[int, int]:
    """Round totals from a usage block, tolerating dicts and SDK objects."""
    return (
        int(usage_value(usage, "input_tokens") or 0),
        int(usage_value(usage, "output_tokens") or 0),
    )


def _log_round(
    session: Session,
    text: str,
    usage: Any,
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
    records: list[ToolCallRecord],
    total_in: int,
    total_out: int,
    protocol_errors: int = 0,
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
    client: GatewayClient,
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

    `history` is the running Messages-API message list and is mutated in place, so the
    caller keeps conversational context across turns. The system prompt is *not* in
    there - it is a top-level parameter on every request. The session log is updated as
    each turn completes rather than at the end, so an interrupted session still leaves
    complete token counts behind.

    `tool_mode` selects how tools are offered - `native` sends the `tools` parameter,
    `prompted_json` puts the catalogue in the system prompt instead. See the module
    docstring for why both exist.
    """
    if tool_mode not in TOOL_MODES:
        raise LLMError(
            f"Unknown tool mode {tool_mode!r}. Use one of: {', '.join(TOOL_MODES)}."
        )

    if history is None:
        history = []

    append_user_note(history, user_text)
    session.add_turn(Turn(role="user", text=user_text))

    if tool_mode == TOOL_MODE_PROMPTED_JSON:
        return _run_prompted_turn(
            client, session, tools, tool_executor, history, on_progress
        )
    return _run_native_turn(client, session, tools, tool_executor, history, on_progress)


def _run_native_turn(
    client: GatewayClient,
    session: Session,
    tools: list[dict[str, Any]] | None,
    tool_executor: ToolExecutor | None,
    history: list[dict[str, Any]],
    on_progress: Callable[[str], None] | None,
) -> AgentResult:
    """The native path: `tools` on the request, `tool_use` blocks on the reply."""
    records: list[ToolCallRecord] = []
    total_in = 0
    total_out = 0

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.complete(history, tools=tools)
        usage = _block_field(response, "usage")
        round_in, round_out = _usage_tokens(usage)
        total_in += round_in
        total_out += round_out

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
                record = ToolCallRecord(
                    name,
                    arguments,
                    f"Your input for {name} was not an object. Send the arguments as "
                    "a JSON object and try again.",
                    True,
                )
            else:
                record = _execute_tool(tool_executor, name, arguments, on_progress)

            round_records.append(record)
            records.append(record)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": _block_field(block, "id"),
                    "content": record.result,
                    "is_error": record.is_error,
                }
            )

        if results:
            # One message carrying every result from this round. Splitting them is rejected.
            history.append({"role": "user", "content": results})

        _log_round(session, text, usage, round_records)

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

    return _exhausted(records, total_in, total_out)


def _run_prompted_turn(
    client: GatewayClient,
    session: Session,
    tools: list[dict[str, Any]] | None,
    tool_executor: ToolExecutor | None,
    history: list[dict[str, Any]],
    on_progress: Callable[[str], None] | None,
) -> AgentResult:
    """The prompted path, for routes that accept `tools` and then ignore it.

    `tools` is deliberately *not* sent on the request here: on such a route it is dead
    weight in the prompt budget, and sending it would make a failure ambiguous between
    "the model ignored it" and "the model chose not to use it". The catalogue goes into
    the system prompt instead, which the Messages API takes as a top-level parameter on
    every call - there is no system *role* to put it in the history once.
    """
    known = tool_names(tools)
    system = prompted_system_prompt(tools)
    records: list[ToolCallRecord] = []
    total_in = 0
    total_out = 0
    protocol_errors = 0
    consecutive_failures = 0

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.complete(
            history, system=system, stop_sequences=PROMPTED_STOP
        )
        usage = _block_field(response, "usage")
        truncated = _block_field(response, "stop_reason") == "max_tokens"
        round_in, round_out = _usage_tokens(usage)
        total_in += round_in
        total_out += round_out

        text = _response_text(response)
        # The visible text only, not `content` verbatim: thinking blocks have to be
        # returned unchanged when they accompany a `tool_use`, and in this mode there is
        # never one - the whole point is that the provider produced no tool structure to
        # preserve. An empty reply is skipped because the API rejects empty content, and
        # inventing text for the assistant would put words in the transcript it never
        # said; `append_user_note` then keeps the correction from forming a second
        # consecutive user turn.
        if text.strip():
            history.append({"role": "assistant", "content": text})
        else:
            log.warning("prompted reply had no visible text; not adding it to history")

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

            append_user_note(
                history,
                PROMPTED_TRUNCATED
                if truncated
                else PROMPTED_CORRECTION.format(reason=exc),
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
