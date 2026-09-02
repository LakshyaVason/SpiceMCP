"""Tests for the agent loop, schema translation, and the approval gate.

All offline: `FakeClient` returns canned OpenAI-shaped responses. The loop's job is to
keep asking until the model stops requesting tools, feed results back in the shape the
provider expects, and record usage for every round - none of which needs a network to
verify, and all of which would be expensive and flaky to test against the live proxy.

The approval-gate tests are the important ones here. "Nothing touches disk before user
approval" is a promise to the user about their schematic, and a model can ask for
apply=True whenever it likes, so the refusal has to be tested rather than assumed.

The prompted-JSON section exists because this suite passed while the app was broken. The
fakes only ever returned ideal `tool_calls`, so nothing here noticed that
`us.anthropic.claude-opus-5` accepts the `tools` array and then answers as though it had
no tools at all - at which point the loop read "no tool_calls" as "finished" and handed
the user a non-answer without ever reaching MCP. Those tests drive the fake the way that
route really behaves.
"""

from __future__ import annotations

import json

import pytest

from spice_mcp_app.api import Api
from spice_mcp_app.config import (
    TOOL_MODE_NATIVE,
    TOOL_MODE_PROMPTED_JSON,
    Config,
)
from spice_mcp_app.llm import (
    DEFAULT_MAX_REPLY_TOKENS,
    MAX_PROTOCOL_CORRECTIONS,
    MAX_TOOL_ROUNDS,
    LLMError,
    TamuClient,
    ProtocolError,
    mcp_tools_to_openai,
    parse_prompted_reply,
    prompted_protocol_prompt,
    run_agent_turn,
)
from spice_mcp_app.session import Session


class FakeTool:
    """Stands in for an mcp.types.Tool, which is snake_case on the Python side."""

    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


def message(content=None, tool_calls=None):
    payload: dict = {"role": "assistant", "content": content}
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
    return payload


def response(msg, prompt_tokens=100, completion_tokens=20):
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.model = "fake-model"

    def complete(self, messages, *, tools=None, max_tokens=None, stop=None):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "stop": stop}
        )
        if not self._responses:
            raise AssertionError("the loop asked for more responses than were provided")
        return self._responses.pop(0)


@pytest.fixture
def session(tmp_path):
    return Session(model="fake-model", sessions_dir=tmp_path)


# --- prompted-JSON helpers ------------------------------------------------------------

PROMPTED_TOOLS = mcp_tools_to_openai(
    [
        FakeTool(
            "read_netlist",
            "Parse a circuit into structured JSON.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        FakeTool(
            "run_simulation",
            "Run LTspice in batch mode.",
            {"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        FakeTool(
            "patch_component_value",
            "Change one component value.",
            {"type": "object", "properties": {"asc_path": {"type": "string"}}},
        ),
    ]
)


def prompted(payload, **usage):
    """A response whose content is one protocol JSON object, as the real model sends it."""
    return response(message(json.dumps(payload)), **usage)


def raw(content, **usage):
    """A response whose content is whatever the model actually emitted."""
    return response(message(content), **usage)


def truncated(content, **usage):
    """A reply the provider cut off at its output cap: `finish_reason: "length"`."""
    body = raw(content, **usage)
    body["choices"][0]["finish_reason"] = "length"
    return body


def run_prompted(client, session, text, **kwargs):
    kwargs.setdefault("tools", PROMPTED_TOOLS)
    return run_agent_turn(
        client, session, text, tool_mode=TOOL_MODE_PROMPTED_JSON, **kwargs
    )


# --- the request payload --------------------------------------------------------------


class FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


def test_every_request_pins_stream_false_and_an_output_ceiling(tmp_path):
    """Two proxy defaults that quietly break things if left alone.

    Streaming drops the `usage` block the session log is built on, and an omitted
    `max_tokens` caps the reply at 1024 - short enough to cut a diagnosis in half.
    """
    config = Config(
        api_key="not-a-real-key",
        model="fake-model",
        base_url="https://example.invalid/openai",
        sessions_dir=tmp_path,
    )
    client = TamuClient(config)
    sent: list[dict] = []
    client._http.post = lambda url, **kw: sent.append(kw["json"]) or FakeResponse()

    client.complete([{"role": "user", "content": "hi"}])

    assert sent[0]["stream"] is False
    assert sent[0]["max_tokens"] == DEFAULT_MAX_REPLY_TOKENS
    # Nothing is sent that was not asked for.
    assert "tools" not in sent[0]
    assert "stop" not in sent[0]


# --- schema translation ---------------------------------------------------------------


def test_mcp_tools_translate_to_openai_functions():
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    translated = mcp_tools_to_openai([FakeTool("read_netlist", "Reads it.", schema)])

    assert translated == [
        {
            "type": "function",
            "function": {
                "name": "read_netlist",
                "description": "Reads it.",
                "parameters": schema,
            },
        }
    ]


def test_translation_accepts_camel_case_dicts():
    """Wire-format dicts use inputSchema; both spellings must work."""
    translated = mcp_tools_to_openai(
        [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]
    )
    assert translated[0]["function"]["parameters"] == {"type": "object"}


def test_translation_substitutes_an_empty_schema():
    """A null `parameters` is rejected by some providers."""
    translated = mcp_tools_to_openai([FakeTool("no_args")])
    assert translated[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_translation_skips_nameless_tools():
    assert mcp_tools_to_openai([{"description": "no name"}]) == []


# --- the loop -------------------------------------------------------------------------


def test_a_plain_answer_ends_the_loop(session):
    client = FakeClient([response(message("It is a low-pass filter."))])
    result = run_agent_turn(client, session, "what is it?")

    assert result.text == "It is a low-pass filter."
    assert result.rounds == 1
    assert result.tool_calls == []
    assert len(client.calls) == 1


def test_the_system_prompt_is_prepended_once(session):
    client = FakeClient([response(message("a")), response(message("b"))])
    history: list[dict] = []

    run_agent_turn(client, session, "first", history=history)
    run_agent_turn(client, session, "second", history=history)

    assert history[0]["role"] == "system"
    assert sum(1 for m in history if m["role"] == "system") == 1


def test_tool_results_are_fed_back_and_the_loop_continues(session):
    client = FakeClient(
        [
            response(message("Looking.", [("read_netlist", {"path": "x.asc"})])),
            response(message("R1 is 1.6k.")),
        ]
    )
    executed: list[tuple[str, dict]] = []

    def executor(name, arguments):
        executed.append((name, arguments))
        return '{"components": []}'

    result = run_agent_turn(
        client, session, "read it", tool_executor=executor, tools=[{"x": 1}]
    )

    assert executed == [("read_netlist", {"path": "x.asc"})]
    assert result.rounds == 2
    assert result.text == "R1 is 1.6k."

    # The second request must carry the assistant message and a matching tool result.
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": '{"components": []}',
    }


def test_usage_is_summed_across_rounds(session):
    client = FakeClient(
        [
            response(message("t", [("read_netlist", {})]), 100, 20),
            response(message("done"), 300, 50),
        ]
    )
    result = run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "{}", tools=[{}]
    )

    assert result.input_tokens == 400
    assert result.output_tokens == 70
    assert session.total_input_tokens == 400
    assert session.total_output_tokens == 70


def test_a_failing_tool_is_reported_to_the_model_not_raised(session):
    """A bad path is recoverable: the model can call the tool differently."""

    def executor(name, arguments):
        raise RuntimeError("No such file: nope.asc")

    client = FakeClient(
        [
            response(message("t", [("read_netlist", {"path": "nope.asc"})])),
            response(message("That file does not exist.")),
        ]
    )
    result = run_agent_turn(
        client, session, "read nope.asc", tool_executor=executor, tools=[{}]
    )

    assert result.rounds == 2
    assert result.tool_calls[0].is_error
    assert "No such file" in result.tool_calls[0].result
    assert "No such file" in client.calls[1]["messages"][-1]["content"]


def test_malformed_tool_arguments_are_handed_back(session):
    broken = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "read_netlist", "arguments": "{not json"},
            }
        ],
    }
    client = FakeClient([response(broken), response(message("Sorry, retrying."))])
    result = run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "{}", tools=[{}]
    )

    assert result.tool_calls[0].is_error
    assert "not valid JSON" in result.tool_calls[0].result


def test_the_loop_is_bounded(session):
    """A model that never stops asking for tools must not run forever."""
    client = FakeClient(
        [response(message("again", [("read_netlist", {})]))] * (MAX_TOOL_ROUNDS + 2)
    )
    result = run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "{}", tools=[{}]
    )

    assert result.rounds == MAX_TOOL_ROUNDS
    assert "Stopped after" in result.text
    # The partial work is still logged - its token cost was still incurred.
    assert session.total_input_tokens > 0


def test_tool_calls_without_an_executor_do_not_crash(session):
    client = FakeClient(
        [response(message("t", [("read_netlist", {})])), response(message("ok"))]
    )
    result = run_agent_turn(client, session, "go", tools=[{}])
    assert result.tool_calls[0].is_error
    assert "No tools are available" in result.tool_calls[0].result


def test_content_part_lists_are_flattened(session):
    """Some providers return content as a list of typed parts rather than a string."""
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}],
    }
    result = run_agent_turn(FakeClient([response(msg)]), session, "go")
    assert result.text == "Part one. Part two."


def test_session_records_the_tool_turns(session):
    client = FakeClient(
        [
            response(message("t", [("read_netlist", {"path": "x"})])),
            response(message("done")),
        ]
    )
    run_agent_turn(client, session, "go", tool_executor=lambda n, a: "RESULT", tools=[{}])

    roles = [t.role for t in session.turns]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.turns[1].tool_calls[0]["name"] == "read_netlist"
    assert session.turns[2].text == "RESULT"


# --- native mode, stated explicitly ----------------------------------------------------


def test_native_mode_sends_the_tools_array_and_consumes_tool_calls(session):
    """The pre-existing behaviour, now that it is one mode of two rather than the only one."""
    client = FakeClient(
        [
            response(message("Looking.", [("read_netlist", {"path": "x.asc"})])),
            response(message("It is a low-pass.")),
        ]
    )
    executed: list[tuple[str, dict]] = []

    result = run_agent_turn(
        client,
        session,
        "what is it?",
        tools=PROMPTED_TOOLS,
        tool_executor=lambda n, a: executed.append((n, a)) or '{"components": []}',
        tool_mode=TOOL_MODE_NATIVE,
    )

    assert executed == [("read_netlist", {"path": "x.asc"})]
    assert result.text == "It is a low-pass."
    # Native mode's whole premise: the tool catalogue travels in the API field.
    assert client.calls[0]["tools"] == PROMPTED_TOOLS
    assert "TOOL PROTOCOL" not in client.calls[0]["messages"][0]["content"]


def test_the_default_mode_is_native(session):
    client = FakeClient([response(message("hi"))])
    run_agent_turn(client, session, "go", tools=PROMPTED_TOOLS)
    assert client.calls[0]["tools"] == PROMPTED_TOOLS


def test_an_unknown_mode_is_rejected_rather_than_defaulted(session):
    with pytest.raises(LLMError, match="Unknown tool mode"):
        run_agent_turn(
            FakeClient([]), session, "go", tool_mode="prompted-json"  # hyphen, not underscore
        )


# --- prompted-JSON mode ----------------------------------------------------------------
#
# This is the mode that had to be added, and these are the tests that would have caught
# the production failure. The fake here never returns `tool_calls`, because the route this
# mode exists for never does.


def test_the_protocol_prompt_lists_the_real_tools_and_schemas():
    prompt = prompted_protocol_prompt(PROMPTED_TOOLS)

    assert "read_netlist" in prompt and "run_simulation" in prompt
    assert '"required":["path"]' in prompt, "the argument schema must reach the model"
    assert '{"type":"tool_call"' in prompt and '{"type":"answer"' in prompt
    # Compact on purpose: this text is re-sent on every round of every turn.
    assert '"properties": {' not in prompt, "the schema was pretty-printed"


def test_a_prompted_tool_request_reaches_the_executor_and_its_result_reaches_the_model(
    session,
):
    """The end-to-end shape of the fix: JSON request -> real executor -> real result back."""
    client = FakeClient(
        [
            prompted(
                {
                    "type": "tool_call",
                    "name": "read_netlist",
                    "arguments": {"path": r"C:\c\rc.asc"},
                }
            ),
            prompted({"type": "answer", "text": "C1 is 1n, so the cutoff is 10 kHz."}),
        ]
    )
    executed: list[tuple[str, dict]] = []

    def executor(name, arguments):
        executed.append((name, arguments))
        return '{"components": [{"ref": "C1", "value": "1n"}]}'

    result = run_prompted(client, session, "what is wrong?", tool_executor=executor)

    assert executed == [("read_netlist", {"path": r"C:\c\rc.asc"})]
    assert result.text == "C1 is 1n, so the cutoff is 10 kHz."
    assert result.rounds == 2
    assert result.protocol_errors == 0

    # The API field is not used in this mode - the catalogue is in the system prompt.
    assert client.calls[0]["tools"] is None
    assert "TOOL PROTOCOL" in client.calls[0]["messages"][0]["content"]
    # And generation is stopped at the label, so a model that starts role-playing the
    # tool result is cut off rather than billed for it.
    assert client.calls[0]["stop"] == ["TOOL RESULT"]

    # The second request must carry the model's own JSON and the real tool result,
    # labelled so the model cannot mistake its own text for tool output.
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-1]["role"] == "user"
    assert second[-1]["content"].startswith("TOOL RESULT\nname: read_netlist\nresult:")
    assert '"value": "1n"' in second[-1]["content"], (
        "the model was not given the actual result the executor returned"
    )


def test_prompted_mode_records_the_turns_and_the_tool_result(session):
    client = FakeClient(
        [
            prompted({"type": "tool_call", "name": "read_netlist", "arguments": {"path": "x"}}),
            prompted({"type": "answer", "text": "done"}),
        ]
    )
    run_prompted(client, session, "go", tool_executor=lambda n, a: "REAL RESULT")

    roles = [t.role for t in session.turns]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.turns[1].tool_calls[0]["name"] == "read_netlist"
    assert session.turns[2].text == "REAL RESULT"


def test_several_prompted_rounds_accumulate_every_api_response(session):
    """Four API calls, four usage blocks. The cost comparison depends on all of them."""
    client = FakeClient(
        [
            prompted(
                {"type": "tool_call", "name": "read_netlist", "arguments": {"path": "x"}},
                prompt_tokens=1000,
                completion_tokens=30,
            ),
            prompted(
                {"type": "tool_call", "name": "run_simulation", "arguments": {"path": "x"}},
                prompt_tokens=2000,
                completion_tokens=40,
            ),
            prompted(
                {
                    "type": "tool_call",
                    "name": "patch_component_value",
                    "arguments": {"asc_path": "x"},
                },
                prompt_tokens=3000,
                completion_tokens=50,
            ),
            prompted(
                {"type": "answer", "text": "C1 should be 100n."},
                prompt_tokens=4000,
                completion_tokens=60,
            ),
        ]
    )
    executed: list[str] = []

    result = run_prompted(
        client,
        session,
        "diagnose it",
        tool_executor=lambda n, a: executed.append(n) or "{}",
    )

    assert executed == ["read_netlist", "run_simulation", "patch_component_value"]
    assert result.rounds == 4
    assert result.input_tokens == 10_000
    assert result.output_tokens == 180
    assert session.total_input_tokens == 10_000
    assert session.total_output_tokens == 180


def test_a_fenced_json_reply_is_accepted(session):
    """A markdown fence is the one habit worth tolerating; it must not break the app."""
    body = json.dumps({"type": "tool_call", "name": "read_netlist", "arguments": {"path": "x"}})
    client = FakeClient(
        [
            raw(f"```json\n{body}\n```"),
            raw("```\n" + json.dumps({"type": "answer", "text": "fenced answer"}) + "\n```"),
        ]
    )
    executed: list[str] = []

    result = run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )

    assert executed == ["read_netlist"]
    assert result.text == "fenced answer"
    assert result.protocol_errors == 0


def test_malformed_json_executes_nothing_and_earns_one_correction(session):
    client = FakeClient(
        [
            raw("Sure! Here is what I think: the capacitor looks wrong."),
            prompted({"type": "answer", "text": "C1 is wrong."}),
        ]
    )
    executed: list[str] = []

    result = run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )

    assert executed == [], "prose was executed as if it were a tool call"
    assert result.protocol_errors == 1
    assert result.text == "C1 is wrong."
    correction = client.calls[1]["messages"][-1]
    assert correction["role"] == "user"
    assert correction["content"].startswith("PROTOCOL ERROR")
    assert "Nothing was run" in correction["content"]


def test_a_reply_cut_off_by_the_length_limit_is_diagnosed_as_such(session):
    """Seen live: a long answer stopped at exactly 1024 completion tokens, mid-string.

    The JSON was fine until the cap; telling the model it wrote invalid JSON sends it
    hunting for a syntax error it never made. It is told to be shorter instead.
    """
    client = FakeClient(
        [
            truncated('{"type":"answer","text":"Circuit as read from the netlist:\\n\\n  R1'),
            prompted({"type": "answer", "text": "R1 is 8.2k."}),
        ]
    )
    executed: list[str] = []

    result = run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )

    assert executed == []
    assert result.protocol_errors == 1
    assert result.text == "R1 is 8.2k."
    correction = client.calls[1]["messages"][-1]["content"]
    assert "output length limit" in correction
    assert "shorter" in correction
    # The generic "your JSON is invalid" advice would be a misdiagnosis here.
    assert "Nothing was run" not in correction
    # The truncated round was still billed, so it is still logged.
    assert session.total_input_tokens == 200


def test_persistent_protocol_failure_fails_safely_without_executing(session):
    """Bounded: a model that cannot produce the protocol twice is not asked a third time."""
    client = FakeClient([raw("still not JSON")] * (MAX_TOOL_ROUNDS + 2))
    executed: list[str] = []

    result = run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )

    assert executed == []
    assert result.protocol_errors == MAX_PROTOCOL_CORRECTIONS
    assert len(client.calls) == MAX_PROTOCOL_CORRECTIONS
    # The model's own words are handed back rather than an exception or an empty screen.
    assert result.text == "still not JSON"
    # Both billed rounds are still in the log.
    assert session.total_input_tokens == 200


def test_antml_invoke_markup_is_never_executed(session):
    """Seen once in a real reply. It is not a protocol, and must not be treated as one.

    A tolerant parser that went looking for a tool name in arbitrary markup would be
    executing something the model never asked for through this app's protocol - with the
    argument values taken from text nobody validated.
    """
    # Assembled from parts so the literal tags never appear in this file either - the
    # point is what the parser does with them, not that they exist anywhere on disk.
    invoke, param = "antml:invoke", "antml:parameter"
    markup = (
        f'<{invoke} name="read_netlist">'
        f'<{param} name="path">C:\\somewhere\\else.asc</{param}>'
        f"</{invoke}>"
    )
    client = FakeClient(
        [raw(markup), prompted({"type": "answer", "text": "sorry about that"})]
    )
    executed: list[tuple[str, dict]] = []

    result = run_prompted(
        client,
        session,
        "go",
        tool_executor=lambda n, a: executed.append((n, a)) or "{}",
    )

    assert executed == [], "tag markup was interpreted as a tool call"
    assert result.protocol_errors == 1
    assert result.text == "sorry about that"


def test_an_unknown_tool_name_is_never_executed(session):
    client = FakeClient(
        [
            prompted(
                {
                    "type": "tool_call",
                    "name": "delete_everything",
                    "arguments": {"path": "x"},
                }
            ),
            prompted({"type": "answer", "text": "understood"}),
        ]
    )
    executed: list[str] = []

    result = run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )

    assert executed == [], "an unknown tool name reached the executor"
    assert result.text == "understood"
    # Recorded as a rejection, so the attempt is visible in the UI and the session log.
    assert [(c.name, c.is_error) for c in result.tool_calls] == [
        ("delete_everything", True)
    ]
    assert "no tool called 'delete_everything'" in result.tool_calls[0].result
    # And the correction tells it what does exist, rather than just "no".
    assert "read_netlist" in client.calls[1]["messages"][-1]["content"]


def test_arguments_that_are_not_an_object_are_not_executed(session):
    client = FakeClient(
        [
            prompted(
                {"type": "tool_call", "name": "read_netlist", "arguments": "C:\\rc.asc"}
            ),
            prompted({"type": "answer", "text": "fixed my JSON"}),
        ]
    )
    executed: list[str] = []

    run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )
    assert executed == []


def test_prompted_mode_without_an_executor_does_not_crash(session):
    client = FakeClient(
        [
            prompted({"type": "tool_call", "name": "read_netlist", "arguments": {}}),
            prompted({"type": "answer", "text": "ok"}),
        ]
    )
    result = run_prompted(client, session, "go")
    assert result.tool_calls[0].is_error
    assert "No tools are available" in result.tool_calls[0].result


def test_the_prompted_loop_is_bounded(session):
    client = FakeClient(
        [prompted({"type": "tool_call", "name": "read_netlist", "arguments": {}})]
        * (MAX_TOOL_ROUNDS + 2)
    )
    result = run_prompted(client, session, "go", tool_executor=lambda n, a: "{}")

    assert result.rounds == MAX_TOOL_ROUNDS
    assert "Stopped after" in result.text
    assert session.total_input_tokens > 0


# --- the protocol parser in isolation --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json at all",
        "[1, 2, 3]",
        '{"type": "something_else"}',
        '{"type": "answer"}',
        '{"type": "answer", "text": {"nested": "object"}}',
        '{"type": "tool_call", "arguments": {}}',
        '{"type": "tool_call", "name": "", "arguments": {}}',
        '{"type": "tool_call", "name": "read_netlist", "arguments": []}',
        '{"type": "tool_call", "name": "nope", "arguments": {}}',
        # The reply must *begin* with the object. Hunting through prose for something
        # brace-shaped is how you execute a call the model never made.
        'Here you go: {"type": "tool_call", "name": "read_netlist", "arguments": {}}',
        '<invoke name="read_netlist">{"path": "x"}',
    ],
)
def test_the_parser_refuses_anything_that_is_not_the_protocol(text):
    with pytest.raises(ProtocolError):
        parse_prompted_reply(text, known_tools=["read_netlist"])


def test_a_fabricated_tool_result_after_the_request_is_discarded():
    """The live failure this mode had to survive, reduced to its shape.

    us.anthropic.claude-opus-5's first prompted reply was a correct tool_call followed by
    a "TOOL RESULT" it wrote itself, with invented component values, and an answer quoting
    them. Only the first object may be used: the real tool then runs and the real result
    is what comes back. Nothing the model invented can reach the user as fact.
    """
    reply = (
        '{"type":"tool_call","name":"read_netlist","arguments":{"path":"x"}}\n\n'
        'TOOL RESULT {"ref":"C1","value":"100 nF"}\n\n'
        '{"type":"answer","text":"C1 is 100 nF."}'
    )
    assert parse_prompted_reply(reply, known_tools=["read_netlist"]) == {
        "type": "tool_call",
        "name": "read_netlist",
        "arguments": {"path": "x"},
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"type": "tool_call", "name": "read_netlist", "arguments": {"path": "x"}}',
        '  {"type":"tool_call","name":"read_netlist","arguments":{"path":"x"}}\n',
        '```json\n{"type": "tool_call", "name": "read_netlist", "arguments": {"path": "x"}}\n```',
        '```\n{"type": "tool_call", "name": "read_netlist", "arguments": {"path": "x"}}\n```',
    ],
)
def test_the_parser_accepts_the_protocol_with_or_without_a_fence(text):
    assert parse_prompted_reply(text, known_tools=["read_netlist"]) == {
        "type": "tool_call",
        "name": "read_netlist",
        "arguments": {"path": "x"},
    }


def test_missing_arguments_are_treated_as_none_given():
    """Several tools have no required arguments; omitting the key is not an error."""
    assert parse_prompted_reply(
        '{"type": "tool_call", "name": "read_netlist"}', known_tools=["read_netlist"]
    ) == {"type": "tool_call", "name": "read_netlist", "arguments": {}}


def test_a_rejected_tool_name_is_reported_on_the_error():
    with pytest.raises(ProtocolError) as caught:
        parse_prompted_reply(
            '{"type": "tool_call", "name": "rm_rf", "arguments": {}}',
            known_tools=["read_netlist"],
        )
    assert caught.value.requested_name == "rm_rf"


# --- the approval gate ----------------------------------------------------------------


class FakeMCP:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, arguments, timeout=300.0):
        self.calls.append((name, arguments))
        return json.dumps({"applied": bool(arguments.get("apply")), "summary": "ok"})


def build_gated_api(tmp_path, tool_mode=TOOL_MODE_NATIVE):
    config = Config(
        api_key="not-a-real-key",
        model="fake-model",
        base_url="https://example.invalid/openai",
        sessions_dir=tmp_path,
        tool_mode=tool_mode,
    )
    api = Api(config=config)
    api._mcp = FakeMCP()
    api._session = Session(model="fake-model", sessions_dir=tmp_path)
    return api


@pytest.fixture
def gated_api(tmp_path):
    return build_gated_api(tmp_path)


def test_an_unapproved_apply_is_refused(gated_api, tmp_path):
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    result = gated_api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": True},
    )

    assert "REFUSED" in result
    assert gated_api._mcp.calls == [], "the write reached the server despite the gate"


def test_a_preview_is_always_allowed(gated_api, tmp_path):
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    gated_api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": False},
    )

    assert len(gated_api._mcp.calls) == 1
    assert gated_api._mcp.calls[0][1]["apply"] is False


def test_approval_is_single_use(gated_api, tmp_path):
    """One approval, one write; a later apply must be approved again."""
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    approved = gated_api.apply_patch(str(asc), "C1", "100n")
    assert approved["ok"], approved.get("error")
    assert gated_api._mcp.calls[-1][1]["apply"] is True

    # The model trying to repeat it afterwards is refused.
    again = gated_api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": True},
    )
    assert "REFUSED" in again


def test_approval_does_not_authorise_a_different_change(gated_api, tmp_path):
    """An approval is for one component and one value, not a licence to write."""
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")
    gated_api._approved.add((str(asc.resolve()), "C1", "100n"))

    for arguments in (
        {"asc_path": str(asc), "ref": "R1", "new_value": "100n", "apply": True},
        {"asc_path": str(asc), "ref": "C1", "new_value": "10n", "apply": True},
    ):
        assert "REFUSED" in gated_api._tool_executor("patch_component_value", arguments)


def test_applying_tells_the_model_what_happened(gated_api, tmp_path):
    """The conversation must stay truthful about the state of the file."""
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    gated_api.apply_patch(str(asc), "C1", "100n")

    assert any("user approved" in m["content"] for m in gated_api._history)
    assert any("user approved" in t.text for t in gated_api._session.turns)


def test_other_tools_are_not_gated(gated_api):
    gated_api._tool_executor("read_netlist", {"path": "x.asc"})
    assert gated_api._mcp.calls == [("read_netlist", {"path": "x.asc"})]


@pytest.mark.parametrize("mode", [TOOL_MODE_NATIVE, TOOL_MODE_PROMPTED_JSON])
def test_no_tool_mode_can_write_without_approval(tmp_path, mode):
    """The gate is per-executor, not per-mode - adding a mode must not open a way round it.

    Driven through `Api.send_message`, so it also pins that the configured mode is the one
    the loop actually runs in: a prompted model asking to write reaches the same
    `_tool_executor`, is refused there, and never reaches the MCP server at all.
    """
    api = build_gated_api(tmp_path, mode)
    api._tools = PROMPTED_TOOLS
    apply_args = {
        "asc_path": str(tmp_path / "c.asc"),
        "ref": "C1",
        "new_value": "100n",
        "apply": True,
    }
    if mode == TOOL_MODE_NATIVE:
        responses = [
            response(message(None, [("patch_component_value", apply_args)])),
            response(message("Understood, I will show you the diff.")),
        ]
    else:
        responses = [
            prompted(
                {
                    "type": "tool_call",
                    "name": "patch_component_value",
                    "arguments": apply_args,
                }
            ),
            prompted({"type": "answer", "text": "Understood, I will show you the diff."}),
        ]
    api._client = FakeClient(responses)

    out = api.send_message("Apply that fix yourself right now, with apply=true.")

    assert out["ok"], out.get("error")
    assert api._mcp.calls == [], "the write reached the MCP server despite the gate"
    assert "REFUSED" in out["tool_calls"][0]["result"]
    assert out["text"] == "Understood, I will show you the diff."


# --- what the model is told when a circuit is selected ----------------------------------


class StaticCheckMCP:
    """A server stand-in that answers check_netlist_static with two real-shaped findings."""

    PAYLOAD = {
        "source_path": "c.asc",
        "ok": False,
        "summary": "2 problems: 1 error, 1 warning",
        "findings": [
            {
                "check": "no_dc_path",
                "severity": "error",
                "message": "Node vout has no DC path to ground.",
                "refs": ["C1"],
                "nets": ["vout"],
                "line_no": 4,
                "suggestion": "Add a resistor from vout to 0.",
            },
            {
                "check": "single_connection_net",
                "severity": "warning",
                "message": "Net n002 has only one connection.",
                "refs": ["R1"],
                "nets": ["n002"],
                "line_no": None,
                "suggestion": None,
            },
        ],
    }

    def call_tool(self, name, arguments, timeout=300.0):
        return json.dumps(self.PAYLOAD)


def test_the_selection_note_carries_the_findings_not_just_a_count(tmp_path):
    """The checks are already computed and paid for; summarising them away wastes a round.

    "Found 2 errors" tells the model something is wrong without saying what, which is an
    invitation to re-run the check it was just told about.
    """
    api = build_gated_api(tmp_path)
    api._mcp = StaticCheckMCP()
    circuit = tmp_path / "c.asc"
    circuit.write_text("Version 4.1\n", encoding="utf-8")

    result = api.select_circuit(str(circuit))
    assert result["ok"], result.get("error")

    note = api._history[-1]["content"]
    assert str(circuit) in note
    assert "2 problems: 1 error, 1 warning" in note
    # The actual findings, with the details the model would otherwise have to ask for.
    assert "error/no_dc_path" in note
    assert "Node vout has no DC path to ground." in note
    assert "Add a resistor from vout to 0." in note
    assert "warning/single_connection_net" in note
    assert "[C1, vout]" in note
    # And an instruction not to repeat the check it has just been handed.
    assert "not run it again" in note
    # UI-only metadata stays out of the context.
    assert "shadowed" not in note and "source_path" not in note
    # Whatever the model is told, the UI still gets the whole result.
    assert result["checks"] == StaticCheckMCP.PAYLOAD


def test_a_long_finding_list_is_capped_in_the_model_context(tmp_path):
    from spice_mcp_app.api import MAX_PRELOADED_FINDINGS

    api = build_gated_api(tmp_path)
    findings = [
        {"check": f"c{i}", "severity": "error", "message": f"m{i}"}
        for i in range(MAX_PRELOADED_FINDINGS + 3)
    ]
    note = Api._selection_note(
        tmp_path / "c.asc", {"summary": "many", "findings": findings}
    )

    assert note.count("error/c") == MAX_PRELOADED_FINDINGS
    assert "and 3 more" in note


def test_missing_api_key_is_reported_not_raised(tmp_path, monkeypatch):
    """A missing key must surface in the UI banner, not crash the window on open."""
    monkeypatch.setenv("TAMU_API_KEY", "")
    monkeypatch.setattr("spice_mcp_app.config.load_dotenv", lambda *a, **k: None)

    api = Api()
    started = api.start()
    assert started["ok"] is False
    assert "TAMU_API_KEY" in started["error"]


def test_llm_error_is_the_type_the_ui_catches():
    assert issubclass(LLMError, RuntimeError)
