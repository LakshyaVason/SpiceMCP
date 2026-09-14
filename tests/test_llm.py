"""Tests for the agent loop, schema translation, and the approval gate.

All offline: `FakeClient` returns canned Messages-API responses. The loop's job is to keep
asking until the model stops requesting tools, feed results back in the shape the API
expects, and record usage for every round - none of which needs a network to verify, and
all of which would be expensive and flaky to test against the live gateway.

The fakes are **objects, not dicts**, because that is what the SDK returns; `_block_field`
tolerates both, and one test below covers the dict path deliberately.

Several of these tests exist because of a specific failure. The transport was previously
OpenAI-shaped and the model narrated its tool calls as plain prose instead of emitting
tool_use blocks - so "a tool_use block is actually executed", "all results from one round
go back in a single message", and "thinking blocks survive the round trip" are each
guarding a way that can silently stop working again.

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
from dataclasses import replace

import anthropic
import httpx2
import pytest

from spice_mcp_app.api import PATCH_TOOL, Api
from spice_mcp_app.compact import compact_tool_result
from spice_mcp_app.config import TOOL_MODE_NATIVE, TOOL_MODE_PROMPTED_JSON
from spice_mcp_app.llm import (
    MAX_PROTOCOL_CORRECTIONS,
    MAX_TOKENS,
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    GatewayClient,
    LLMError,
    ProtocolError,
    append_user_note,
    mcp_tools_to_anthropic,
    parse_prompted_reply,
    prompted_protocol_prompt,
    run_agent_turn,
)
from spice_mcp_app.mcp_client import MCPClientError
from spice_mcp_app.session import Session


class FakeTool:
    """Stands in for an mcp.types.Tool, which is snake_case on the Python side."""

    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class Block:
    """A content block, shaped like the SDK's - attributes, not keys."""

    def __init__(self, type, **fields):
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class Usage:
    def __init__(self, input_tokens=100, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class Response:
    def __init__(self, content, stop_reason="end_turn", input_tokens=100, output_tokens=20):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = Usage(input_tokens, output_tokens)


def answer(text, **kwargs):
    """A finished reply: text only, stop_reason end_turn."""
    return Response([Block("text", text=text)], **kwargs)


def wants_tools(text, tool_calls, **kwargs):
    """A reply that asks for one or more tools, the way stop_reason "tool_use" arrives."""
    content = [Block("text", text=text)] if text else []
    content += [
        Block("tool_use", id=f"toolu_{i}", name=name, input=args)
        for i, (name, args) in enumerate(tool_calls)
    ]
    return Response(content, stop_reason="tool_use", **kwargs)


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.model = "fake-model"

    def complete(
        self,
        messages,
        *,
        tools=None,
        system=SYSTEM_PROMPT,
        max_tokens=None,
        stop_sequences=None,
    ):
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "system": system,
                "stop_sequences": stop_sequences,
            }
        )
        if not self._responses:
            raise AssertionError("the loop asked for more responses than were provided")
        return self._responses.pop(0)


@pytest.fixture
def session(tmp_path):
    return Session(model="fake-model", sessions_dir=tmp_path)


def tool_results(message):
    """The tool_result blocks of a message the loop appended."""
    return [b for b in message["content"] if b.get("type") == "tool_result"]


# --- prompted-JSON helpers ------------------------------------------------------------

PROMPTED_TOOLS = mcp_tools_to_anthropic(
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


def raw(content, **kwargs):
    """A reply whose visible text is whatever the model actually emitted."""
    return Response([Block("text", text=content)], **kwargs)


def prompted(payload, **kwargs):
    """A reply that is one protocol JSON object, as the real model sends it."""
    return raw(json.dumps(payload), **kwargs)


def cut_off(content, **kwargs):
    """A reply the provider stopped at its output ceiling: stop_reason max_tokens."""
    return raw(content, stop_reason="max_tokens", **kwargs)


def run_prompted(client, session, text, **kwargs):
    kwargs.setdefault("tools", PROMPTED_TOOLS)
    return run_agent_turn(
        client, session, text, tool_mode=TOOL_MODE_PROMPTED_JSON, **kwargs
    )


# --- schema translation ---------------------------------------------------------------


def test_mcp_tools_translate_to_anthropic_tools():
    """Barely a translation - which is the point. MCP already uses `input_schema`."""
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    translated = mcp_tools_to_anthropic([FakeTool("read_netlist", "Reads it.", schema)])

    assert translated == [
        {
            "name": "read_netlist",
            "description": "Reads it.",
            "input_schema": schema,
        }
    ]


def test_translation_accepts_camel_case_dicts():
    """Wire-format dicts use inputSchema; both spellings must work."""
    translated = mcp_tools_to_anthropic(
        [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]
    )
    assert translated[0]["input_schema"] == {"type": "object"}


def test_translation_substitutes_an_empty_schema():
    """`input_schema` is required by the API, so a null one has to become an empty object."""
    translated = mcp_tools_to_anthropic([FakeTool("no_args")])
    assert translated[0]["input_schema"] == {"type": "object", "properties": {}}


def test_translation_skips_nameless_tools():
    assert mcp_tools_to_anthropic([{"description": "no name"}]) == []


# --- the response policy in the system prompt -------------------------------------------
#
# The prompt is the only lever on verbosity that exists here: `temperature` was removed from
# Claude Opus 5 and is a 400 on this gateway, and a small `max_tokens` would truncate a
# legitimate answer rather than shorten it. So these assertions are on prompt *text* - weak
# evidence about the model, but the strongest available offline, and enough to fail loudly if
# a future edit puts the lecture back.


def test_the_prompt_asks_for_fault_then_fix_then_reason():
    ordering = SYSTEM_PROMPT.index("Lead with the fault. Then the specific fix.")
    assert ordering > 0
    assert "two to\n    five sentences" in SYSTEM_PROMPT


def test_the_prompt_rules_out_the_three_digressions_from_the_baseline():
    """The baseline answer added divider gain, a cutoff derivation and an aside about
    `AC 0.7 3000`, to a question that asked about none of them."""
    assert "do not tour the" in SYSTEM_PROMPT
    assert "characteristics the user did not ask about" in SYSTEM_PROMPT
    assert "secondary observations unless they change the answer" in SYSTEM_PROMPT
    # And the instruction that invited the arithmetic in the first place is gone.
    assert "Show the numbers you relied on" not in SYSTEM_PROMPT


def test_brevity_is_a_default_and_not_a_ceiling():
    """A hard cap would be the wrong fix: a genuinely hard circuit has to be allowed room,
    and a stated spec still has to be checked arithmetically and shown."""
    assert "Brevity is the default, not a ceiling." in SYSTEM_PROMPT
    assert "show the arithmetic when you do" in SYSTEM_PROMPT
    assert MAX_TOKENS == 16000


def test_the_generic_tool_ordering_is_gone():
    """`read_netlist -> check_netlist_static -> run_simulation` as a numbered sequence is
    what produced the redundant round: the model followed the list rather than asking what
    it was missing."""
    assert "An efficient order of work" not in SYSTEM_PROMPT
    assert "1. read_netlist" not in SYSTEM_PROMPT
    assert "There is no fixed order." in SYSTEM_PROMPT
    assert "Verified information already in this conversation counts as read." in SYSTEM_PROMPT
    assert "Call a tool only for something you do not already have." in SYSTEM_PROMPT


def test_the_ltspice_correctness_facts_survive_the_rewrite():
    """These are why the answers are right, and they are cheap. Cutting them to save tokens
    would be trading correctness for the metric."""
    assert "M means milli, not mega" in SYSTEM_PROMPT
    assert 'A simulation that "succeeds" can still be wrong.' in SYSTEM_PROMPT
    assert "Neither subsumes the" in SYSTEM_PROMPT
    assert "can still have a wrong value" in SYSTEM_PROMPT


def test_the_prompted_protocol_carries_the_same_evidence_rule():
    """Prompted mode gets its own copy of the header, so a rule added only to the shared
    prompt would still apply - but the protocol rules are what that mode's model reads most
    closely, and the redundant call is a protocol-shaped mistake there."""
    protocol = prompted_protocol_prompt([])
    assert "already stated in this conversation as tool output" in protocol


# --- the loop -------------------------------------------------------------------------


def test_a_plain_answer_ends_the_loop(session):
    client = FakeClient([answer("It is a low-pass filter.")])
    result = run_agent_turn(client, session, "what is it?")

    assert result.text == "It is a low-pass filter."
    assert result.rounds == 1
    assert result.tool_calls == []
    assert len(client.calls) == 1


def test_the_system_prompt_is_passed_out_of_band(session):
    """The Messages API has no system *role*; it is a top-level parameter.

    Leaving a `{"role": "system"}` message in the history would be rejected outright, and
    sending the prompt on only the first round would quietly drop it mid-conversation.
    """
    client = FakeClient([answer("a"), answer("b")])
    history: list[dict] = []

    run_agent_turn(client, session, "first", history=history)
    run_agent_turn(client, session, "second", history=history)

    assert [c["system"] for c in client.calls] == [SYSTEM_PROMPT, SYSTEM_PROMPT]
    assert not any(m["role"] == "system" for m in history)


def test_tool_results_are_fed_back_and_the_loop_continues(session):
    client = FakeClient(
        [
            wants_tools("Looking.", [("read_netlist", {"path": "x.asc"})]),
            answer("R1 is 1.6k."),
        ]
    )
    executed: list[tuple[str, dict]] = []

    def executor(name, arguments):
        executed.append((name, arguments))
        return '{"components": []}'

    result = run_agent_turn(
        client, session, "read it", tool_executor=executor, tools=[{"name": "x"}]
    )

    assert executed == [("read_netlist", {"path": "x.asc"})]
    assert result.rounds == 2
    assert result.text == "R1 is 1.6k."

    # The second request must carry the assistant turn and a matching tool_result, keyed
    # by tool_use_id - a mismatched or missing id is rejected by the API.
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-1]["role"] == "user"
    assert tool_results(second[-1]) == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_0",
            "content": '{"components": []}',
            "is_error": False,
        }
    ]


def test_the_assistant_turn_is_handed_back_verbatim(session):
    """The tool_use blocks must return unchanged, or the results cannot be matched up."""
    asked = wants_tools("Looking.", [("read_netlist", {"path": "x.asc"})])
    client = FakeClient([asked, answer("done")])

    run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "{}", tools=[{"name": "x"}]
    )

    assert client.calls[1]["messages"][-2]["content"] is asked.content


def test_all_results_from_one_round_go_back_in_one_message(session):
    """Parallel tool_use blocks must not be answered across separate messages."""
    client = FakeClient(
        [
            wants_tools(
                "Both, please.",
                [("read_netlist", {"path": "x.asc"}), ("check_netlist_static", {"path": "x.asc"})],
            ),
            answer("done"),
        ]
    )
    result = run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: f"result of {n}", tools=[{"name": "x"}]
    )

    assert [r.name for r in result.tool_calls] == ["read_netlist", "check_netlist_static"]
    sent = client.calls[1]["messages"]
    assert sum(1 for m in sent if m["role"] == "user") == 2  # the question, then one result turn
    assert [b["tool_use_id"] for b in tool_results(sent[-1])] == ["toolu_0", "toolu_1"]


def test_usage_is_summed_across_rounds(session):
    client = FakeClient(
        [
            wants_tools("t", [("read_netlist", {})], input_tokens=100, output_tokens=20),
            answer("done", input_tokens=300, output_tokens=50),
        ]
    )
    result = run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "{}", tools=[{"name": "x"}]
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
            wants_tools("t", [("read_netlist", {"path": "nope.asc"})]),
            answer("That file does not exist."),
        ]
    )
    result = run_agent_turn(
        client, session, "read nope.asc", tool_executor=executor, tools=[{"name": "x"}]
    )

    assert result.rounds == 2
    assert result.tool_calls[0].is_error
    assert "No such file" in result.tool_calls[0].result

    sent_back = tool_results(client.calls[1]["messages"][-1])[0]
    assert "No such file" in sent_back["content"]
    assert sent_back["is_error"] is True


def test_a_non_object_tool_input_is_handed_back(session):
    """The API parses tool input for us, so this should be impossible.

    It is checked anyway because of where a non-dict would end up: `api.py`'s approval gate
    reads `arguments.get("apply")`, and on a non-dict that check cannot fire - the gate
    would fail *open* and an unapproved write would reach the schematic.
    """
    called: list[str] = []
    client = FakeClient(
        [
            Response(
                [Block("tool_use", id="toolu_0", name="read_netlist", input="x.asc")],
                stop_reason="tool_use",
            ),
            answer("Sorry, retrying."),
        ]
    )
    result = run_agent_turn(
        client,
        session,
        "go",
        tool_executor=lambda n, a: called.append(n) or "{}",
        tools=[{"name": "x"}],
    )

    assert called == [], "a non-dict input reached the tool executor"
    assert result.tool_calls[0].is_error
    assert result.tool_calls[0].arguments == {}
    assert "not an object" in result.tool_calls[0].result


def test_the_loop_is_bounded(session):
    """A model that never stops asking for tools must not run forever."""
    client = FakeClient(
        [wants_tools("again", [("read_netlist", {})])] * (MAX_TOOL_ROUNDS + 2)
    )
    result = run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "{}", tools=[{"name": "x"}]
    )

    assert result.rounds == MAX_TOOL_ROUNDS
    assert "Stopped after" in result.text
    # The partial work is still logged - its token cost was still incurred.
    assert session.total_input_tokens > 0


def test_tool_calls_without_an_executor_do_not_crash(session):
    client = FakeClient([wants_tools("t", [("read_netlist", {})]), answer("ok")])
    result = run_agent_turn(client, session, "go", tools=[{"name": "x"}])
    assert result.tool_calls[0].is_error
    assert "No tools are available" in result.tool_calls[0].result


def test_thinking_blocks_are_returned_to_the_api_but_not_to_the_user(session):
    """Thinking is on by default on Opus 5, so this is the normal case, not an edge one.

    The API requires thinking blocks back unchanged alongside a tool_use from the same turn,
    and the user must not be shown the model's scratch work as if it were the answer.
    """
    thinking = Block("thinking", thinking="Let me check R1.", signature="sig")
    client = FakeClient(
        [
            Response(
                [thinking, Block("text", text="Looking."),
                 Block("tool_use", id="toolu_0", name="read_netlist", input={})],
                stop_reason="tool_use",
            ),
            answer("R1 is 1.6k."),
        ]
    )
    history: list[dict] = []
    result = run_agent_turn(
        client,
        session,
        "go",
        tool_executor=lambda n, a: "{}",
        tools=[{"name": "x"}],
        history=history,
    )

    assert "Let me check" not in result.text
    assert thinking in history[1]["content"]


def test_a_truncated_answer_says_so(session):
    """max_tokens must not be presented as a finished diagnosis."""
    result = run_agent_turn(
        FakeClient([answer("The cutoff is", stop_reason="max_tokens")]), session, "go"
    )
    assert "incomplete" in result.text


def test_response_dicts_are_handled_as_well_as_objects(session):
    """The SDK returns objects, but a dict must not break the loop."""
    plain = {
        "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    result = run_agent_turn(FakeClient([plain]), session, "go")
    assert result.text == "Part one. Part two."
    assert result.input_tokens == 5


def test_session_records_the_tool_turns(session):
    client = FakeClient(
        [wants_tools("t", [("read_netlist", {"path": "x"})]), answer("done")]
    )
    run_agent_turn(
        client, session, "go", tool_executor=lambda n, a: "RESULT", tools=[{"name": "x"}]
    )

    roles = [t.role for t in session.turns]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.turns[1].tool_calls[0]["name"] == "read_netlist"
    assert session.turns[2].text == "RESULT"


# --- native mode, stated explicitly ----------------------------------------------------


def test_native_mode_sends_the_tools_parameter_and_consumes_tool_use(session):
    """The default behaviour, now that it is one mode of two rather than the only one."""
    client = FakeClient(
        [
            wants_tools("Looking.", [("read_netlist", {"path": "x.asc"})]),
            answer("It is a low-pass."),
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
    # Native mode's whole premise: the tool catalogue travels in the API field, so the
    # protocol text is not paid for in the prompt.
    assert client.calls[0]["tools"] == PROMPTED_TOOLS
    assert "TOOL PROTOCOL" not in client.calls[0]["system"]
    assert client.calls[0]["stop_sequences"] is None


def test_the_default_mode_is_native(session):
    client = FakeClient([answer("hi")])
    run_agent_turn(client, session, "go", tools=PROMPTED_TOOLS)
    assert client.calls[0]["tools"] == PROMPTED_TOOLS


def test_an_unknown_mode_is_rejected_rather_than_defaulted(session):
    with pytest.raises(LLMError, match="Unknown tool mode"):
        run_agent_turn(
            FakeClient([]), session, "go", tool_mode="prompted-json"  # hyphen, not underscore
        )


# --- prompted-JSON mode ----------------------------------------------------------------
#
# The fake here never returns a tool_use block, because the route this mode exists for
# never does: it accepts the `tools` parameter and then answers as though it had no tools
# at all. That is the failure the mode was added for, and these tests drive it.


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

    # The API field is not used in this mode - the catalogue is in the system prompt,
    # which the Messages API takes on every call because there is no system role to hold it.
    assert client.calls[0]["tools"] is None
    assert "TOOL PROTOCOL" in client.calls[0]["system"]
    assert "TOOL PROTOCOL" in client.calls[1]["system"], "the protocol was sent only once"
    assert not any(m["role"] == "system" for m in client.calls[0]["messages"])
    # And generation stops at the label, so a model that starts role-playing the tool
    # result is cut off rather than billed for it.
    assert client.calls[0]["stop_sequences"] == ["TOOL RESULT"]

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
                input_tokens=1000,
                output_tokens=30,
            ),
            prompted(
                {"type": "tool_call", "name": "run_simulation", "arguments": {"path": "x"}},
                input_tokens=2000,
                output_tokens=40,
            ),
            prompted(
                {
                    "type": "tool_call",
                    "name": "patch_component_value",
                    "arguments": {"asc_path": "x"},
                },
                input_tokens=3000,
                output_tokens=50,
            ),
            prompted(
                {"type": "answer", "text": "C1 should be 100n."},
                input_tokens=4000,
                output_tokens=60,
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


def test_a_correction_never_forms_two_consecutive_user_turns(session):
    """The Messages API rejects that outright, so a failed round must fold, not append.

    Under the old OpenAI-shaped transport a second user message was tolerated; here it is
    a 400, which would turn a recoverable formatting slip into a dead turn.
    """
    client = FakeClient(
        [raw("not JSON"), prompted({"type": "answer", "text": "sorry"})]
    )
    history: list[dict] = []

    run_prompted(client, session, "go", history=history)

    roles = [m["role"] for m in history]
    assert not any(a == b for a, b in zip(roles, roles[1:])), roles


def test_an_empty_reply_is_not_sent_back_as_an_empty_assistant_turn(session):
    """The API rejects empty content, and inventing text would falsify the transcript."""
    client = FakeClient([raw(""), prompted({"type": "answer", "text": "sorry"})])
    history: list[dict] = []

    result = run_prompted(client, session, "go", history=history)

    assert result.protocol_errors == 1
    assert all(str(m["content"]).strip() for m in history), history
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_a_reply_cut_off_at_the_output_ceiling_is_diagnosed_as_such(session):
    """The JSON was fine until the cap; "your JSON is invalid" is a misdiagnosis.

    Seen live on the retired gateway, which silently capped a reply at 1024 tokens
    mid-string. The Messages API cannot cap silently, but a long enough answer still can, and
    telling the model to hunt for a syntax error it never made wastes a whole round.
    """
    client = FakeClient(
        [
            cut_off('{"type":"answer","text":"Circuit as read from the netlist:\\n\\n  R1'),
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


def test_prompted_mode_ignores_thinking_blocks_when_parsing(session):
    """Adaptive thinking is on by default, so the JSON arrives after a thinking block."""
    client = FakeClient(
        [
            Response(
                [
                    Block("thinking", thinking="I should read the netlist.", signature="s"),
                    Block(
                        "text",
                        text=json.dumps(
                            {"type": "tool_call", "name": "read_netlist", "arguments": {}}
                        ),
                    ),
                ]
            ),
            prompted({"type": "answer", "text": "done"}),
        ]
    )
    executed: list[str] = []

    result = run_prompted(
        client, session, "go", tool_executor=lambda n, a: executed.append(n) or "{}"
    )

    assert executed == ["read_netlist"]
    assert result.protocol_errors == 0
    # The thinking text must not travel back: there is no tool_use block to attach it to,
    # and the model's scratch work is not part of the protocol exchange.
    assert "I should read" not in str(client.calls[1]["messages"])


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


# --- history shape --------------------------------------------------------------------


def test_notes_never_produce_two_user_turns_in_a_row(session):
    """The API rejects consecutive same-role messages; the app appends notes freely.

    `select_circuit` and `apply_patch` both add a user-role note, and then the user's next
    question adds another. Under the old transport that was tolerated. Here it is a 400,
    so the note has to fold into the pending turn.
    """
    history: list[dict] = []
    append_user_note(history, "[circuit opened]")
    append_user_note(history, "[fix applied]")

    client = FakeClient([answer("ok")])
    run_agent_turn(client, session, "why no gain?", history=history)

    roles = [m["role"] for m in client.calls[0]["messages"]]
    assert roles == ["user"]
    sent = client.calls[0]["messages"][0]["content"]
    assert "[circuit opened]" in sent and "[fix applied]" in sent and "why no gain?" in sent
    assert not any(
        a["role"] == b["role"] for a, b in zip(history, history[1:])
    ), f"consecutive same-role turns: {[m['role'] for m in history]}"


def test_a_note_folds_into_a_pending_block_list_too():
    """After a tool round the trailing user turn holds blocks, not a string."""
    history = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]}]
    append_user_note(history, "[fix applied]")

    assert len(history) == 1
    assert history[0]["content"][-1] == {"type": "text", "text": "[fix applied]"}


# --- error translation ----------------------------------------------------------------


def _gateway_client_raising(exc, fake_config):
    """A GatewayClient whose transport raises, without constructing a real one.

    `object.__new__` skips `__init__`, which is what keeps this offline: the real
    constructor reads TAMU_API_KEY and builds an SDK client, and the autouse fixture has
    deliberately removed the token.
    """
    client = object.__new__(GatewayClient)
    client._config = fake_config

    class Boom:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise exc

    client._client = Boom()
    return client


def _response(status):
    return httpx2.Response(
        status, request=httpx2.Request("POST", "https://gateway.invalid/v1/messages")
    )


def test_an_unrouted_model_says_it_is_the_gateway_and_names_the_model(fake_config):
    """A 404 here is the gateway's routing table, not a broken URL or a missing AWS grant.

    The old message sent the reader to the Bedrock console and to a boto3 helper script.
    On this transport that advice is not merely stale, it is unreachable - the user has no
    AWS access at all - so the wording is asserted, not just the exception type.
    """
    client = _gateway_client_raising(
        anthropic.NotFoundError("no such model", response=_response(404), body=None),
        fake_config,
    )

    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    message = str(caught.value)
    assert "fake-model" in message
    assert "gateway" in message.lower()
    assert "SPICE_MCP_MODEL" in message
    # No AWS advice, ever: there is no console to visit and no region to change.
    assert "region" not in message.lower()
    assert "bedrock" not in message.lower()
    assert "aws" not in message.lower()


@pytest.mark.parametrize(
    "exc_type", [anthropic.AuthenticationError, anthropic.PermissionDeniedError]
)
def test_a_rejected_token_names_the_variable_but_never_its_value(fake_config, exc_type):
    """401 and 403 both mean "this token will not do", and point at the same fix.

    They shared a branch with the 404 on the previous transport, where all three meant
    "no model access". Here they do not, so they are separated.
    """
    status = 401 if exc_type is anthropic.AuthenticationError else 403
    client = _gateway_client_raising(
        exc_type("nope", response=_response(status), body=None), fake_config
    )

    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    message = str(caught.value)
    assert "TAMU_API_KEY" in message
    # The label may appear; the token itself must not exist anywhere to appear from.
    assert "TAMU_API_KEY (9 chars)" in message


def test_an_unreachable_gateway_is_named_as_a_network_problem(fake_config):
    """Distinct from a rejected token: nothing to fix in .env, so do not send them there."""
    client = _gateway_client_raising(
        anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x/")),
        fake_config,
    )

    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    message = str(caught.value)
    assert "https://gateway.invalid/v1/messages" in message
    assert "network" in message.lower()


def test_a_generic_api_error_still_becomes_an_llm_error(fake_config):
    """The UI only catches LLMError; anything narrower would surface as a raw traceback."""
    client = _gateway_client_raising(
        anthropic.BadRequestError("bad", response=_response(400), body=None), fake_config
    )

    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    assert "bad" in str(caught.value)


def test_llm_error_is_the_type_the_ui_catches():
    assert issubclass(LLMError, RuntimeError)


# --- the approval gate ----------------------------------------------------------------


def test_an_unapproved_apply_is_refused(api, tmp_path):
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    result = api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": True},
    )

    assert "REFUSED" in result
    assert api._mcp.calls == [], "the write reached the server despite the gate"


def test_a_preview_is_always_allowed(api, tmp_path):
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": False},
    )

    assert len(api._mcp.calls) == 1
    assert api._mcp.calls[0][1]["apply"] is False


def test_approval_is_single_use(api, tmp_path):
    """One approval, one write; a later apply must be approved again."""
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    approved = api.apply_patch(str(asc), "C1", "100n")
    assert approved["ok"], approved.get("error")
    assert api._mcp.calls[-1][1]["apply"] is True

    # The model trying to repeat it afterwards is refused.
    again = api._tool_executor(
        "patch_component_value",
        {"asc_path": str(asc), "ref": "C1", "new_value": "100n", "apply": True},
    )
    assert "REFUSED" in again


def test_approval_does_not_authorise_a_different_change(api, tmp_path):
    """An approval is for one component and one value, not a licence to write."""
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")
    api._approved.add((str(asc.resolve()), "C1", "100n"))

    for arguments in (
        {"asc_path": str(asc), "ref": "R1", "new_value": "100n", "apply": True},
        {"asc_path": str(asc), "ref": "C1", "new_value": "10n", "apply": True},
    ):
        assert "REFUSED" in api._tool_executor("patch_component_value", arguments)


def test_applying_tells_the_model_what_happened(api, tmp_path):
    """The conversation must stay truthful about the state of the file."""
    asc = tmp_path / "c.asc"
    asc.write_text("Version 4.1\n", encoding="utf-8")

    api.apply_patch(str(asc), "C1", "100n")

    assert any("user approved" in str(m["content"]) for m in api._history)
    assert any("user approved" in t.text for t in api._session.turns)


def test_other_tools_are_not_gated(api):
    api._tool_executor("read_netlist", {"path": "x.asc"})
    assert api._mcp.calls == [("read_netlist", {"path": "x.asc"})]


@pytest.mark.parametrize("mode", [TOOL_MODE_NATIVE, TOOL_MODE_PROMPTED_JSON])
def test_no_tool_mode_can_write_without_approval(api, tmp_path, mode):
    """The gate is per-executor, not per-mode - adding a mode must not open a way round it.

    Driven through `Api.send_message`, so it also pins that the configured mode is the one
    the loop actually runs in: a prompted model asking to write reaches the same
    `_tool_executor`, is refused there, and never reaches the MCP server at all.
    """
    api._config = replace(api._config, tool_mode=mode)
    api._tools = PROMPTED_TOOLS
    apply_args = {
        "asc_path": str(tmp_path / "c.asc"),
        "ref": "C1",
        "new_value": "100n",
        "apply": True,
    }
    if mode == TOOL_MODE_NATIVE:
        responses = [
            wants_tools(None, [("patch_component_value", apply_args)]),
            answer("Understood, I will show you the diff."),
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


# --- the model-facing catalogue ---------------------------------------------------------
#
# Seven tool schemas cost 6588 chars on every round. Two of them - `diff_netlist` and
# `export_netlist`, 1342 chars - are not reachable from a diagnosis or from the Apply flow,
# so they are withheld from the request while `self._tools` stays the full seven.

SEVEN_TOOLS = mcp_tools_to_anthropic(
    [
        FakeTool(name, f"Does {name}.", {"type": "object", "properties": {}})
        for name in (
            "read_netlist",
            "check_netlist_static",
            "run_simulation",
            "read_sim_log",
            "patch_component_value",
            "diff_netlist",
            "export_netlist",
        )
    ]
)


def test_the_diagnosis_catalogue_drops_the_two_unreachable_tools(api):
    api._tools = SEVEN_TOOLS

    names = [t["name"] for t in api._model_tools("why is my circuit not getting any gain")]

    assert names == [
        "read_netlist",
        "check_netlist_static",
        "run_simulation",
        "read_sim_log",
        "patch_component_value",
    ]
    # The full set is untouched: `start()` reports it and `app_smoke.py` stage 1 counts it.
    assert len(api._tools) == 7


def test_patch_component_value_is_never_withheld(api):
    """It is the largest single schema and the most tempting to cut.

    Without a model call there is no `pending_patch`, so no diff panel, no Apply button and
    no approval flow. Any turn text at all must still carry it.
    """
    api._tools = SEVEN_TOOLS
    for text in ("why no gain", "export the netlist", "", "fix C1"):
        assert PATCH_TOOL in [t["name"] for t in api._model_tools(text)]


def test_asking_about_a_diff_or_an_export_widens_the_catalogue(api):
    """The trigger can only widen. A rule that withholds on a guess would block a
    legitimate workflow; one that offers too much merely costs the tokens it was saving."""
    api._tools = SEVEN_TOOLS
    for text in (
        "diff the two netlists for me",
        "compare it against the backup",
        "export this to a .net file",
        "show me before and after",
    ):
        assert len(api._model_tools(text)) == 7, text


def test_the_request_carries_the_trimmed_catalogue(api, tmp_path):
    api._tools = SEVEN_TOOLS
    api._client = FakeClient([answer("No gain because R1's pin is on NC_01.")])

    out = api.send_message("why is my circuit not getting any gain")

    assert out["ok"], out.get("error")
    sent = [t["name"] for t in api._client.calls[0]["tools"]]
    assert "diff_netlist" not in sent and "export_netlist" not in sent
    assert PATCH_TOOL in sent


# --- compact for the model, complete for the record -------------------------------------
#
# The projections themselves are tested in test_compact.py. What matters here is the split:
# the request carries the projection, and everything that is a *record* - the session log,
# `to_dict()` for the UI, `_pending_patch` - still carries the full tool output. Getting
# that backwards would either cost the tokens anyway or quietly gut the session log the
# external cost comparison depends on.


def full_netlist_json():
    from tests.test_compact import RCLP_NETLIST

    return json.dumps(RCLP_NETLIST, indent=2)


def test_the_model_sees_the_projection_while_the_record_keeps_the_full_result(session):
    client = FakeClient(
        [
            wants_tools("Reading it.", [("read_netlist", {"path": "c.asc"})]),
            answer("R1's left pin is on NC_01, so Vin never reaches the filter."),
        ]
    )
    full = full_netlist_json()
    history: list[dict] = []

    result = run_agent_turn(
        client,
        session,
        "why is there no gain",
        tools=[{"name": "read_netlist", "description": "", "input_schema": {}}],
        tool_executor=lambda name, arguments: full,
        history=history,
        compactor=compact_tool_result,
    )

    sent = tool_results(history[2])[0]["content"]
    assert "R1 Vout NC_01 10k" in sent
    assert "spice_mcp_work" not in sent
    assert len(sent) < len(full) / 4

    record = result.tool_calls[0]
    assert record.result == full
    assert record.to_dict()["result"] == full
    # `to_dict` is what reaches the session log and the UI, so the projection must not
    # appear there under any key - a reader reconstructing the turn needs the real output.
    assert "model_result" not in record.to_dict()

    logged = [t for t in session.to_dict()["turns"] if t["role"] == "tool"]
    assert logged[0]["text"] == full


def test_the_projection_reaches_the_model_in_prompted_mode_too(session):
    """Both modes converge on `_execute_tool`, so neither can be the one that leaks bulk."""
    client = FakeClient(
        [
            prompted({"type": "tool_call", "name": "read_netlist", "arguments": {"path": "c.asc"}}),
            prompted({"type": "answer", "text": "R1's pin is dangling on NC_01."}),
        ]
    )
    full = full_netlist_json()
    history: list[dict] = []

    run_prompted(
        client,
        session,
        "why is there no gain",
        tool_executor=lambda name, arguments: full,
        history=history,
        compactor=compact_tool_result,
    )

    result_message = history[2]["content"]
    assert result_message.startswith("TOOL RESULT")
    assert "R1 Vout NC_01 10k" in result_message
    assert "netlist_path" not in result_message


def test_no_compactor_means_the_full_result_goes_to_the_model(session):
    """The default, and what every other test in this file relies on.

    Compaction is the caller's decision; the loop must not acquire an opinion of its own.
    """
    client = FakeClient(
        [
            wants_tools("", [("read_netlist", {"path": "c.asc"})]),
            answer("done"),
        ]
    )
    full = full_netlist_json()
    history: list[dict] = []

    run_agent_turn(
        client,
        session,
        "q",
        tools=[{"name": "read_netlist", "description": "", "input_schema": {}}],
        tool_executor=lambda name, arguments: full,
        history=history,
    )

    assert tool_results(history[2])[0]["content"] == full


def test_a_declining_compactor_sends_the_result_unchanged(session):
    """`None` is the fail-open answer, and it has to survive the plumbing.

    An error result - the approval gate's REFUSED prose, for instance - travels this path,
    and it is often the actual diagnosis.
    """
    client = FakeClient(
        [
            wants_tools("", [("patch_component_value", {"asc_path": "c.asc"})]),
            answer("Here is the diff."),
        ]
    )
    refusal = "REFUSED: writing to the schematic needs the user's approval first."
    history: list[dict] = []

    run_agent_turn(
        client,
        session,
        "fix it",
        tools=[{"name": "patch_component_value", "description": "", "input_schema": {}}],
        tool_executor=lambda name, arguments: refusal,
        history=history,
        compactor=compact_tool_result,
    )

    assert tool_results(history[2])[0]["content"] == refusal


def test_token_accounting_still_sums_across_rounds_with_compaction_active(session):
    """The metric the whole experiment is judged on must not be the thing it breaks."""
    client = FakeClient(
        [
            wants_tools("", [("read_netlist", {"path": "c.asc"})], input_tokens=900, output_tokens=40),
            wants_tools("", [("run_simulation", {"path": "c.asc"})], input_tokens=700, output_tokens=30),
            answer("It solves now.", input_tokens=800, output_tokens=120),
        ]
    )
    full = full_netlist_json()

    result = run_agent_turn(
        client,
        session,
        "check it",
        tools=[
            {"name": "read_netlist", "description": "", "input_schema": {}},
            {"name": "run_simulation", "description": "", "input_schema": {}},
        ],
        tool_executor=lambda name, arguments: full,
        history=[],
        compactor=compact_tool_result,
    )

    assert result.rounds == 3
    assert (result.input_tokens, result.output_tokens) == (2400, 190)
    data = session.to_dict()
    assert data["total_input_tokens"] == 2400
    assert data["total_output_tokens"] == 190
    assert data["total_input_tokens"] == sum(t["input_tokens"] for t in data["turns"])
    assert data["total_output_tokens"] == sum(t["output_tokens"] for t in data["turns"])


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


def test_the_selection_note_carries_the_findings_not_just_a_count(api, tmp_path):
    """The checks are already computed and paid for; summarising them away wastes a round.

    "Found 2 errors" tells the model something is wrong without saying what, which is an
    invitation to re-run the check it was just told about.
    """
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
    assert "do not call these again" in note
    assert "check_netlist_static" in note
    # This stand-in answers *every* tool name with the static payload, so the topology
    # preload gets a StaticCheckResult where a Netlist should be. `circuit_summary` fails
    # closed on that - and the note must then say the topology is missing rather than claim
    # a read that did not happen.
    assert "topology was NOT read" in note
    assert "Components:" not in note
    # UI-only metadata stays out of the context.
    assert "shadowed" not in note and "source_path" not in note
    # Whatever the model is told, the UI still gets the whole result.
    assert result["checks"] == StaticCheckMCP.PAYLOAD


def test_a_long_finding_list_is_capped_in_the_model_context(tmp_path):
    from spice_mcp_app.api import MAX_PRELOADED_FINDINGS

    findings = [
        {"check": f"c{i}", "severity": "error", "message": f"m{i}"}
        for i in range(MAX_PRELOADED_FINDINGS + 3)
    ]
    note = Api._selection_note(
        tmp_path / "c.asc", {"summary": "many", "findings": findings}
    )

    assert note.count("error/c") == MAX_PRELOADED_FINDINGS
    assert "and 3 more" in note


# --- the preloaded topology, and the round it removes ------------------------------------
#
# The baseline spent a whole extra inference round on `read_netlist`, and that was not the
# model being wasteful: the static *findings* were in context but the components were not,
# so it had no way to name the fix. These tests cover both halves of the fix - the topology
# being there, and the note telling the truth about whether it is.


class PreloadingMCP:
    """A server stand-in that answers both preload tools with real captured RCLP output."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, arguments, timeout=300.0):
        from tests.test_compact import RCLP_NETLIST, RCLP_STATIC

        self.calls.append((name, arguments))
        if name == "check_netlist_static":
            return json.dumps(RCLP_STATIC, indent=2)
        if name == "read_netlist":
            return json.dumps(RCLP_NETLIST, indent=2)
        raise AssertionError(f"the test did not expect a {name} call")


class NoTopologyMCP:
    """A server whose *preload* read fails, the way an ExpressPCB `.net` really does.

    The preload is the call carrying `include_raw_text=False`; a call the model makes itself
    does not, which is how this stand-in tells the two apart. So the selection note is
    honest about having no topology, and the tool remains available when the model asks.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, arguments, timeout=300.0):
        from tests.test_compact import RCLP_NETLIST

        self.calls.append((name, arguments))
        if name == "check_netlist_static":
            return json.dumps({"summary": "Static checks clean.", "findings": [], "ok": True})
        if name == "read_netlist":
            if arguments.get("include_raw_text") is False:
                raise MCPClientError("Tool read_netlist failed: not a SPICE netlist.")
            return json.dumps(RCLP_NETLIST, indent=2)
        raise AssertionError(f"the test did not expect a {name} call")


def _select(api, tmp_path, mcp):
    api._mcp = mcp
    circuit = tmp_path / "RCLP.asc"
    circuit.write_text("Version 4.1\n", encoding="utf-8")
    result = api.select_circuit(str(circuit))
    assert result["ok"], result.get("error")
    return circuit


def test_selection_preloads_the_topology_and_says_which_tools_ran(api, tmp_path):
    """Selection runs both read tools once, and the note names exactly those two."""
    mcp = PreloadingMCP()
    _select(api, tmp_path, mcp)

    assert [name for name, _ in mcp.calls] == ["check_netlist_static", "read_netlist"]
    # The projection drops `raw_text`, so asking the server for it would be pure overhead.
    assert mcp.calls[1][1]["include_raw_text"] is False

    note = api._history[-1]["content"]
    assert "check_netlist_static" in note and "read_netlist" in note
    assert "do not call these again unless the file changes" in note
    assert "topology was NOT read" not in note
    # Findings *and* topology. Either one alone leaves the model a reason to spend a round.
    assert "floating_pin" in note and "NC_01" in note
    assert "R1 Vout NC_01 10k" in note
    assert "NC_01: R1" in note
    assert ".ac dec 10 1k 100k" in note
    # The projection, not the payload: no staging paths, no duplicate netlist text.
    assert "spice_mcp_work" not in note
    assert "Generated by LTspice" not in note


def test_a_preloaded_fault_can_be_answered_without_a_single_tool_call(api, tmp_path):
    """The round the experiment is trying to remove, removed.

    **Honest limit:** this proves the loop *can* finish in one round and that the context
    holds everything needed to. It cannot prove the live model will choose to - only the A/B
    run shows that. What it does pin is that a one-round answer is reachable, so a future
    change that puts the topology back out of context fails here rather than only on the
    bench.
    """
    mcp = PreloadingMCP()
    _select(api, tmp_path, mcp)
    api._tools = PROMPTED_TOOLS
    api._client = FakeClient(
        [
            answer(
                "Your input is disconnected. R1's left pin is floating on NC_01, so Vin "
                "never reaches the filter. Wire that pin to Vin and re-run."
            )
        ]
    )

    out = api.send_message("why is my circuit not getting any gain")

    assert out["ok"], out.get("error")
    assert out["tool_calls"] == []
    assert len(api._client.calls) == 1, "a second inference round means a tool was called"
    # Only the two preload calls; nothing the model asked for.
    assert len(mcp.calls) == 2
    # And the evidence really was in front of it.
    sent = json.dumps(api._client.calls[0]["messages"])
    assert "NC_01" in sent and "10k" in sent


@pytest.mark.parametrize("mode", [TOOL_MODE_NATIVE, TOOL_MODE_PROMPTED_JSON])
def test_the_model_can_still_read_the_netlist_when_the_preload_failed(api, tmp_path, mode):
    """The other direction, and the more important one.

    A prompt that discourages redundant calls plus a note that wrongly claims the topology
    is loaded would leave the model answering from nothing. So when the preload fails the
    tool has to stay reachable, and its result has to come back - in both tool modes.
    """
    api._config = replace(api._config, tool_mode=mode)
    mcp = NoTopologyMCP()
    circuit = _select(api, tmp_path, mcp)
    api._tools = PROMPTED_TOOLS

    note = api._history[-1]["content"]
    assert "topology was NOT read" in note
    assert "read_netlist" in note
    assert "Components:" not in note, "the note claimed a read that failed"

    read_args = {"path": str(circuit)}
    if mode == TOOL_MODE_NATIVE:
        responses = [
            wants_tools(None, [("read_netlist", read_args)]),
            answer("R1's left pin sits on NC_01."),
        ]
    else:
        responses = [
            prompted({"type": "tool_call", "name": "read_netlist", "arguments": read_args}),
            prompted({"type": "answer", "text": "R1's left pin sits on NC_01."}),
        ]
    api._client = FakeClient(responses)

    out = api.send_message("why is my circuit not getting any gain")

    assert out["ok"], out.get("error")
    assert [c["name"] for c in out["tool_calls"]] == ["read_netlist"]
    assert ("read_netlist", read_args) in mcp.calls
    # The record keeps the full payload even though the model saw the projection.
    assert "netlist_path" in out["tool_calls"][0]["result"]
    assert out["text"] == "R1's left pin sits on NC_01."


def test_applying_a_fix_marks_the_preloaded_context_stale(api, tmp_path):
    """The counterweight to "do not call these again".

    Once the file is written, the preloaded check and topology describe the *old* circuit.
    Without saying so, the same instruction that saves a round before the fix would suppress
    the re-simulation that confirms it.
    """
    from tests.conftest import FakeMCP

    mcp = PreloadingMCP()
    circuit = _select(api, tmp_path, mcp)
    api._mcp = FakeMCP()

    approved = api.apply_patch(str(circuit), "C1", "100n")
    assert approved["ok"], approved.get("error")

    note = api._history[-1]["content"]
    assert "stale" in note
    assert "re-read" in note


def test_missing_credentials_is_reported_not_raised(tmp_path):
    """A missing token must surface in the UI banner, not crash the window on open.

    The autouse `isolated_credential_environment` fixture is what makes this a real test:
    it removes `TAMU_API_KEY` and neuters `load_dotenv`, so the token is genuinely absent
    even on this machine, whose `.env` holds a live one.
    """
    started = Api().start()

    assert started["ok"] is False
    assert "TAMU_API_KEY" in started["error"]
    # The banner has to say where to put it, or it is not actionable.
    assert ".env" in started["error"]
