"""Tests for the agent loop, schema translation, and the approval gate.

All offline: `FakeClient` returns canned Messages-API responses. The loop's job is to keep
asking until the model stops requesting tools, feed results back in the shape the API
expects, and record usage for every round - none of which needs a network to verify, and
all of which would be expensive and flaky to test against live Bedrock.

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
"""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from spice_mcp_app.api import Api
from spice_mcp_app.llm import (
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    BedrockClient,
    LLMError,
    append_user_note,
    mcp_tools_to_anthropic,
    run_agent_turn,
)
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

    def complete(self, messages, *, tools=None, system=SYSTEM_PROMPT, max_tokens=None):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "system": system}
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


def _bedrock_client_raising(exc, fake_config):
    """A BedrockClient whose transport raises, without constructing a real one."""
    client = object.__new__(BedrockClient)
    client._config = fake_config

    class Boom:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise exc

    client._client = Boom()
    return client


def test_an_unavailable_model_says_which_region_and_what_to_do(fake_config):
    """The most likely first-run failure, and the SDK's own message does not explain it."""
    response = httpx2.Response(
        404, request=httpx2.Request("POST", "https://bedrock.invalid/x")
    )
    client = _bedrock_client_raising(
        anthropic.NotFoundError("no such model", response=response, body=None), fake_config
    )

    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    assert "us-east-1" in str(caught.value)
    assert "fake-model" in str(caught.value)
    assert "list_bedrock_models" in str(caught.value)


def test_a_botocore_failure_becomes_an_llm_error(fake_config):
    """Credential resolution and SigV4 signing raise before the anthropic layer sees it."""
    client = _bedrock_client_raising(RuntimeError("Unable to locate credentials"), fake_config)

    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    assert "Unable to locate credentials" in str(caught.value)


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


def test_missing_credentials_is_reported_not_raised(tmp_path):
    """No AWS setup must surface in the UI banner, not crash the window on open.

    The autouse `isolated_aws_environment` fixture is what makes this a real test: it
    removes every `AWS_*` variable and moves `~` to a temp directory, so the chain has
    genuinely nothing to find even on a machine that is configured for AWS.
    """
    started = Api().start()

    assert started["ok"] is False
    assert "AWS_ACCESS_KEY_ID" in started["error"]
