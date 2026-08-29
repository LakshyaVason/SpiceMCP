"""Tests for the agent loop, schema translation, and the approval gate.

All offline: `FakeClient` returns canned OpenAI-shaped responses. The loop's job is to
keep asking until the model stops requesting tools, feed results back in the shape the
provider expects, and record usage for every round - none of which needs a network to
verify, and all of which would be expensive and flaky to test against the live proxy.

The approval-gate tests are the important ones here. "Nothing touches disk before user
approval" is a promise to the user about their schematic, and a model can ask for
apply=True whenever it likes, so the refusal has to be tested rather than assumed.
"""

from __future__ import annotations

import json

import pytest

from spice_mcp_app.api import Api
from spice_mcp_app.config import Config
from spice_mcp_app.llm import (
    MAX_TOOL_ROUNDS,
    LLMError,
    mcp_tools_to_openai,
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

    def complete(self, messages, *, tools=None, max_tokens=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._responses:
            raise AssertionError("the loop asked for more responses than were provided")
        return self._responses.pop(0)


@pytest.fixture
def session(tmp_path):
    return Session(model="fake-model", sessions_dir=tmp_path)


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


# --- the approval gate ----------------------------------------------------------------


class FakeMCP:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, arguments, timeout=300.0):
        self.calls.append((name, arguments))
        return json.dumps({"applied": bool(arguments.get("apply")), "summary": "ok"})


@pytest.fixture
def gated_api(tmp_path):
    config = Config(
        api_key="not-a-real-key",
        model="fake-model",
        base_url="https://example.invalid/openai",
        sessions_dir=tmp_path,
    )
    api = Api(config=config)
    api._mcp = FakeMCP()
    api._session = Session(model="fake-model", sessions_dir=tmp_path)
    return api


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
