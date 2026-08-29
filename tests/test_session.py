"""Tests for the session log.

The log is the deliverable for the external cost comparison, so these tests are about
the schema being exactly as specified and the token counts being complete. A log that
is subtly reshaped, or that silently records zeros, would invalidate the comparison
without failing anything else.
"""

from __future__ import annotations

import json

from spice_mcp_app.session import Session, Turn

EXPECTED_KEYS = [
    "session_id",
    "started_at",
    "model",
    "circuit_file",
    "turns",
    "total_input_tokens",
    "total_output_tokens",
    "resolved",
]


def make_session(tmp_path, **kwargs) -> Session:
    return Session(model="test-model", sessions_dir=tmp_path, **kwargs)


def test_schema_keys_and_order(tmp_path):
    session = make_session(tmp_path)
    assert list(session.to_dict().keys()) == EXPECTED_KEYS


def test_openai_usage_names_are_mapped(tmp_path):
    """TAMU returns prompt_tokens/completion_tokens; the log must not leak those names."""
    session = make_session(tmp_path)
    session.add_turn(
        Turn.from_usage(
            "assistant",
            "hello",
            {"prompt_tokens": 120, "completion_tokens": 34, "total_tokens": 154},
        )
    )

    turn = session.to_dict()["turns"][0]
    assert turn["input_tokens"] == 120
    assert turn["output_tokens"] == 34
    assert "prompt_tokens" not in turn
    assert "completion_tokens" not in turn


def test_missing_usage_warns_but_does_not_crash(tmp_path, caplog):
    """A null count would silently under-report cost, so it has to be noisy."""
    session = make_session(tmp_path)
    with caplog.at_level("WARNING"):
        session.add_turn(Turn.from_usage("assistant", "hi", None))

    assert "incomplete usage" in caplog.text
    turn = session.to_dict()["turns"][0]
    assert turn["input_tokens"] == 0 and turn["output_tokens"] == 0


def test_user_turn_without_usage_does_not_warn(tmp_path, caplog):
    """Only assistant turns are billed, so a user turn with no usage is normal."""
    session = make_session(tmp_path)
    with caplog.at_level("WARNING"):
        session.add_turn(Turn(role="user", text="what is wrong?"))
    assert "incomplete usage" not in caplog.text


def test_totals_are_the_sum_of_turns(tmp_path):
    session = make_session(tmp_path)
    session.add_turn(Turn("assistant", "a", 100, 10))
    session.add_turn(Turn("assistant", "b", 250, 40))

    data = session.to_dict()
    assert data["total_input_tokens"] == 350
    assert data["total_output_tokens"] == 50
    assert data["total_input_tokens"] == sum(t["input_tokens"] for t in data["turns"])


def test_tool_calls_are_omitted_when_absent(tmp_path):
    """`tool_calls` is optional in the schema; an empty list would be noise."""
    session = make_session(tmp_path)
    session.add_turn(Turn("assistant", "plain answer", 1, 1))
    session.add_turn(
        Turn("assistant", "used a tool", 1, 1, tool_calls=[{"name": "read_netlist"}])
    )

    turns = session.to_dict()["turns"]
    assert "tool_calls" not in turns[0]
    assert turns[1]["tool_calls"] == [{"name": "read_netlist"}]


def test_every_turn_is_flushed_to_disk(tmp_path):
    """A session that crashes mid-debug must still leave a usable log."""
    session = make_session(tmp_path)
    session.add_turn(Turn("user", "first"))

    on_disk = json.loads(session.path.read_text(encoding="utf-8"))
    assert len(on_disk["turns"]) == 1

    session.add_turn(Turn("assistant", "second", 5, 5))
    on_disk = json.loads(session.path.read_text(encoding="utf-8"))
    assert len(on_disk["turns"]) == 2
    assert on_disk["total_input_tokens"] == 5


def test_no_temp_files_are_left_behind(tmp_path):
    """The atomic write must not litter the sessions directory."""
    session = make_session(tmp_path)
    for i in range(3):
        session.add_turn(Turn("user", f"turn {i}"))

    assert [p.name for p in tmp_path.iterdir()] == [session.path.name]


def test_circuit_file_and_resolved_round_trip(tmp_path):
    session = make_session(tmp_path)
    assert session.to_dict()["circuit_file"] is None
    assert session.to_dict()["resolved"] is False

    session.set_circuit_file(tmp_path / "x.asc")
    session.mark_resolved(True)

    on_disk = json.loads(session.path.read_text(encoding="utf-8"))
    assert on_disk["circuit_file"].endswith("x.asc")
    assert on_disk["resolved"] is True


def test_export_matches_the_live_log(tmp_path):
    session = make_session(tmp_path)
    session.add_turn(Turn("assistant", "answer", 7, 3))

    target = session.export_to(tmp_path / "out" / "exported.json")
    assert json.loads(target.read_text(encoding="utf-8")) == session.to_dict()


def test_unicode_survives_the_round_trip(tmp_path):
    """LTspice output is full of Ω, µ and °; the log must not mangle or escape them."""
    session = make_session(tmp_path)
    session.add_turn(Turn("assistant", "C1 is 1µF, R1 is 1.6 kΩ at 25 °C", 1, 1))

    text = session.path.read_text(encoding="utf-8")
    assert "1µF" in text and "kΩ" in text and "°C" in text
