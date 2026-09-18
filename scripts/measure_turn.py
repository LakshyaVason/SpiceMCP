"""Measure what one debugging turn costs: rounds, tool calls, tokens, answer length.

The instrument for the token/verbosity experiment. BEFORE and AFTER have to be produced
by the *same* code to be comparable, which is why this lands as the experiment branch's
first commit - before anything that changes behaviour - and why it drives the real
`spice_mcp_app.api.Api` rather than a parallel harness, exactly as `scripts/app_smoke.py`
does.

    .venv\\Scripts\\activate

    REM the professor's case: the whole point of the experiment
    python scripts/measure_turn.py --circuit RCLP.asc \
        --question "why is my circuit not getting any gain" \
        --expect R1 --expect NC_01 --label before

    REM free: no inference, just the size of the request that would be sent
    python scripts/measure_turn.py --circuit RCLP.asc \
        --question "why is my circuit not getting any gain" --count-only

Two modes, and the difference matters:

  * default - runs the turn for real and **spends tokens**. Token counts come from the
    session log's own totals, so this cannot report a saving the external
    screenshot-vs-MCP comparison will not also see.
  * `--count-only` - calls `messages.count_tokens` instead. The gateway prices the input
    and runs no inference, so it is free. This is how a prompt, preload or tool-catalogue
    change gets attributed to the change that caused it without paying for a completion
    per iteration. It says nothing about output tokens or about which tools the model
    would have chosen.

Always runs against a **copy** of the circuit in a temp directory. Hard constraint 2 is
not conditional on intent: LTspice writes `-netlist` output next to its input, so even a
read-only question would litter the user's folder.

Exit code 0 means every `--expect` string was found in the answer. A smaller token count
with a worse answer is a failed experiment, so the two are reported together and the
exit code follows correctness, not cost.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spice_mcp_app.api import Api  # noqa: E402
from spice_mcp_app.config import (  # noqa: E402
    REPO_ROOT,
    TOOL_MODE_PROMPTED_JSON,
    TOOL_MODES,
    load_config,
)
from spice_mcp_app.llm import (  # noqa: E402
    SYSTEM_PROMPT,
    append_user_note,
    prompted_system_prompt,
)

BENCH_DIR = REPO_ROOT / "bench"


def normalized(text: str) -> str:
    """Fold whitespace and case so `100n`, `100 nF` and `100nF` all match.

    Lifted from `app_smoke.py`'s stage-3 check, for the same reason: an expectation that
    fails on a space is an expectation that trains you to ignore the harness.
    """
    return text.replace(" ", "").lower()


def measure(
    api: Api,
    question: str,
    expectations: list[str],
) -> dict[str, Any]:
    """Run one question and return its metric row.

    Tokens are read as a **difference in the session's own totals** rather than summed
    from the turn's `usage`. Same numbers, but sourced from the file the external cost
    comparison reads, so the harness cannot flatter a change the comparison would not see.
    """
    before = api.totals()["totals"]
    turn = api.send_message(question)
    after = api.totals()["totals"]

    if not turn.get("ok"):
        return {"question": question, "ok": False, "error": turn.get("error", "")}

    answer = turn["text"]
    calls = turn["tool_calls"]
    haystack = normalized(answer)
    missing = [e for e in expectations if normalized(e) not in haystack]

    return {
        "question": question,
        "ok": True,
        "rounds": turn["rounds"],
        "tool_calls": [c["name"] for c in calls],
        "tool_call_count": len(calls),
        # How much text each result put in front of the model. The point of comparison
        # for the compaction work, and invisible in the token totals alone.
        "tool_result_chars": [len(c["result"]) for c in calls],
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "total_tokens": (after["input_tokens"] - before["input_tokens"])
        + (after["output_tokens"] - before["output_tokens"]),
        "answer_words": len(answer.split()),
        "answer_chars": len(answer),
        "correct": not missing,
        "missing": missing,
        "answer": answer,
    }


def count_only(api: Api, question: str, tool_mode: str) -> dict[str, Any]:
    """Size the request that *would* be sent, without paying for a completion.

    Reaches into `api._history` and `api._tools` deliberately: the number is only worth
    having if it is the exact `system` + `messages` + `tools` the loop would send, and
    reconstructing that from the public surface would be a second implementation to keep
    in step. A measurement script may know more than the UI does.
    """
    assert api._client is not None
    # Deep, not shallow: `append_user_note` folds into a *list* content block in place when
    # the pending turn has one, which on a shallow copy would edit the live history and
    # make the app's next real turn carry the measurement's question.
    history = copy.deepcopy(api._history)
    append_user_note(history, question)

    # The catalogue the *request* carries, not the seven the server exposes: `_model_tools`
    # uses tool_policy to select based on preload state.
    catalogue = api._model_tools(question)
    if tool_mode == TOOL_MODE_PROMPTED_JSON:
        system = prompted_system_prompt(catalogue)
        tools = None
    else:
        system = SYSTEM_PROMPT
        tools = catalogue

    return {
        "question": question,
        "ok": True,
        "count_only": True,
        "tool_mode": tool_mode,
        "input_tokens": api._client.count_tokens(
            history, tools=tools, system=system
        ),
        "system_chars": len(system),
        "tool_catalogue_names": [t["name"] for t in catalogue],
        "tools_sent": len(catalogue),
    }


def breakdown(api: Api, question: str, tool_mode: str) -> dict[str, Any]:
    """Five-pass attribution: how much does each component cost?

    Sends five free count_tokens calls, each adding one more component:
      1. system_only      — base system prompt, no tools, empty history, empty user msg
      2. + tools          — add the tool catalogue
      3. + user_msg       — add the user message
      4. + history        — add the full preloaded history (selection note, prior turns)
      5. full             — all of the above combined (verification, should match #4)

    The purpose is attribution: which component is responsible for a token saving after
    a change? Input tokens are all that matter here; output tokens and rounds come from a
    live run.

    The "full" count should equal the count_only() result. If it differs, the breakdown
    is counting something the real request does not, which would be a bug worth knowing.
    """
    assert api._client is not None
    catalogue = api._model_tools(question)
    if tool_mode == TOOL_MODE_PROMPTED_JSON:
        full_system = prompted_system_prompt(catalogue)
        full_tools = None
    else:
        full_system = SYSTEM_PROMPT
        full_tools = catalogue

    # The preloaded history (selection note, any prior turns). Excluding the user message.
    history_only = copy.deepcopy(api._history)

    # Full history with the user message appended.
    full_history = copy.deepcopy(api._history)
    append_user_note(full_history, question)

    empty: list = []
    user_only = [{"role": "user", "content": question}]

    def count(msgs, *, sys=SYSTEM_PROMPT, t=None):
        return api._client.count_tokens(msgs, tools=t, system=sys)

    c_system_only = count(empty, sys=SYSTEM_PROMPT)
    c_with_tools = count(empty, sys=full_system, t=full_tools)
    c_with_user = count(user_only, sys=full_system, t=full_tools)
    c_with_history = count(full_history, sys=full_system, t=full_tools)
    c_full = count(full_history, sys=full_system, t=full_tools)  # same as above, for verification

    return {
        "question": question,
        "ok": True,
        "breakdown": True,
        "tool_mode": tool_mode,
        "tool_catalogue_names": [t["name"] for t in catalogue],
        "tools_sent": len(catalogue),
        "system_chars": len(full_system),
        "components": {
            "system_only": c_system_only,
            "plus_tools": c_with_tools,
            "plus_user_msg": c_with_user,
            "plus_history": c_with_history,
            "full": c_full,
        },
        "deltas": {
            "tools": c_with_tools - c_system_only,
            "user_msg": c_with_user - c_with_tools,
            "history": c_with_history - c_with_user,
        },
    }


def audit_session(api: Api) -> list[str]:
    """Re-run app_smoke's stage-9 checks, so a token win cannot hide broken accounting.

    The session log's schema is fixed by the external cost comparison. A harness that
    reported a saving while the log that proves it had drifted would be worse than no
    harness.
    """
    data = api.reveal_session()["data"]
    problems: list[str] = []

    expected_keys = [
        "session_id", "started_at", "model", "circuit_file", "turns",
        "total_input_tokens", "total_output_tokens", "resolved",
    ]
    if list(data.keys()) != expected_keys:
        problems.append(f"session schema keys drifted: {list(data.keys())}")
    if any(
        t["input_tokens"] is None or t["output_tokens"] is None for t in data["turns"]
    ):
        problems.append("a turn has a null token count")
    if data["total_input_tokens"] != sum(t["input_tokens"] for t in data["turns"]):
        problems.append("total_input_tokens does not match the sum of the turns")
    if data["total_output_tokens"] != sum(t["output_tokens"] for t in data["turns"]):
        problems.append("total_output_tokens does not match the sum of the turns")
    assistant = [t for t in data["turns"] if t["role"] == "assistant"]
    if not assistant:
        problems.append("no assistant turn was logged")
    elif any(t["input_tokens"] <= 0 or t["output_tokens"] <= 0 for t in assistant):
        problems.append("an assistant turn has a zero token count")
    return problems


def report(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print()
        print(f"  question: {row['question']}")
        if not row.get("ok"):
            print(f"  FAILED: {row.get('error')}")
            continue
        if row.get("breakdown"):
            print(f"  mode:          {row['tool_mode']}")
            print(f"  system prompt: {row['system_chars']} chars")
            print(f"  tools sent:    {row['tools_sent']}")
            c = row["components"]
            d = row["deltas"]
            print()
            print("  Token attribution (cumulative / delta):")
            print(f"    system_only:  {c['system_only']:>6}")
            print(f"    + tools:      {c['plus_tools']:>6}  (+{d['tools']})")
            print(f"    + user_msg:   {c['plus_user_msg']:>6}  (+{d['user_msg']})")
            print(f"    + history:    {c['plus_history']:>6}  (+{d['history']})")
            print(f"    full:         {c['full']:>6}  (verification)")
            continue
        if row.get("count_only"):
            print(f"  mode:          {row['tool_mode']}")
            print(f"  system prompt: {row['system_chars']} chars")
            print(f"  tools sent:    {row['tools_sent']}")
            print(f"  INPUT TOKENS:  {row['input_tokens']}  (no inference run)")
            continue
        print(f"  rounds:        {row['rounds']}")
        print(
            f"  tool calls:    {row['tool_call_count']}"
            + (f"  ({', '.join(row['tool_calls'])})" if row["tool_calls"] else "")
        )
        if row["tool_result_chars"]:
            print(f"  result chars:  {row['tool_result_chars']}")
        print(f"  input tokens:  {row['input_tokens']}")
        print(f"  output tokens: {row['output_tokens']}")
        print(f"  TOTAL tokens:  {row['total_tokens']}")
        print(f"  answer:        {row['answer_words']} words")
        print(
            f"  correct:       {'yes' if row['correct'] else 'NO'}"
            + (f"  (missing: {', '.join(row['missing'])})" if row["missing"] else "")
        )
        print("  --- answer ---")
        for line in row["answer"].splitlines():
            print(f"  | {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit", required=True,
        help="Path to a .asc, absolute or relative to the repo root.",
    )
    parser.add_argument(
        "--question", action="append", required=True,
        help="Repeatable. Asked in order, in one session, so later questions see the "
             "earlier answers.",
    )
    parser.add_argument(
        "--expect", action="append", default=[],
        help="Repeatable. A substring every answer set must contain for the run to count "
             "as correct. Checked whitespace- and case-insensitively.",
    )
    parser.add_argument("--label", help="Write the rows to bench/<label>.json.")
    parser.add_argument("--tool-mode", choices=TOOL_MODES, help="Override the configured mode.")
    parser.add_argument(
        "--count-only", action="store_true",
        help="Price the request without running inference. Free.",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Five-pass token attribution: system / tools / user_msg / history. Free.",
    )
    parser.add_argument("--json", action="store_true", help="Print the rows as JSON too.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    source = Path(args.circuit)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.is_file():
        print(f"No such circuit: {source}", file=sys.stderr)
        return 2

    config = load_config()
    if args.tool_mode:
        config = dataclasses.replace(config, tool_mode=args.tool_mode)

    workdir = Path(tempfile.mkdtemp(prefix="spice_mcp_measure_"))
    circuit = workdir / source.name
    shutil.copy2(source, circuit)

    api = Api(config)
    free_mode = args.count_only or args.breakdown
    print(f"circuit:   {circuit}")
    print(f"model:     {config.model}")
    print(f"tool mode: {config.tool_mode}")
    print(f"mode:      {'breakdown (free)' if args.breakdown else 'count-only (free)' if args.count_only else 'live (spends tokens)'}")

    rows: list[dict[str, Any]] = []
    try:
        started = api.start()
        if not started.get("ok"):
            print(f"start failed: {started.get('error')}", file=sys.stderr)
            return 1
        print(f"tools:     {len(started['tools'])} exposed by the server")

        selection = api.select_circuit(str(circuit))
        if not selection.get("ok"):
            print(f"select failed: {selection.get('error')}", file=sys.stderr)
            return 1
        print(f"static:    {selection['checks'].get('summary', 'n/a')}")

        for question in args.question:
            if args.breakdown:
                rows.append(breakdown(api, question, config.tool_mode))
            elif args.count_only:
                rows.append(count_only(api, question, config.tool_mode))
            else:
                rows.append(measure(api, question, args.expect))

        report(rows)

        problems = audit_session(api) if not free_mode else []
        session_path = api.reveal_session()["path"]
    finally:
        api.shutdown()

    if not free_mode:
        total_in = sum(r.get("input_tokens", 0) for r in rows)
        total_out = sum(r.get("output_tokens", 0) for r in rows)
        print()
        print(f"SESSION TOTAL: {total_in} in / {total_out} out / {total_in + total_out}")
        print(f"session log:   {session_path}")
        print(
            "session log audit: "
            + ("clean" if not problems else "PROBLEMS -> " + "; ".join(problems))
        )

    payload = {
        "label": args.label,
        "circuit": source.name,
        "model": config.model,
        "tool_mode": config.tool_mode,
        "count_only": args.count_only,
        "breakdown": args.breakdown,
        "session_log": None if free_mode else session_path,
        "rows": rows,
    }
    if args.json:
        print()
        print(json.dumps(payload, indent=2))
    if args.label:
        BENCH_DIR.mkdir(exist_ok=True)
        out = BENCH_DIR / f"{args.label}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    failed = [r for r in rows if not r.get("ok") or r.get("missing")]
    if failed or (not free_mode and problems):
        print("\nFAILED: the run did not produce a correct, well-accounted answer.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
