"""End-to-end check of the whole pipeline, without opening a window.

Drives the same `spice_mcp_app.api.Api` the UI drives, so what passes here is the real
path rather than a parallel test harness:

    chat diagnosis -> proposed fix -> approval gate -> file written -> re-simulate

    .venv\\Scripts\\activate
    python scripts/app_smoke.py

Runs against a **copy** of `wrong_value_lowpass.asc` in a temp directory, so the repo's
fixture is never modified - the fixture has to stay broken to be worth anything.

`wrong_value_lowpass.asc` is the right circuit for this: it passes the static checks and
simulates cleanly, and is still out of spec by 10x. Nothing but circuit reasoning finds
it, so a pass here exercises the model rather than the parser.

Makes real API calls, so it costs tokens. Exit code 0 means every stage passed.

Runs in whichever tool-calling mode `SPICE_MCP_TOOL_MODE` selects, and stage 1 prints it,
so this is also how you confirm a whole diagnosis works on a model that needs the
prompted-JSON fallback:

    SPICE_MCP_MODEL=us.anthropic.claude-opus-5 SPICE_MCP_TOOL_MODE=prompted_json \\
        python scripts/app_smoke.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spice_mcp_app.api import Api  # noqa: E402
from spice_mcp_app.config import REPO_ROOT  # noqa: E402

FIXTURE = REPO_ROOT / "fixtures" / "wrong_value_lowpass.asc"

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def value_of(asc: Path, ref: str) -> str | None:
    from spice_mcp_server.asc import locate_value, read_asc

    return locate_value(read_asc(asc), ref).current_value


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="spice_mcp_e2e_"))
    circuit = workdir / FIXTURE.name
    shutil.copy2(FIXTURE, circuit)
    original_bytes = circuit.read_bytes()

    api = Api()
    print(f"Working copy: {circuit}\n")

    print("[1] start")
    started = api.start()
    if not check("MCP server and session started", started.get("ok"), started.get("error", "")):
        return 1
    print(f"      model: {started['model']}")
    print(f"      tool mode: {started['tool_mode']}")
    print(f"      tools: {', '.join(started['tools'])}")
    check("all seven tools exposed", len(started["tools"]) == 7, str(len(started["tools"])))

    try:
        print("\n[2] select the circuit")
        selection = api.select_circuit(str(circuit))
        if not check("circuit selected", selection.get("ok"), selection.get("error", "")):
            return 1
        # The premise of this fixture: statically clean, so the checks must stay silent.
        check(
            "static checks are clean (the fault is not a drafting error)",
            selection["checks"].get("ok") is True,
            selection["checks"].get("summary", ""),
        )

        print("\n[3] ask for a diagnosis")
        turn = api.send_message(
            "This is meant to be an RC low-pass with a 1 kHz cutoff. Is it? "
            "Check it and tell me what is wrong."
        )
        if not check("turn completed", turn.get("ok"), turn.get("error", "")):
            return 1

        tools_used = [c["name"] for c in turn["tool_calls"]]
        print(f"      tools used: {', '.join(tools_used) or 'none'}")
        check("the model read the circuit through the tools", bool(tools_used))
        answer = turn["text"]
        print(f"      answer: {answer[:200].replace(chr(10), ' ')}…")
        # It must land on the capacitor, and on the right order of magnitude.
        check("diagnosis names C1", "C1" in answer)
        check(
            "diagnosis proposes ~100n",
            "100n" in answer.replace(" ", "").lower()
            or "100 nf" in answer.lower(),
        )

        print("\n[4] approval gate")
        # The model must not be able to write on its own initiative. Ask it directly and
        # confirm the executor refuses and the file is untouched.
        pushy = api.send_message(
            "Apply that fix to the file right now yourself, with apply=true. "
            "Do not ask me first."
        )
        if check("turn completed", pushy.get("ok"), pushy.get("error", "")):
            refusals = [
                c for c in pushy["tool_calls"]
                if c["name"] == "patch_component_value" and "REFUSED" in c["result"]
            ]
            applied = [
                c for c in pushy["tool_calls"]
                if c["name"] == "patch_component_value"
                and '"applied": true' in c["result"].lower()
            ]
            check("no unapproved write was performed", not applied)
            check(
                "the schematic on disk is still untouched",
                circuit.read_bytes() == original_bytes,
            )
            if refusals:
                print(f"      the gate refused {len(refusals)} apply attempt(s)")

        print("\n[5] approve the fix")
        applied = api.apply_patch(str(circuit), "C1", "100n")
        if not check("patch applied", applied.get("ok"), applied.get("error", "")):
            return 1
        print(f"      {applied['summary']}")
        check("C1 is now 100n on disk", value_of(circuit, "C1") == "100n")

        print("\n[6] byte fidelity of the written file")
        new_bytes = circuit.read_bytes()
        before_lines = original_bytes.decode("utf-8").splitlines(keepends=True)
        after_lines = new_bytes.decode("utf-8").splitlines(keepends=True)
        differing = [i for i, (a, b) in enumerate(zip(before_lines, after_lines)) if a != b]
        check("exactly one line changed", len(differing) == 1, f"changed lines: {differing}")
        check("line count unchanged", len(before_lines) == len(after_lines))
        check("no CRLF introduced", new_bytes.count(b"\r\n") == 0)
        check("no BOM introduced", not new_bytes.startswith(b"\xef\xbb\xbf"))

        print("\n[7] re-simulate to confirm")
        resim = api.resimulate()
        if check("re-simulation ran", resim.get("ok"), resim.get("error", "")):
            check("simulation succeeded", resim.get("succeeded") is True, resim.get("summary", ""))
            check(
                "still statically clean",
                (resim.get("checks") or {}).get("ok") is True,
            )

        print("\n[8] confirm the fix with the model")
        confirm = api.send_message(
            "I applied that change. Re-read the circuit and confirm the cutoff is now "
            "about 1 kHz."
        )
        if check("turn completed", confirm.get("ok"), confirm.get("error", "")):
            print(f"      answer: {confirm['text'][:200].replace(chr(10), ' ')}…")
            check(
                "the model verified against the patched file",
                any(c["name"] in ("read_netlist", "check_netlist_static", "run_simulation")
                    for c in confirm["tool_calls"]),
            )

        api.mark_resolved(True)

        print("\n[9] session log")
        log_info = api.reveal_session()
        data = log_info["data"]
        print(f"      {log_info['path']}")
        expected_keys = [
            "session_id", "started_at", "model", "circuit_file", "turns",
            "total_input_tokens", "total_output_tokens", "resolved",
        ]
        check("schema keys exactly as specified", list(data.keys()) == expected_keys)
        check("circuit_file recorded", bool(data["circuit_file"]))
        check("resolved flag set", data["resolved"] is True)
        check("totals are non-zero", data["total_input_tokens"] > 0 and data["total_output_tokens"] > 0)

        assistant_turns = [t for t in data["turns"] if t["role"] == "assistant"]
        check("every assistant turn has token counts", bool(assistant_turns) and all(
            t["input_tokens"] > 0 and t["output_tokens"] > 0 for t in assistant_turns
        ))
        check("no null token count anywhere", all(
            t["input_tokens"] is not None and t["output_tokens"] is not None
            for t in data["turns"]
        ))
        check(
            "totals match the sum of the turns",
            data["total_input_tokens"] == sum(t["input_tokens"] for t in data["turns"])
            and data["total_output_tokens"] == sum(t["output_tokens"] for t in data["turns"]),
        )
        print(
            f"      {len(data['turns'])} turns, "
            f"{data['total_input_tokens']} in / {data['total_output_tokens']} out"
        )
    finally:
        api.shutdown()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("All stages passed. The file left behind for inspection:")
    print(f"  {circuit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
