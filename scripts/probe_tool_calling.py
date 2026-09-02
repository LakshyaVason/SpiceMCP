"""Probe how a TAMU AI Chat model can be made to call tools - in two separate ways.

OpenAI-compatible proxies sometimes drop function calling, or support it only on some
routes. That is not hypothetical here: `protected.Claude Opus 4.8` emits `tool_calls`
normally, and `us.anthropic.claude-opus-5` accepts the `tools` array and then answers as
though it had no tools at all. The app supports both through `SPICE_MCP_TOOL_MODE`, and
this script is how you tell which one a model needs.

    .venv\\Scripts\\activate
    python scripts/probe_tool_calling.py                                 # model from .env
    python scripts/probe_tool_calling.py protected.gpt-5                 # or explicit
    python scripts/probe_tool_calling.py us.anthropic.claude-opus-5 --prompted-json

**These are two different capabilities and the script keeps them distinguishable.** The
native probe is unchanged: it passes only when the model really does emit
`message.tool_calls`, and no amount of prompted-JSON support makes it pass. Run it without
a flag for native, with `--prompted-json` for the fallback.

Native mode (default) checks three things, because each is a separate way the proxy can
let us down:

  1. plain chat completion works at all, and `usage` is returned (the session log's
     token counts depend on it);
  2. the model emits `tool_calls` when given a `tools` array;
  3. a `role: "tool"` result is accepted back and produces a final answer.

Prompted-JSON mode (`--prompted-json`) checks the fallback end to end, using the real
protocol prompt and the real parser from `spice_mcp_app.llm` rather than a copy - so a
pass here is evidence about the shipping code path, not about this script:

  1. the same plain completion check;
  2. `tools` is *not* sent; the model is asked to reply with one protocol JSON object,
     and `parse_prompted_reply` accepts it as a tool call;
  3. a "TOOL RESULT" message is accepted back and produces `{"type":"answer"}` that uses
     the value only the tool could have supplied.

Exit code 0 means the probed mode works end to end for that model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spice_mcp_app.config import ConfigError, load_config  # noqa: E402
from spice_mcp_app.llm import (  # noqa: E402
    PROMPTED_STOP,
    ProtocolError,
    parse_prompted_reply,
    prompted_protocol_prompt,
    prompted_tool_result_message,
)

# A deliberately trivial stand-in for a real MCP tool. The answer is not derivable
# without calling it, so a model that answers without a tool call has ignored `tools`.
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_component_value",
        "description": (
            "Look up the value of a component in the currently open circuit. "
            "The only way to learn a component's value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Reference designator, e.g. R1 or C2.",
                }
            },
            "required": ["ref"],
        },
    },
}

PROBE_QUESTION = "What is the value of C1 in the circuit I have open?"
PROBE_TOOL_RESULT = '{"ref": "C1", "value": "100n"}'


def post_chat(cfg, payload: dict) -> dict:
    # "stream": false is mandatory, not a default. Omit it and the TAMU proxy replies
    # with text/event-stream *and* drops the usage block entirely - which would leave
    # the session log's token counts null. Verified 2026-08-29.
    response = requests.post(
        cfg.chat_completions_url,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        json={**payload, "stream": False},
        timeout=120,
    )
    if response.status_code >= 400:
        # The body is where the proxy explains itself; a bare status is not enough to
        # tell "no tool support" apart from "bad key".
        raise SystemExit(
            f"HTTP {response.status_code} from {cfg.chat_completions_url}\n"
            f"{response.text[:1500]}"
        )
    return response.json()


def show_usage(label: str, result: dict) -> bool:
    usage = result.get("usage") or {}
    if not usage:
        print(f"  {label}: usage MISSING - session token logging will not work")
        return False
    print(
        f"  {label}: prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')} "
        f"total={usage.get('total_tokens')}"
    )
    return usage.get("prompt_tokens") is not None


def probe_plain(cfg, model: str) -> bool:
    """Stage 1, shared by both modes: does the route answer at all, with usage?"""
    print("[1/3] plain chat completion")
    plain = post_chat(
        cfg,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 16,
        },
    )
    text = (plain["choices"][0]["message"].get("content") or "").strip()
    print(f"  reply: {text!r}")
    return show_usage("usage", plain)


def probe_prompted_json(cfg, model: str, usage_ok: bool) -> int:
    """The fallback protocol, exercised through the app's own prompt builder and parser.

    Note what is deliberately absent from both requests: a `tools` key. On a route that
    ignores `tools`, sending it would only make a pass ambiguous.
    """
    system = prompted_protocol_prompt([PROBE_TOOL])
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": PROBE_QUESTION},
    ]

    print("\n[2/3] prompted tool request")
    print(f"  protocol prompt: {len(system)} chars, no `tools` field sent")
    turn = post_chat(
        cfg, {"model": model, "messages": messages, "stop": PROMPTED_STOP}
    )
    content = turn["choices"][0]["message"].get("content") or ""
    print(f"  raw reply: {content.strip()[:300]!r}")
    show_usage("usage", turn)

    try:
        parsed = parse_prompted_reply(content, known_tools=[PROBE_TOOL["function"]["name"]])
    except ProtocolError as exc:
        print(f"  the reply is not a protocol message: {exc}")
        print("\nRESULT: prompted JSON tool calling FAILED for this model.")
        return 1

    if parsed["type"] != "tool_call":
        print(f"  the model answered without calling the tool: {parsed!r}")
        print(
            "\nRESULT: prompted JSON tool calling FAILED - it guessed instead of "
            "requesting the tool."
        )
        return 1
    print(f"  parsed: name={parsed['name']!r} arguments={parsed['arguments']!r}")

    # --- 3. feed a real result back through the framed message -----------------------
    print("\n[3/3] TOOL RESULT round trip")
    messages.append({"role": "assistant", "content": content})
    messages.append(prompted_tool_result_message(parsed["name"], PROBE_TOOL_RESULT))
    final = post_chat(
        cfg, {"model": model, "messages": messages, "stop": PROMPTED_STOP}
    )
    final_content = final["choices"][0]["message"].get("content") or ""
    print(f"  raw reply: {final_content.strip()[:300]!r}")
    show_usage("usage", final)

    try:
        answer = parse_prompted_reply(
            final_content, known_tools=[PROBE_TOOL["function"]["name"]]
        )
    except ProtocolError as exc:
        print(f"  the reply is not a protocol message: {exc}")
        print("\nRESULT: prompted JSON round trip FAILED at the result stage.")
        return 1

    print()
    if answer["type"] != "answer":
        print("RESULT: the model kept calling tools instead of answering. FAILED.")
        return 1
    if "100n" not in answer["text"].replace(" ", "") and "100" not in answer["text"]:
        print(
            "RESULT: the tool result was delivered but the answer does not use it. "
            "Inspect the reply above before trusting the loop."
        )
        return 1
    if not usage_ok:
        print("RESULT: the protocol works but `usage` is missing; token logging needs work.")
        return 1
    print(
        "RESULT: prompted JSON tool round trip works. Set "
        "SPICE_MCP_TOOL_MODE=prompted_json for this model."
    )
    return 0


def probe_native(cfg, model: str, usage_ok: bool) -> int:
    # --- 2. does it emit tool_calls? -------------------------------------------------
    print("\n[2/3] tool call request")
    messages: list[dict] = [{"role": "user", "content": PROBE_QUESTION}]
    tool_turn = post_chat(
        cfg,
        {
            "model": model,
            "messages": messages,
            "tools": [PROBE_TOOL],
            "tool_choice": "auto",
        },
    )
    message = tool_turn["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    show_usage("usage", tool_turn)

    if not tool_calls:
        print("  NO tool_calls returned.")
        print(f"  content was: {(message.get('content') or '')[:300]!r}")
        print(
            "\nRESULT: this model ignored `tools`. Native tool calling is NOT available "
            "on this route.\nRe-run with --prompted-json to check the fallback, and set "
            "SPICE_MCP_TOOL_MODE=prompted_json if it passes."
        )
        return 1

    call = tool_calls[0]
    print(f"  tool_calls: {len(tool_calls)}")
    print(f"  id:         {call.get('id')}")
    print(f"  name:       {call['function']['name']}")
    print(f"  arguments:  {call['function']['arguments']!r}")

    try:
        parsed_args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError as exc:
        print(f"\nRESULT: arguments are not valid JSON ({exc}).")
        return 1
    print(f"  parsed:     {parsed_args}")

    # --- 3. feed the result back ------------------------------------------------------
    print("\n[3/3] tool result round trip")
    messages.append(message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": PROBE_TOOL_RESULT,
        }
    )
    final = post_chat(cfg, {"model": model, "messages": messages, "tools": [PROBE_TOOL]})
    final_text = (final["choices"][0]["message"].get("content") or "").strip()
    print(f"  reply: {final_text[:300]!r}")
    show_usage("usage", final)

    round_trip_ok = "100n" in final_text.replace(" ", "") or "100" in final_text
    print()
    if round_trip_ok and usage_ok:
        print(
            "RESULT: full native tool-calling round trip works. Leave "
            "SPICE_MCP_TOOL_MODE at native for this model."
        )
        return 0
    if not round_trip_ok:
        print(
            "RESULT: the tool result was accepted but the model did not use it. "
            "Inspect the reply above before trusting the loop."
        )
        return 1
    print("RESULT: tool calling works but `usage` is missing; token logging needs work.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", help="model id; defaults to SPICE_MCP_MODEL")
    parser.add_argument(
        "--prompted-json",
        action="store_true",
        help="probe the prompted-JSON fallback instead of native `tools` support",
    )
    args = parser.parse_args()

    try:
        cfg = load_config()
    except ConfigError as exc:
        return int(bool(print(exc, file=sys.stderr))) or 1

    model = args.model or cfg.model
    mode = "prompted_json" if args.prompted_json else "native"
    print(f"Probing model: {model}")
    print(f"Mode:          {mode}")
    print(f"Endpoint:      {cfg.chat_completions_url}\n")

    usage_ok = probe_plain(cfg, model)
    if args.prompted_json:
        return probe_prompted_json(cfg, model, usage_ok)
    return probe_native(cfg, model, usage_ok)


if __name__ == "__main__":
    raise SystemExit(main())
