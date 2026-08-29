"""Probe whether a TAMU AI Chat model supports OpenAI-style tool calling.

This is the project's main open risk: OpenAI-compatible proxies sometimes drop function
calling, or support it only on some models. The agent loop in spice_mcp_app/llm.py is
built on it, so it is worth confirming before building on top rather than after.

    .venv\\Scripts\\activate
    python scripts/probe_tool_calling.py                    # model from .env
    python scripts/probe_tool_calling.py protected.gpt-5    # or an explicit one

Three things are checked, in order, because each is a separate way the proxy can let us
down:

  1. plain chat completion works at all, and `usage` is returned (the session log's
     token counts depend on it);
  2. the model emits `tool_calls` when given a `tools` array;
  3. a `role: "tool"` result is accepted back and produces a final answer.

Exit code 0 means the full round trip works and the agent loop can proceed as planned.
Non-zero means the fallback (a prompted JSON tool protocol) is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spice_mcp_app.config import ConfigError, load_config  # noqa: E402

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


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        return int(bool(print(exc, file=sys.stderr))) or 1

    model = sys.argv[1] if len(sys.argv) > 1 else cfg.model
    print(f"Probing model: {model}")
    print(f"Endpoint:      {cfg.chat_completions_url}\n")

    # --- 1. plain completion + usage -------------------------------------------------
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
    usage_ok = show_usage("usage", plain)

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
            "\nRESULT: this model ignored `tools`. Use the prompted-JSON fallback, or "
            "try another model id from scripts/list_tamu_models.py."
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
        print("RESULT: full tool-calling round trip works. Agent loop can proceed.")
        return 0
    if not round_trip_ok:
        print(
            "RESULT: the tool result was accepted but the model did not use it. "
            "Inspect the reply above before trusting the loop."
        )
        return 1
    print("RESULT: tool calling works but `usage` is missing; token logging needs work.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
