"""Probe whether Bedrock will serve the configured model, with working tool use.

Native tool use is a documented first-class feature of the Messages API, so the old
question this script asked - "does the proxy honour `tools` at all?" - is no longer the
risk. What can still go wrong on Bedrock is all environmental, and all of it fails in
ways that are easy to misread:

  * the inference profile is not enabled in this region, or the account lacks access;
  * credentials do not resolve, or resolve to the wrong account;
  * `usage` is absent, which would silently null the session log's token counts.

    .venv\\Scripts\\activate
    python scripts/probe_tool_calling.py                            # model from .env
    python scripts/probe_tool_calling.py global.anthropic.claude-opus-5   # or an explicit one
    python scripts/probe_tool_calling.py global.anthropic.claude-opus-5 --prompted-json

Native mode (the default) has three stages, in order, because each is a separate way
this can let us down:

  1. a plain completion works at all, and `usage` carries non-zero counts;
  2. the model emits a real `tool_use` content block when given `tools`, rather than
     narrating the call in its visible text (the exact bug this migration fixed);
  3. a `tool_result` block is accepted back and produces a final answer.

`--prompted-json` probes the other tool mode instead - the one selected by
`SPICE_MCP_TOOL_MODE=prompted_json`, where the catalogue goes in the system prompt and
the model replies with one JSON object. It uses the app's own prompt builder, stop
sequence and parser rather than copies, so a pass is evidence about the shipping code
path. Stage 1 is the same plain completion; then:

  2. `tools` is *not* sent, and `parse_prompted_reply` accepts the reply as a tool call;
  3. a "TOOL RESULT" user message is accepted back and produces `{"type":"answer"}`
     carrying the value only the tool could have supplied.

**The two are different capabilities and this script keeps them distinguishable.** The
native probe passes only when a real `tool_use` block arrives; no amount of
prompted-JSON support makes it pass, and it is not relaxed to suit any model.

Exit code 0 means the probed mode works end to end and the agent loop can proceed.
Non-zero prints what to change - usually a region or a model-access grant in the Bedrock
console.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import anthropic
from anthropic import AnthropicBedrock

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
# Note the flat shape - name/description/input_schema, no function wrapper.
PROBE_TOOL = {
    "name": "get_component_value",
    "description": (
        "Look up the value of a component in the currently open circuit. "
        "The only way to learn a component's value."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Reference designator, e.g. R1 or C2.",
            }
        },
        "required": ["ref"],
    },
}

PROBE_QUESTION = "What is the value of C1 in the circuit I have open?"
PROBE_TOOL_RESULT = '{"ref": "C1", "value": "100n"}'

# Enough for an answer plus adaptive thinking, which is on by default and counts here.
MAX_TOKENS = 2048

# A create() closure bound to one client and model. Both probes take one, so neither can
# quietly probe a different model than the banner printed.
Create = Callable[..., Any]


def show_usage(response: object) -> bool:
    """Print the usage block and report whether it is actually usable.

    Bedrock uses the session log's own names (input_tokens/output_tokens) and has no
    total_tokens. A null or zero count here would invalidate the cost comparison, so this
    is a pass/fail check, not a diagnostic print.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        print("  usage:      MISSING - the session log would record zeros")
        return False
    prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "output_tokens", None)
    print(f"  usage:      input={prompt} output={completion}")
    if not prompt or not completion:
        print("  -> incomplete; per-turn token counts would be wrong")
        return False
    return True


def response_text(response: object) -> str:
    """Visible text only - thinking blocks are skipped, as in the app."""
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def explain_and_exit(exc: Exception, model: str, region: str) -> int:
    """Turn the SDK's exception into the thing the user actually has to go do."""
    if isinstance(exc, (anthropic.NotFoundError, anthropic.PermissionDeniedError)):
        print(f"\nFAIL: Bedrock will not serve {model} in {region}.", file=sys.stderr)
        print(
            "  Either the inference profile does not exist in that region or this\n"
            "  account has not been granted access to the model.\n"
            "    * list what is available:  python scripts/list_bedrock_models.py\n"
            "    * enable it:               Bedrock console -> Model access\n"
            "    * or point elsewhere:      SPICE_MCP_AWS_REGION=<region> in .env",
            file=sys.stderr,
        )
    elif isinstance(exc, anthropic.AuthenticationError):
        print("\nFAIL: Bedrock rejected the credentials.", file=sys.stderr)
        print(
            "  The chain found something, but it was not accepted. Check the account,\n"
            "  and that the principal has bedrock:InvokeModel on this model.",
            file=sys.stderr,
        )
    else:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1


def uses_the_result(text: str) -> bool:
    """Did the answer actually carry the value only the tool could have supplied?"""
    return "100n" in text.replace(" ", "") or "100" in text


def probe_plain(create: Create) -> tuple[str, bool]:
    """Stage 1, shared by both modes: does the model answer at all, with usage?

    Returns (text, usage_ok). An empty text is the caller's cue to stop.
    """
    print("[1/3] plain completion")
    plain = create(messages=[{"role": "user", "content": "Reply with just: ok"}])
    text = response_text(plain)
    print(f"  stop_reason: {plain.stop_reason}")
    print(f"  reply:       {text!r}")
    return text, show_usage(plain)


def probe_native(create: Create, model: str, region: str, usage_ok: bool) -> int:
    """Stages 2 and 3 of native tool use: a real tool_use block, then a tool_result."""
    # --- 2. does it emit a real tool_use block? ------------------------------------
    print("\n[2/3] tool_use emission")
    messages: list[dict[str, object]] = [{"role": "user", "content": PROBE_QUESTION}]
    try:
        tool_turn = create(messages=messages, tools=[PROBE_TOOL])
    except Exception as exc:
        return explain_and_exit(exc, model, region)

    print(f"  stop_reason: {tool_turn.stop_reason}")
    print(f"  blocks:      {[b.type for b in tool_turn.content]}")
    uses = [b for b in tool_turn.content if b.type == "tool_use"]
    if tool_turn.stop_reason != "tool_use" or not uses:
        print(
            f"\nFAIL: no tool_use block. The model said:\n  {response_text(tool_turn)!r}",
            file=sys.stderr,
        )
        print(
            "  If that text looks like a tool call written out in prose, the tool\n"
            "  channel is not open - check that `tools` is being sent and that\n"
            "  thinking has not been disabled.\n"
            "  Re-run with --prompted-json to see whether the fallback mode works for\n"
            "  this model, and set SPICE_MCP_TOOL_MODE=prompted_json if it does.",
            file=sys.stderr,
        )
        return 1

    call = uses[0]
    print(f"  name:        {call.name}")
    print(f"  input:       {call.input!r}")
    # input arrives already parsed - there is no JSON string to decode, and so no
    # malformed-arguments failure mode to guard against.
    if not isinstance(call.input, dict) or "ref" not in call.input:
        print(
            f"\nFAIL: input is not an object carrying `ref`: {call.input!r}",
            file=sys.stderr,
        )
        return 1

    # --- 3. round trip a tool_result ----------------------------------------------
    print("\n[3/3] tool_result round trip")
    messages.append({"role": "assistant", "content": tool_turn.content})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": PROBE_TOOL_RESULT,
                }
            ],
        }
    )
    try:
        final = create(messages=messages, tools=[PROBE_TOOL])
    except Exception as exc:
        return explain_and_exit(exc, model, region)

    final_text = response_text(final)
    print(f"  stop_reason: {final.stop_reason}")
    print(f"  answer:      {final_text!r}")
    usage_ok = show_usage(final) and usage_ok

    round_trip_ok = final.stop_reason == "end_turn" and uses_the_result(final_text)

    print()
    if round_trip_ok and usage_ok:
        print(f"PASS: {model} does native tool use on Bedrock, with usable token counts.")
        return 0
    if not round_trip_ok:
        print(
            "FAIL: the tool_result was accepted but the answer did not use it.",
            file=sys.stderr,
        )
    if not usage_ok:
        print(
            "FAIL: token counts are missing or zero, so the session log would\n"
            "      under-report cost - which is the reason the log exists.",
            file=sys.stderr,
        )
    return 1


def probe_prompted_json(create: Create, model: str, region: str, usage_ok: bool) -> int:
    """The fallback protocol, exercised through the app's own prompt builder and parser.

    Note what is deliberately absent from both requests: a `tools` argument. Sending it
    would make a pass ambiguous - the model might be answering off the native channel.
    The system prompt is the protocol alone rather than `prompted_system_prompt`, which
    would also carry the app's SPICE instructions; this probe's tool is not a SPICE tool.
    """
    known = [PROBE_TOOL["name"]]
    system = prompted_protocol_prompt([PROBE_TOOL])
    messages: list[dict[str, object]] = [{"role": "user", "content": PROBE_QUESTION}]

    print("\n[2/3] prompted tool request")
    print(f"  system:      {len(system)} chars of protocol, `tools` NOT sent")
    try:
        turn = create(messages=messages, system=system, stop_sequences=PROMPTED_STOP)
    except Exception as exc:
        return explain_and_exit(exc, model, region)

    text = response_text(turn)
    print(f"  stop_reason: {turn.stop_reason}")
    print(f"  raw reply:   {text[:300]!r}")
    show_usage(turn)
    if turn.stop_reason == "max_tokens":
        print(
            "\nFAIL: the reply was truncated before the JSON object closed. The app\n"
            "      recovers by re-asking; a probe should not have to.",
            file=sys.stderr,
        )
        return 1

    try:
        parsed = parse_prompted_reply(text, known_tools=known)
    except ProtocolError as exc:
        print(f"\nFAIL: the reply is not a protocol message: {exc}", file=sys.stderr)
        return 1

    if parsed["type"] != "tool_call":
        print(
            f"\nFAIL: the model answered without calling the tool: {parsed!r}\n"
            "      It guessed rather than requesting the only source of the value.",
            file=sys.stderr,
        )
        return 1
    print(f"  parsed:      name={parsed['name']!r} arguments={parsed['arguments']!r}")

    # --- 3. feed a real result back through the framed message ---------------------
    print("\n[3/3] TOOL RESULT round trip")
    # The assistant turn is its visible text, exactly as the loop replays it: prompted
    # mode has no tool_use_id, so the JSON object itself is the record of the request.
    messages.append({"role": "assistant", "content": text})
    messages.append(prompted_tool_result_message(parsed["name"], PROBE_TOOL_RESULT))
    try:
        final = create(messages=messages, system=system, stop_sequences=PROMPTED_STOP)
    except Exception as exc:
        return explain_and_exit(exc, model, region)

    final_text = response_text(final)
    print(f"  stop_reason: {final.stop_reason}")
    print(f"  raw reply:   {final_text[:300]!r}")
    usage_ok = show_usage(final) and usage_ok

    try:
        answer = parse_prompted_reply(final_text, known_tools=known)
    except ProtocolError as exc:
        print(f"\nFAIL: the reply is not a protocol message: {exc}", file=sys.stderr)
        return 1

    print()
    if answer["type"] != "answer":
        print(
            "FAIL: the model kept calling tools instead of answering.", file=sys.stderr
        )
        return 1
    if not uses_the_result(answer["text"]):
        print(
            "FAIL: the tool result was delivered but the answer does not use it.\n"
            "      Inspect the reply above before trusting the loop.",
            file=sys.stderr,
        )
        return 1
    if not usage_ok:
        print(
            "FAIL: the protocol works, but token counts are missing or zero - so the\n"
            "      session log would under-report cost.",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: {model} does prompted-JSON tool calling on Bedrock. Set\n"
        "      SPICE_MCP_TOOL_MODE=prompted_json to use it for this model."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", help="model id; defaults to SPICE_MCP_MODEL")
    parser.add_argument(
        "--prompted-json",
        action="store_true",
        help="probe the prompted-JSON tool mode instead of native `tools` support",
    )
    args = parser.parse_args()

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    model = args.model or cfg.model
    mode = "prompted_json" if args.prompted_json else "native"

    print(f"Model:         {model}")
    print(f"Region:        {cfg.aws_region}")
    print(f"Credentials:   {cfg.credentials_source}")
    print(f"Tool mode:     {mode}\n")

    client = AnthropicBedrock(aws_region=cfg.aws_region)

    def create(**kwargs: Any) -> Any:
        # Never pass `thinking`. Disabling it on Opus 5 makes the model occasionally write
        # a tool call into its visible text instead of a tool_use block - which is the
        # failure this script exists to detect, so we must not induce it ourselves.
        return client.messages.create(model=model, max_tokens=MAX_TOKENS, **kwargs)

    try:
        text, usage_ok = probe_plain(create)
    except Exception as exc:
        return explain_and_exit(exc, model, cfg.aws_region)
    if not text:
        print("\nFAIL: the model returned no text at all.", file=sys.stderr)
        return 1

    if args.prompted_json:
        return probe_prompted_json(create, model, cfg.aws_region, usage_ok)
    return probe_native(create, model, cfg.aws_region, usage_ok)


if __name__ == "__main__":
    raise SystemExit(main())
