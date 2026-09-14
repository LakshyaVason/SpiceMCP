"""Probe tool calling on the TAMU AI Gateway's Anthropic Messages endpoint.

The gateway exposes two request paths, and *the path decides the wire format* while the
model id only names the gateway's own backend provider:

    POST {base}/v1/chat/completions   OpenAI-shaped
    POST {base}/v1/messages           Anthropic-shaped   <- what this probes

`us.anthropic.claude-opus-5` therefore means "the gateway reaches this model via Bedrock
internally". It does **not** mean this client should hold AWS credentials, and nothing in
this script touches AWS, boto3 or a region.

The open question this exists to answer: `/v1/chat/completions` was already observed
returning no OpenAI `tool_calls` for this model, which is why `prompted_json` mode exists
and was verified live. Does `/v1/messages` do better and return real Anthropic `tool_use`
blocks? Until this passes, `SPICE_MCP_TOOL_MODE` stays where it is.

    .venv\\Scripts\\activate
    python scripts/probe_tool_calling.py                          # model from .env
    python scripts/probe_tool_calling.py us.anthropic.claude-opus-5
    python scripts/probe_tool_calling.py --prompted-json          # the fallback mode
    python scripts/probe_tool_calling.py --effort low             # + the effort parameter

Native mode (the default) runs three stages, each a distinct way the route can fail:

  1. a plain completion works at all, and `usage` carries non-zero token counts;
  2. given Anthropic-format `tools`, the model emits a real `tool_use` **content block** -
     not a description of a call in its visible text;
  3. an Anthropic `tool_result` block is accepted back, and the final answer carries the
     value only that tool could have supplied.

`--prompted-json` probes the other mode using the app's own prompt builder, stop sequence
and parser - not copies - so a pass is evidence about the shipping code path.

`--effort LEVEL` adds a pre-flight stage for `output_config={"effort": LEVEL}`, the only
parameter that reaches the *thinking* half of the output cost - `usage.output_tokens`
includes thinking tokens, so prompt wording alone cannot touch it. The gateway is a routing
layer and need not forward every Messages parameter, so this asks whether it does, and then
measures the same prompt with and without it. When accepted, the level is applied to the
remaining stages too, so a mode is never certified under settings the app would not use.
**Never `temperature`** (absent from the SDK, and a 400 on Opus 5) and never `thinking`.

**The native probe's meaning is fixed.** Stage 2 passes only on a genuine `tool_use`
block. Prose that merely looks like a tool call is a FAIL, and the bar is not lowered so
that some model prints PASS. Stage 3's value (`PROBE_VALUE`) is deliberately an odd
number no one would volunteer, so an answer composed from memory cannot pass either.

Exit 0 means the probed mode works end to end over the gateway.

Authentication reuses TAMU_API_KEY, exactly as the app's working transport did: it is sent
as an `Authorization: Bearer` header by the SDK's `auth_token=` parameter. The token's
value is never printed, logged, or included in any error text this script emits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spice_mcp_app.config import EFFORT_LEVELS  # noqa: E402
from spice_mcp_app.llm import (  # noqa: E402
    PROMPTED_STOP,
    ProtocolError,
    parse_prompted_reply,
    prompted_protocol_prompt,
    prompted_tool_result_message,
)

# The URL and model default are deliberately *not* read through `load_config()`, so that a
# configuration bug cannot masquerade as a gateway failure - this script has to be able to
# say "the route is fine, your .env is not". `EFFORT_LEVELS` above is imported rather than
# re-typed because it is a list of literals with no behaviour, and a second copy would
# drift.
DEFAULT_BASE_URL = "https://gateway.api.tamu.ai"
DEFAULT_MODEL = "us.anthropic.claude-opus-5"

# A trivial stand-in for a real MCP tool. The answer is not derivable without calling it,
# so a model that answers straight away has ignored `tools`. Note the flat Anthropic shape
# - name/description/input_schema, with no OpenAI `function` wrapper.
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

# An odd value on purpose. `100n` - the previous probe's choice - is the most guessable
# capacitance there is, so an answer built from memory rather than from the tool result
# would have passed. Nothing in this repository contains this string.
PROBE_VALUE = "3.917n"
PROBE_TOOL_RESULT = f'{{"ref": "C1", "value": "{PROBE_VALUE}"}}'

# Enough for an answer plus adaptive thinking, which is on by default and counts here.
MAX_TOKENS = 2048

# A create() closure bound to one client and model, so neither probe can quietly exercise
# a different model than the banner printed.
Create = Callable[..., Any]


def show_usage(response: object) -> bool:
    """Print the usage block and report whether it is actually usable.

    The Messages API uses the session log's own names (input_tokens/output_tokens). A null
    or zero count here would invalidate the cost comparison the log exists for, so this is
    a pass/fail check rather than a diagnostic print.
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


def explain_and_exit(exc: Exception, model: str, base_url: str) -> int:
    """Turn the SDK's exception into the thing the user actually has to go do.

    Every branch names the gateway. None of them mentions AWS: the client has no AWS
    identity to fix, and a Bedrock-flavoured diagnostic here would send the reader off to
    a console they have no access to.
    """
    if isinstance(exc, anthropic.NotFoundError):
        print(f"\nFAIL: {base_url} did not route {model!r}.", file=sys.stderr)
        print(
            "  Either the gateway does not offer that model id, or the account is not\n"
            "  entitled to it. Try another id from the gateway's model list, or pass one\n"
            "  on the command line. A 404 here is about the gateway's routing table, not\n"
            "  about anything on this machine.",
            file=sys.stderr,
        )
    elif isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        print(f"\nFAIL: {base_url} rejected the credential.", file=sys.stderr)
        print(
            "  TAMU_API_KEY was sent as an Authorization: Bearer header. Check that the\n"
            "  token in .env is current and entitled to this model. (The value is not\n"
            "  printed here by design.)",
            file=sys.stderr,
        )
    elif isinstance(exc, anthropic.APIConnectionError):
        print(f"\nFAIL: could not reach {base_url}.", file=sys.stderr)
        print(
            "  A network or VPN issue rather than a configuration one - the request\n"
            "  never got an HTTP status back.",
            file=sys.stderr,
        )
    else:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1


def uses_the_result(text: str) -> bool:
    """Did the answer carry the value only the tool could have supplied?

    Matched on the digits, so `3.917n`, `3.917 nF` and `3.917nF` all count, while a
    plausible-sounding invented value does not.
    """
    return "3.917" in text.replace(" ", "")


# Something with a little arithmetic in it, so there is thinking for `effort` to act on. A
# prompt with nothing to work out would show no difference between levels and prove nothing.
EFFORT_QUESTION = (
    "A 10k and a 15k resistor form a divider across a 5 V source. What is the voltage at "
    "the midpoint? Give the number and nothing else."
)


def probe_effort(create: Create, level: str, model: str, base_url: str) -> bool:
    """Does the gateway forward `output_config`, and does it change anything?

    Returns whether the parameter is usable. A rejection is **not** a failed probe: unset is
    the app's default and a gateway that refuses the parameter is a fact to record, not a
    broken route. The app's own fallback is the same - warn once, continue without it.

    The second, unparameterised request is what makes the answer worth having. A gateway can
    accept an unknown field and drop it on the floor, in which case the request "succeeds"
    and the setting does nothing; only the token counts distinguish that from a real effect.
    """
    print(f"[effort] output_config={{'effort': '{level}'}} pre-flight")
    try:
        with_effort = create(
            messages=[{"role": "user", "content": EFFORT_QUESTION}],
            output_config={"effort": level},
        )
    except anthropic.BadRequestError as exc:
        text = str(exc).lower()
        if "output_config" in text or "effort" in text:
            print(f"  REJECTED: {exc}")
            print(
                "  -> the gateway does not forward output_config. Leave SPICE_MCP_EFFORT\n"
                "     unset; the app logs one warning and continues without it, so this\n"
                "     costs nothing but the parameter is not available on this route."
            )
            return False
        # A 400 about something else is a real failure and must not be reported as
        # "effort unsupported" - that would send the reader after the wrong thing.
        raise

    without = create(messages=[{"role": "user", "content": EFFORT_QUESTION}])
    hi = getattr(without.usage, "output_tokens", 0) or 0
    lo = getattr(with_effort.usage, "output_tokens", 0) or 0
    print(f"  accepted.    output tokens: {lo} with effort={level}, {hi} unset")
    print(f"  answer:      {response_text(with_effort)[:80]!r}")
    if hi and lo:
        print(f"  -> {(hi - lo) / hi:+.0%} on this one prompt (a single sample, not a mean)")
    if lo >= hi:
        print(
            "  -> accepted but no saving here. One prompt is not evidence of no effect;\n"
            "     it is evidence not to assume one. Measure a real turn before relying on it."
        )
    print()
    return True


def probe_plain(create: Create) -> tuple[str, bool]:
    """Stage 1, shared by both modes: does the model answer at all, with usage?

    Returns (text, usage_ok). An empty text is the caller's cue to stop.
    """
    print("[1/3] plain completion over /v1/messages")
    plain = create(messages=[{"role": "user", "content": "Reply with just: ok"}])
    text = response_text(plain)
    print(f"  stop_reason: {plain.stop_reason}")
    print(f"  reply:       {text!r}")
    return text, show_usage(plain)


def probe_native(create: Create, model: str, base_url: str, usage_ok: bool) -> int:
    """Stages 2 and 3 of native tool use: a real tool_use block, then a tool_result."""
    # --- 2. does it emit a real tool_use block? ------------------------------------
    print("\n[2/3] tool_use emission (Anthropic-format `tools` sent)")
    messages: list[dict[str, object]] = [{"role": "user", "content": PROBE_QUESTION}]
    try:
        tool_turn = create(messages=messages, tools=[PROBE_TOOL])
    except Exception as exc:
        return explain_and_exit(exc, model, base_url)

    print(f"  stop_reason: {tool_turn.stop_reason}")
    print(f"  blocks:      {[b.type for b in tool_turn.content]}")
    uses = [b for b in tool_turn.content if b.type == "tool_use"]
    if tool_turn.stop_reason != "tool_use" or not uses:
        print(
            f"\nFAIL: no tool_use block. The model said:\n  {response_text(tool_turn)!r}",
            file=sys.stderr,
        )
        print(
            "  This is the same shape of failure already seen on /v1/chat/completions:\n"
            "  the route accepts `tools` and then answers as though it had none. If the\n"
            "  text above reads like a tool call written out in prose, the tool channel\n"
            "  is not open on this route - and no amount of prompting makes this stage\n"
            "  pass, which is the point of keeping it strict.\n"
            "  Re-run with --prompted-json to confirm the fallback still works, and\n"
            "  leave SPICE_MCP_TOOL_MODE=prompted_json in place.",
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
    # The assistant turn is replayed verbatim, which is what carries any thinking block
    # back alongside the tool_use from the same turn.
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
        return explain_and_exit(exc, model, base_url)

    final_text = response_text(final)
    print(f"  stop_reason: {final.stop_reason}")
    print(f"  answer:      {final_text!r}")
    usage_ok = show_usage(final) and usage_ok

    round_trip_ok = final.stop_reason == "end_turn" and uses_the_result(final_text)

    print()
    if round_trip_ok and usage_ok:
        print(
            f"PASS: {model} does native Anthropic tool use through {base_url},\n"
            f"      with usable token counts. SPICE_MCP_TOOL_MODE=native is viable."
        )
        return 0
    if not round_trip_ok:
        print(
            f"FAIL: the tool_result was accepted but the answer did not use "
            f"{PROBE_VALUE}.\n"
            "      A tool channel that delivers results the model then ignores is not\n"
            "      usable for diagnosis.",
            file=sys.stderr,
        )
    if not usage_ok:
        print(
            "FAIL: token counts are missing or zero, so the session log would\n"
            "      under-report cost - which is the reason the log exists.",
            file=sys.stderr,
        )
    return 1


def probe_prompted_json(create: Create, model: str, base_url: str, usage_ok: bool) -> int:
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
        return explain_and_exit(exc, model, base_url)

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
        return explain_and_exit(exc, model, base_url)

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
            f"FAIL: the tool result was delivered but the answer does not use "
            f"{PROBE_VALUE}.\n"
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
        f"PASS: {model} does prompted-JSON tool calling through {base_url}.\n"
        "      SPICE_MCP_TOOL_MODE=prompted_json is the working mode for this route."
    )
    return 0


def _normalize_base_url(raw: str) -> str:
    """Trim a trailing /v1, /openai or /api off a pasted gateway URL.

    The SDK appends `/v1/messages` itself, so a base URL that already ends in `/v1`
    produces `/v1/v1/messages` and a 404 that reads like a missing model.
    """
    value = (raw or "").strip().rstrip("/")
    if not value:
        return DEFAULT_BASE_URL
    for suffix in ("/v1", "/openai", "/api"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
    return value or DEFAULT_BASE_URL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", help="model id; defaults to SPICE_MCP_MODEL")
    parser.add_argument(
        "--prompted-json",
        action="store_true",
        help="probe the prompted-JSON tool mode instead of native `tools` support",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        help="also probe output_config={'effort': ...}, and apply it to the other stages "
             "if the gateway forwards it",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    token = (os.environ.get("TAMU_API_KEY") or "").strip()
    if not token:
        print(
            "TAMU_API_KEY is not set, so the gateway cannot be reached.\n\n"
            f"Put it in {REPO_ROOT / '.env'} (git-ignored) as:\n"
            "  TAMU_API_KEY=<your gateway token>\n"
            "It is sent as an Authorization: Bearer header and is never logged.",
            file=sys.stderr,
        )
        return 1

    base_url = _normalize_base_url(os.environ.get("SPICE_MCP_BASE_URL") or "")
    model = args.model or (os.environ.get("SPICE_MCP_MODEL") or "").strip() or DEFAULT_MODEL
    mode = "prompted_json" if args.prompted_json else "native"
    configured = (os.environ.get("SPICE_MCP_TOOL_MODE") or "").strip() or "(unset)"

    print(f"Gateway:       {base_url}/v1/messages")
    print(f"Model:         {model}")
    print("Credential:    TAMU_API_KEY, sent as Authorization: Bearer")
    print(f"Probing mode:  {mode}   (.env currently says {configured})")
    print(f"Effort:        {args.effort or 'not probed (the app default is unset)'}\n")

    # auth_token= produces `Authorization: Bearer <token>` and omits x-api-key, which is
    # the same scheme the app's previously-working TAMU transport used.
    client = Anthropic(base_url=base_url, auth_token=token, timeout=180.0)

    # Set only once the pre-flight has shown the gateway forwards it, so a rejection cannot
    # take the tool-calling stages down with it.
    effort: str | None = None

    def create(**kwargs: Any) -> Any:
        # Never pass `thinking`. Disabling it on Opus 5 makes the model occasionally write
        # a tool call into its visible text instead of a tool_use block - which is the
        # failure this script exists to detect, so we must not induce it ourselves.
        if effort and "output_config" not in kwargs:
            kwargs["output_config"] = {"effort": effort}
        return client.messages.create(model=model, max_tokens=MAX_TOKENS, **kwargs)

    if args.effort:
        try:
            if probe_effort(create, args.effort, model, base_url):
                effort = args.effort
        except Exception as exc:
            return explain_and_exit(exc, model, base_url)

    try:
        text, usage_ok = probe_plain(create)
    except Exception as exc:
        return explain_and_exit(exc, model, base_url)
    if not text:
        print("\nFAIL: the model returned no text at all.", file=sys.stderr)
        return 1

    if args.prompted_json:
        return probe_prompted_json(create, model, base_url, usage_ok)
    return probe_native(create, model, base_url, usage_ok)


if __name__ == "__main__":
    raise SystemExit(main())
