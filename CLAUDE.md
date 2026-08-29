# SPICE MCP client — working notes

An MCP server + desktop app that lets an engineer debug LTspice circuits by talking to an
LLM about the **netlist as text** instead of pasting screenshots. Cheaper in input tokens,
and the model gets structure instead of pixels.

Milestone 1 of a broader effort to build MCP servers for EDA tools. LTspice is the open
sandbox for validating the approach before Cadence-class tools (blocked on PDK/NDA
clearance). Read `README.md` for the user-facing picture; the approved build plan is at
`C:\Users\Expertician\.claude\plans\smooth-forging-pie.md`.

**v1 is done when:** a broken circuit is diagnosed in plain English, a fix is proposed as a
reviewable before/after diff, the fix is written back to the `.asc` so it stays usable in
the LTspice GUI, the sim is re-run to confirm, and every turn's token usage is logged for an
**external** cost comparison against the old screenshot workflow.

## Hard constraints — do not relax these

1. **No secrets in source.** The TAMU key loads from `TAMU_API_KEY` in a git-ignored `.env`
   or the environment. Never hardcode, never log it. `.env.example` documents the shape.
2. **Never write into the user's source directory.** LTspice writes `-netlist` and `-b`
   output *next to the input file*, so every invocation stages the input into
   `%TEMP%\spice_mcp_work\` first. This is not hypothetical: an ad-hoc shell command during
   development destroyed the user's hand-exported `RCLP.net` and left three stray
   artifacts. `tests/test_ltspice.py` is the regression test. **If you run LTspice by hand,
   go through `spice_mcp_server.ltspice`, never a raw `subprocess` call from the repo root.**
3. **Use the venv**, never global installs: `.venv\Scripts\activate` (Git Bash:
   `source .venv/Scripts/activate`). Python 3.13.9.
4. **Windows only** for v1. Non-Windows raises `UnsupportedPlatform` rather than guessing at
   a Wine setup that can't be tested.
5. **Byte fidelity on `.asc` files.** `RCLP.asc` is bare-LF/no-BOM; `RCLP.net` is CRLF.
   `.gitattributes` exempts `*.asc/*.asy/*.net/*.cir` from `text=auto` so git never rewrites
   schematic bytes. Patching must preserve encoding and line endings exactly.
6. **Preserve the session-log schema as specified** (see Step 5 below). The external cost
   comparison depends on it. No analysis UI in the app — it records, it does not analyse.

## Decisions already made with the user — don't relitigate

- Name: "SPICE MCP client". Repo stays `SpiceMCP`.
- Python-only desktop app (pywebview/PySide). No Electron, no Node toolchain.
- `spicelib` is an accepted dependency; no from-scratch parser.
- LLM is the **TAMU AI Chat** proxy (`https://chat-api.tamu.ai/openai`), OpenAI-compatible —
  not Anthropic directly.
- Two-process MCP split over local stdio. The server **must never learn what an LLM is** —
  that's what makes it reusable for the later EDA-tool work.

## Environment facts, verified — not assumed

Several of these contradict the original brief and most online tutorials. They were each
confirmed empirically; please don't "fix" the code back toward the wrong version.

- **`mcp` SDK is 2.1.1**, a rewrite of v1. Entry point is `from mcp.server import MCPServer`
  — **not `FastMCP`**. `mcp.run()` with no argument is stdio. Keep `mcp>=2.1` pinned.
- **The return type annotation IS the output schema.** Pydantic fields must be annotated on
  the class body; a class whose attributes are only assigned in `__init__` produces no
  schema, with no warning, and the model silently sees a bare `repr`.
- **stdout is the MCP wire.** Any `print()` in the server can corrupt the protocol. All
  diagnostics go through `logging`, which the SDK flushes to stderr.
- **Raise `ToolError` for anticipated failures** —
  `from mcp.server.mcpserver.exceptions import ToolError` (only importable from that deep
  path in 2.1). Any other exception discards your message and the model receives only
  "Error executing tool <name>". Our error text is often the actual diagnosis, so this
  matters.
- MCP Python objects use **snake_case**: `input_schema` / `output_schema`, not camelCase.
  `client.list_tools()` returns a result object — use `.tools`.
- **LTspice is 26.0.2** at `%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe` (per-user), not
  the `Program Files\LTC\LTspiceXVII` path the brief assumed. `LTSPICE_EXE` overrides.
- **`-I<path>` must be the LAST argument, after the filename, with no space.** Placed
  earlier, LTspice hangs forever instead of erroring. Verified three ways.
- **`RCLP.net` is an ExpressPCB netlist, not SPICE.** That's what the GUI's "Export Netlist"
  and `-PCBnetlist` write. Real SPICE netlists come from `-netlist`. The parser detects the
  format and falls back to converting the sibling `.asc` when one exists.
- **The exit code lies, and so does the `.raw` file.** A floating current source logs
  `ERROR:` and exits **0**; a missing ground exits 1. The over-defined-matrix failure still
  wrote a complete-looking `.raw`. `succeeded` is derived from **log text only**.
- **Log messages have no common shape.** Of five distinct failures, only two carry an
  `ERROR:`/`WARNING:` prefix; undefined models arrive as `path(3): Undefined model "x"`.
  A grep for `ERROR` misses the two worst failures.
- `spicelib` caveats: `AscEditor` wants designators *without* the SPICE prefix (`U1`),
  `SpiceEditor` *with* (`XU1`); library-sourced components raise on write.

## Commands

```bat
.venv\Scripts\activate
python -m pytest                        REM 103 tests, ~22s
python -m spice_mcp_server              REM stdio server; sits waiting for a client
python scripts\list_tamu_models.py      REM needs TAMU_API_KEY
```

Prefix Python invocations with `PYTHONIOENCODING=utf-8` — LTspice output contains `Ω`, `µ`
and `°`, which crash a cp1252 console.

`.mcp.json` registers the server so it can be driven from Claude Code directly.

## State: Steps 0–4 complete

`spice_mcp_server/` exposes **four working tools**, all verified over a real MCP stdio
handshake: `read_netlist`, `check_netlist_static`, `run_simulation`, `read_sim_log`.
Modules: `models.py` (pydantic schemas), `netlist.py` (parser + ExpressPCB detection),
`checks.py` (nine static checks), `logparse.py`, `ltspice.py` (staged invocation).

### Testing philosophy — keep this
`tests/test_checks.py` asserts the **exact set** of checks each fixture produces. Equality
rather than membership is deliberate: it makes the suite a false-positive guard. A check
that cries wolf trains both the user and the model to ignore the output, so a new check
misfiring on the known-good circuit must fail the suite loudly.

Log-parser test samples are **real captured LTspice output**, never invented. The messages
are inconsistent enough that plausible-looking fabricated samples would test the wrong thing.

### Fixtures pull in different directions on purpose
`no_dc_path.asc` is caught by the static check but **simulates fine** (exit 0, solves `.op`
"by inspection"). `source_conflict.asc` is the mirror: **statically clean**, genuinely fails
to simulate. `wrong_value_lowpass.asc` passes both and is out of spec by 10× — only circuit
*reasoning* catches it. `RCLP.asc` also simulates "successfully" despite `R1` dangling and
`Vin` driving nothing. Neither pass subsumes the other; that's the argument for two stages.

## Remaining work

**Blocked on the user:** they must run `scripts\list_tamu_models.py` and report the model
list before the first live LLM call. Everything else can proceed.

- **Step 5 — app shell.** pywebview window, folder picker, chat panel. One hardcoded TAMU
  call to prove token logging works before the agent loop exists. Session log per debug
  session to `./sessions/<uuid>.json`, plus an export button:
  `session_id`, `started_at`, `model`, `circuit_file`, `turns[]` (each `role`, `text`,
  optional `tool_calls`, `input_tokens`, `output_tokens`), `total_input_tokens`,
  `total_output_tokens`, `resolved`. TAMU returns OpenAI-shaped
  `usage.prompt_tokens`/`completion_tokens` — **map** these to `input_tokens`/`output_tokens`
  so the specified schema is preserved.
- **Step 6 — MCP client + agent loop.** `spice_mcp_app/mcp_client.py` (stdio) and `llm.py`
  (MCP→OpenAI tool-schema translation, loop on `tool_calls`, `role:"tool"` results, record
  usage per turn). **Probe TAMU tool-calling support with a one-tool smoke test first** —
  it is unverified and is the project's main open risk. If the model rejects `tools`, the
  fallback is a prompted JSON tool protocol.
- **Step 7 — write-back + diff UI.** `patch_component_value(asc_path, ref, new_value)`:
  find the `SYMATTR Value` line belonging to the matching `SYMATTR InstName <ref>` and
  rewrite **only** that line. Hand-rolled, not `AscEditor.save_netlist`, which reformats.
  Plus `diff_netlist` and `export_netlist`. Nothing touches disk before user approval.
- **Step 8 — end-to-end** on `wrong_value_lowpass.asc`: chat diagnosis → accepted fix →
  file written → re-simulate → confirm fixed.

### Acceptance bar still to meet
- Byte-fidelity test on `patch_component_value`: only the intended line changes, encoding
  and line endings unchanged, and the patched file **still opens in the LTspice GUI**.
- `sessions/*.json` inspected for complete, non-null token counts on every turn. The
  external cost comparison depends entirely on that file being complete.

## Useful symbol geometry (read from LTspice's `lib.zip`)

Pin offsets for placing parts in a `.asc`: `res`/`ind`/`fixedind` = (16,16) & (16,96);
`cap`/`polcap` = (16,0) & (16,64); `voltage` = (0,16) & (0,96); `current` = (0,0) & (0,80).
R90 maps local (x,y)→(−y,x), so a `res` at (X,Y) R90 has pins at (X−16,Y+16) and (X−96,Y+16).

## Working style the user expects

Verify against the real tool rather than reasoning from documentation — nearly every
correction above came from running something and reading the output. When a fixture or
comment turns out to claim something false, fix the claim rather than working around it.
Report honestly what failed.
