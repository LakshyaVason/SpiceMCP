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
6. **Preserve the session-log schema as specified** (see "Session log" below). The external
   cost comparison depends on it. No analysis UI in the app — it records, it does not analyse.
7. **Nothing touches the user's schematic before they approve it.** `patch_component_value`
   defaults to a preview, but the model can pass `apply=True` itself, so the real gate is
   `Api._tool_executor`, which refuses any apply not registered by `Api.apply_patch` (the
   UI-only path). Approvals are keyed on `(resolved path, ref, new value)` and are
   single-use. `tests/test_llm.py` is the regression test — it asserts the refused call
   never reaches the server.

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

### Explorer-launch facts, verified 2026-08-29

- **Under `pythonw.exe` with no console — which is what Explorer gives us — `sys.stdout`,
  `sys.stderr` and `sys.stdin` are ALL `None`.** `logging.StreamHandler(sys.stderr)` then
  builds a handler whose `.stream` is `None` and **silently discards every record**. Both
  `launch.py` and `spice_mcp_app/__main__.py` therefore guard with
  `if sys.stderr is not None else logging.NullHandler()`, and `launch.py` adds a
  `FileHandler` on `launch.log` as the primary sink. Don't "simplify" these back to
  `basicConfig(stream=...)`.
- **The MCP stdio handshake works fine under a console-less `pythonw` parent**, with
  `pythonw.exe` as the server interpreter too: probed to `handshake OK, 7 tools` plus a real
  `check_netlist_static` (which shells out to `LTspice -netlist`) returning findings. This
  was the feature's biggest risk — `sys.stdin`/`stdout` being `None` in the *parent* does not
  affect the pipes the SDK creates for the child. Without this the whole thing would have
  worked from a terminal and failed silently from Explorer.
- **A `SystemFileAssociations` verb lands in the Win11 *legacy* menu** ("Show more options"
  / Shift+right-click), not the compact one. The compact menu takes only packaged MSIX apps
  implementing `IExplorerCommand`; no registry setting promotes a classic verb into it.
  Documented as an expectation to confirm at install time — not verified on this machine yet.
- `HKCU\Software\Classes\.asc` is owned by the ProgID `Analog Devices Inc..LTspice_1`, and
  `SystemFileAssociations\.asc` did not exist in HKCU or HKCR. Hence the installer creates a
  clean subtree. Hanging the verb off LTspice's ProgID was the alternative and is rejected:
  that name is version-suffixed and would break when LTspice re-registers itself.
- `psutil.process_iter(['name'])` enumerates all 441 processes here **unelevated**, so
  `ltspice_is_running()` needs no `ctypes`/`CreateToolhelp32Snapshot` fallback.

### The app is allowed to import from the server

`spice_mcp_app.api` imports `ltspice_is_running` from `spice_mcp_server.ltspice` directly,
not over MCP. The invariant is one-directional: the **server** must never learn what an LLM
is. App→server is already the real dependency direction (the app spawns the server), and
`tests/conftest.py` imports the same module. `ltspice_is_running` is deliberately **not** an
MCP tool — it is a UX detail the model never needs, and an eighth tool schema would be paid
for in every request. `tests/test_write_conflict.py` asserts the tool set is still exactly
seven.

### TAMU proxy facts (verified 2026-08-29 via `scripts\probe_tool_calling.py`)

- **`"stream": false` is mandatory, not the default.** Omit it and the proxy answers with
  `Content-Type: text/event-stream` — `response.json()` raises `JSONDecodeError` on line 1 —
  **and the streamed form carries no `usage` block at all.** Silently null token counts would
  invalidate the whole cost comparison, so `TamuClient.complete` always sends it.
- **No `/v1` in the path.** `…/openai/v1/chat/completions` returns 403 "Direct API passthrough
  is disabled." The correct URL is `{base_url}/chat/completions`.
- **Tool calling works fully** on `protected.Claude Opus 4.8`: `tools` is accepted, the model
  emits OpenAI-shaped `tool_calls`, and a `role:"tool"` + `tool_call_id` reply closes the
  round trip. The plan's prompted-JSON fallback is **not needed** — don't build it.
- The model id contains a space (`protected.Claude Opus 4.8`). That is real, not a typo.
- **`SYMATTR Value2` is a trap.** `RCLP.asc`'s `V1` has both `Value` and `Value2` (`AC 0.7
  3000`). A `startswith("Value")` match patches the wrong line, so `asc.py` tokenises the
  attribute name.

### Known limitation

**Automated verification that the pywebview window *renders* is blocked.** `evaluate_js` from
a worker thread deadlocks (WebView2 requires UI-thread access), and a push-based probe — a
temp copy of `web/` calling back into a `ProbeApi` subclass — produced no output either.
The window does open and the bridge does work (JS calling `start()` reaches Python). In place
of a render test, `tests/test_app_wiring.py` cross-references UI↔Python statically: every
`pywebview.api.X` resolves to a real `Api` method, every id `app.js` looks up exists in
`index.html`. Visual confirmation needs the user's eyes. The WebView2 noise at launch
(`AllowExternalDrop`, `DefaultBackgroundColor`, `Failed to unregister class
Chrome_WidgetWin_0`) is pywebview probing optional properties — harmless.

## Commands

```bat
.venv\Scripts\activate
python -m pytest                        REM 209 tests, ~25s
python -m spice_mcp_app                 REM the desktop app; --folder fixtures --debug
python -m spice_mcp_app --file fixtures\wrong_value_lowpass.asc   REM one circuit, pre-checked
python -m spice_mcp_app.launch fixtures\wrong_value_lowpass.asc --no-ltspice
python -m spice_mcp_server              REM stdio server; sits waiting for a client
python scripts\install_context_menu.py  REM right-click verb; --status / --uninstall
python scripts\list_tamu_models.py      REM needs TAMU_API_KEY
python scripts\probe_tool_calling.py    REM 3-stage TAMU check; costs a few tokens
python scripts\app_smoke.py             REM headless end-to-end, no window; costs tokens
```

Prefix Python invocations with `PYTHONIOENCODING=utf-8` — LTspice output contains `Ω`, `µ`
and `°`, which crash a cp1252 console.

`.mcp.json` registers the server so it can be driven from Claude Code directly.

## State: Steps 0–8 plus the Explorer launcher; 209 tests pass

**Server half — `spice_mcp_server/`, seven tools**, all verified over a real MCP stdio
handshake. It still knows nothing about LLMs.

| tool | notes |
| --- | --- |
| `read_netlist` | structured components/nets/directives; ExpressPCB detection |
| `check_netlist_static` | nine checks, no simulation |
| `run_simulation` | staged into `%TEMP%`; `succeeded` from log text only |
| `read_sim_log` | re-read a `.log` from a previous run or the GUI |
| `patch_component_value` | **preview by default**; `apply=True` writes byte-faithfully |
| `diff_netlist` | before/after by ref; value change vs. rewire reported separately |
| `export_netlist` | SPICE netlist to a chosen path, CRLF, refuses to overwrite |

Modules: `models.py` (pydantic schemas), `netlist.py`, `checks.py`, `logparse.py`,
`ltspice.py` (staged invocation), `asc.py` (byte-preserving patcher), `diff.py`.

**App half — `spice_mcp_app/`.** The only half that knows what an LLM is.
`config.py` (env/`.env`, redacted logging), `session.py` (the spec'd log, atomic write per
turn), `llm.py` (TAMU client, MCP→OpenAI schema translation, bounded agent loop),
`mcp_client.py` (stdio client holding one server subprocess open), `api.py` (JS bridge +
**the approval gate**), `web/` (single-window UI), `__main__.py`, `launch.py` (the Explorer
entry point). Plus `spice_mcp_launch.py` at the repo root — a one-line shim so the registry
command can be an absolute script path, since the registry cannot set a working directory.

### Launching from Explorer

Right-click a `.asc` → "Debug with SPICE MCP" opens LTspice **and** the client with that
circuit selected and its static checks already run. Three decisions the user made, which are
requirements and not preferences:

1. **The trigger is the right-click verb.** Not a combined shortcut, not a background
   watcher, and not the client opening LTspice. So the two apps come up together *only* when
   you start from the schematic — opening LTspice from the Start menu summons nothing.
2. **The circuit is bound once, at launch.** Do not add polling of LTspice window titles.
3. **Applying a fix while LTspice runs warns, then writes** — never blocks. The user asked
   for the fix and the file is theirs; the warning (File ▸ Revert) exists so the fix cannot
   quietly disappear when they next save from the GUI.

Two corrections this feature forced, both hard-constraint-2 adjacent, because the launcher
starts with the cwd set to the user's circuit folder:

- `config.py` now anchors a **relative** `SPICE_MCP_SESSIONS_DIR` to `REPO_ROOT`. Resolved
  against the cwd it would have scattered session logs into the user's source directory.
- `mcp_client.py` passed `env=` as a **replacement** dict, so `LTSPICE_EXE` never reached the
  server despite `.env.example` documenting that it would. It now merges over `os.environ`
  minus `_LLM_ONLY_ENV` — the key in particular has no business in a process that knows
  nothing about LLMs. `tests/test_config.py` pins both.

### Session log
`./sessions/<uuid>.json`, flushed after every turn: `session_id`, `started_at`, `model`,
`circuit_file`, `turns[]` (each `role`, `text`, optional `tool_calls`, `input_tokens`,
`output_tokens`), `total_input_tokens`, `total_output_tokens`, `resolved`. `Turn.from_usage`
is the **one** place `prompt_tokens`/`completion_tokens` are mapped to the spec's names, and
it warns when an assistant turn arrives without usage.

### Testing philosophy — keep this
`tests/test_checks.py` asserts the **exact set** of checks each fixture produces. Equality
rather than membership is deliberate: it makes the suite a false-positive guard. A check
that cries wolf trains both the user and the model to ignore the output, so a new check
misfiring on the known-good circuit must fail the suite loudly.

Log-parser test samples are **real captured LTspice output**, never invented. The messages
are inconsistent enough that plausible-looking fabricated samples would test the wrong thing.

The suite is offline and free: `tests/test_llm.py` drives the agent loop with a `FakeClient`
returning canned OpenAI-shaped responses. Live-model checks live in `scripts/`, not in
`pytest`, so running the tests never costs tokens or depends on the proxy being up.

### Fixtures pull in different directions on purpose
`no_dc_path.asc` is caught by the static check but **simulates fine** (exit 0, solves `.op`
"by inspection"). `source_conflict.asc` is the mirror: **statically clean**, genuinely fails
to simulate. `wrong_value_lowpass.asc` passes both and is out of spec by 10× — only circuit
*reasoning* catches it. `RCLP.asc` also simulates "successfully" despite `R1` dangling and
`Vin` driving nothing. Neither pass subsumes the other; that's the argument for two stages.

## Acceptance bar — met, except two items needing the user

`scripts\app_smoke.py` drives the real `Api` headlessly through nine stages on a temp copy of
`wrong_value_lowpass.asc` and passed end to end: diagnosis → **approval-gate probe** (the
model is asked to write unilaterally; the gate refuses and the file stays byte-identical) →
approval → byte fidelity → re-simulate → model confirmation → session-log audit
(16 turns, 41 996 in / 1 572 out, schema keys exactly as specified, no null counts, totals
matching).

- ✅ Byte fidelity on `patch_component_value` — only the intended line changes, encoding and
  line endings unchanged, and the patched file survives an LTspice `-netlist` round trip and
  re-simulates (`tests/test_asc.py`, incl. `@needs_ltspice`).
- ✅ Token counts complete and non-null on every turn (`tests/test_session.py` + the smoke
  script's stage 9 audit).
- ⏳ **Needs the user:** open a patched `.asc` in the LTspice **GUI**. Only the `-netlist`/`-b`
  proxy is automated, and the bar names the GUI specifically.
- ⏳ **Needs the user:** look at the app window (`python -m spice_mcp_app --folder fixtures`).
  See "Known limitation" above for why this can't be automated here.

**Both are closed by one pass of the Explorer-launcher walkthrough**, which is why that
feature was the natural next move rather than a detour:

1. `python scripts\install_context_menu.py` — should print four keys.
2. Right-click `fixtures\wrong_value_lowpass.asc` → **Show more options** → "Debug with SPICE
   MCP". (If it appears in the compact menu directly, that is better than expected — fix the
   README's Win11 caveat.)
3. LTspice opens the schematic; the client window opens; the sidebar lists `fixtures`;
   `wrong_value_lowpass.asc` is highlighted with its static checks already rendered, no
   clicking. ← closes the app-window item.
4. Ask what is wrong, let it propose the `C1` fix, press **Apply** → the amber File ▸ Revert
   warning appears and the auto re-simulation still runs.
5. In LTspice: **File ▸ Revert** → `C1` shows the new value in the GUI. ← closes the GUI item.
6. `python scripts\install_context_menu.py --uninstall` — the entry is gone.

Automated up to that point: 209 tests, plus a headless launcher run verified to reach window
creation with `get_initial_folder()` returning the resolved folder and circuit, cwd back at
the repo root, and nothing written beside the fixture.

## Useful symbol geometry (read from LTspice's `lib.zip`)

Pin offsets for placing parts in a `.asc`: `res`/`ind`/`fixedind` = (16,16) & (16,96);
`cap`/`polcap` = (16,0) & (16,64); `voltage` = (0,16) & (0,96); `current` = (0,0) & (0,80).
R90 maps local (x,y)→(−y,x), so a `res` at (X,Y) R90 has pins at (X−16,Y+16) and (X−96,Y+16).

## Working style the user expects

Verify against the real tool rather than reasoning from documentation — nearly every
correction above came from running something and reading the output. When a fixture or
comment turns out to claim something false, fix the claim rather than working around it.
Report honestly what failed.
