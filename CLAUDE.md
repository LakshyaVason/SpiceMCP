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

1. **No secrets in source — and none in `Config` either.** Auth is one bearer token,
   `TAMU_API_KEY`, and `Config` deliberately does not hold it: `GatewayClient.__init__`
   reads it from the environment at the moment it builds the SDK client and keeps it only
   inside the `Anthropic` instance. `Config` carries `model`, `base_url`, a
   `credentials_source` **label**, `sessions_dir` and `tool_mode`, and nothing else — so
   there is nothing for a stray debug print or a `repr()` in a traceback to leak, by
   construction rather than by filtering. The label is `TAMU_API_KEY (<n> chars)`: enough to
   distinguish "set" from "truncated paste", and deliberately **not** a last-four
   fingerprint — four characters of a live token are still four characters of a live token.
   `tests/test_config.py` sweeps `redacted()` and `repr()` with a **sliding 4-character
   window** over a fake token, differenced against a no-token baseline so a coincidental
   match in the hostname or repo path cannot be reported as a leak.
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
- LLM is **Claude reached through the TAMU AI Gateway** (`https://gateway.api.tamu.ai`),
  via `anthropic.Anthropic(base_url=…, auth_token=…)` and the native Messages API. The
  gateway owns authentication *and* billing; it routes to a model provider on its own side.
  The architecture is:
  `app → TAMU gateway /v1/messages → us.anthropic.claude-opus-5 → Bedrock, TAMU-side`.
  **The user has no AWS account, no AWS credentials, no region and no Bedrock console
  access, and the app must never require any of them.** `AnthropicBedrock` is therefore
  wrong here and must not be reintroduced; nor may boto3 return as a dependency. A previous
  version of this file claimed the gateway was "retired 2026-09-01" — **that was false**,
  and it is what sent an entire round of work into a direct-Bedrock rewrite that could not
  possibly run on this machine.
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

### Gateway / Messages API facts

**The endpoint matters more than the model.** The gateway exposes both
`POST /v1/messages` (Anthropic-shaped) and `POST /v1/chat/completions` (OpenAI-shaped), and
the *path* decides the wire format while the model id only decides which backend the gateway
routes to. The app used the OpenAI path until 2026-09-01, and the failure that forced the
move is worth keeping in mind because it is silent: given an OpenAI `tools` array a Claude
model **narrates the tool call as prose in its visible text** — the turn "succeeds", no call
runs, and nothing raises. `sessions/beda7b77-….json` is the evidence (no `tool_calls` on any
turn, assistant text containing raw `antml:invoke` syntax and an invented netlist). A session
log with no `tool_calls` key is the symptom to look for.

**Verified live 2026-09-02:** native `tool_use` works on `/v1/messages` with
`us.anthropic.claude-opus-5` — `stop_reason: "tool_use"`, content blocks
`['text', 'tool_use']`, and a final answer built from the value the `tool_result` supplied.
`prompted_json` passes through the same route too. So the old failure was never the model
refusing to call tools; it was the endpoint.

- **`max_tokens` is required on every request.** There is no default. `MAX_TOKENS = 16000`,
  sized to leave room for thinking tokens as well as the answer.
- **Never pass `thinking`.** Adaptive thinking is on by default on Opus 5, and
  `thinking: {"type": "disabled"}` makes it occasionally write a tool call into visible text
  instead of a `tool_use` block — the exact failure above. `budget_tokens` is rejected with a
  400 on Opus 5; don't reintroduce it either.
- **Thinking blocks come back in `content` and must be handed back unchanged** alongside a
  `tool_use` from the same turn. The loop appends `response.content` verbatim, which covers
  it; `_response_text` filters them out of what the user sees.
- **All `tool_result` blocks from one round go back in a single user message**, keyed by
  `tool_use_id`. The model may emit several `tool_use` blocks at once and splitting their
  results across messages is rejected.
- **Consecutive same-role turns are rejected**, which OpenAI tolerated. `api.py` appends
  user-role notes ("the user opened this circuit", "the user approved your fix") and then the
  user's question appends another — hence `llm.py:append_user_note`, which folds a note into
  the pending user turn instead of adding a second one.
- **`tool_use.input` arrives already parsed.** There is no `json.loads`, and so no
  malformed-arguments failure mode — but `llm.py` still guards that it is a `Mapping`, because
  the approval gate reads `arguments.get("apply")` and on a non-dict that check cannot fire:
  the gate would fail *open*.
- **There is no system *role*.** The system prompt is a top-level `system=` parameter, sent on
  every request. A `{"role": "system"}` message in the history is a 400.
- **`SPICE_MCP_MODEL` names what the *gateway* routes to**, not anything on this machine.
  `us.anthropic.claude-opus-5` is billed at a **10% premium** over
  `global.anthropic.claude-opus-5`; say so in the cost write-up so the number is not read as
  base pricing. Spend is the gateway's to report: `GET /v1/tamus/billing` on the same token.
- **`SPICE_MCP_BASE_URL` must not carry a version segment.** The SDK appends `/v1/messages`
  itself, so a pasted `…/v1` yields `/v1/v1/messages` — which 404s, and a 404 from the
  gateway reads like a missing model, putting the diagnosis a long way from the cause.
  `config.py:_normalize_base_url` trims a trailing `/v1`, `/openai` or `/api`.
- The gateway does **not** offer server-side web search/fetch, code execution, the
  Batches/Files APIs, or Agent Skills. Messages, tool use, adaptive thinking and token
  counting all work. None of the missing pieces are used here.
- **`SYMATTR Value2` is a trap.** `RCLP.asc`'s `V1` has both `Value` and `Value2` (`AC 0.7
  3000`). A `startswith("Value")` match patches the wrong line, so `asc.py` tokenises the
  attribute name.

### Two tool-calling modes — `SPICE_MCP_TOOL_MODE`

`native` (the default, and verified live on the gateway's `/v1/messages`) sends the MCP tools
in the `tools` parameter and reads back `tool_use` blocks. `prompted_json` puts the catalogue
in the system prompt, has the model reply with **one JSON object**, and frames results as
`TOOL RESULT` user messages. It exists because a route can accept `tools` and then answer as
though it had none — which is what the *same* gateway's `/v1/chat/completions` does to
`us.anthropic.claude-opus-5`, silently. Both live in `llm.py`; `run_agent_turn` dispatches on
the config value.

- **The mode is explicit configuration, never inferred from a model name.** A typo raises
  `ConfigError` rather than falling back to native: a silent fallback is precisely the failure
  the setting exists to catch. The observation that produced this setting was about an
  *endpoint*, not a model family — don't restate it as "`us.anthropic` profiles ignore
  `tools`", because on `/v1/messages` that same profile honours them.
- **The approval gate covers both modes.** Both converge on `llm.py:_execute_tool`, which
  calls the executor `api.py` supplies, so `prompted_json` buys the model no extra reach.
  `tests/test_llm.py::test_no_tool_mode_can_write_without_approval` is parametrized over both.
- **The parser never interprets prose.** `parse_prompted_reply` `raw_decode`s from index 0, so
  the reply must *begin* with the object; one wrapping ``` fence is tolerated and nothing else.
  Unknown tool names, non-object `arguments` and wrong `type` values all raise `ProtocolError`,
  which becomes a correction message to the model — never an execution. Regex-scanning text
  for `antml:invoke` markup was explicitly ruled out by the user; do not reintroduce it.
- A malformed reply costs a re-ask, bounded by `MAX_PROTOCOL_CORRECTIONS = 2`. The correction
  goes through `append_user_note`, not a second user message — the Messages API rejects
  consecutive same-role turns, and a `TOOL RESULT` message may already be pending.
- `stop_sequences=PROMPTED_STOP` (`["TOOL RESULT"]`) stops the model role-playing the tool's
  reply. It has been seen fabricating a result and an answer quoting it; anything after the
  first JSON object is discarded with a warning, which is the safe direction.
- **`prompted_system_prompt` is rebuilt per request** because there is nowhere in the history
  to keep it. `tests/test_llm.py` asserts the protocol text is present on *every* call and
  that no `{"role": "system"}` message is ever appended.
- `SPICE_MCP_TOOL_MODE` is in `mcp_client._LLM_ONLY_ENV`: how the model is asked to call tools
  is not the server's business.
- `scripts\probe_tool_calling.py --prompted-json` probes the fallback using the app's real
  prompt builder, stop sequence and parser, so a pass is evidence about shipping code. **The
  native probe's meaning is fixed** — it passes only on a real `tool_use` block, and is not to
  be relaxed so that some model prints PASS.

### Known limitation

**Automated verification that the pywebview window *renders* is blocked.** `evaluate_js` from
a worker thread deadlocks (WebView2 requires UI-thread access), and a push-based probe — a
temp copy of `web/` calling back into a `ProbeApi` subclass — produced no output either.
The window does open and the bridge does work (JS calling `start()` reaches Python). In place
of a render test, `tests/test_app_wiring.py` cross-references UI↔Python statically: every
`pywebview.api.X` resolves to a real `Api` method, every id `app.js` looks up exists in
`index.html`. Visual confirmation needs the user's eyes.

A previous version of this section called the WebView2 noise at launch (`AllowExternalDrop`,
`DefaultBackgroundColor`, `Failed to unregister class Chrome_WidgetWin_0`) "pywebview probing
optional properties — harmless." **That was wrong, and it cost a day.** It was the bridge walk
described below chewing through the native object graph, and it hung every launch. A clean
launch now prints none of it — if you see that noise again, something is being walked that
should not be.

### The pywebview bridge walk — verified 2026-08-30

**pywebview 6.2.1 builds `window.pywebview.api` by recursively walking every *public*
attribute of the `js_api` object** (`webview/util.py:180-211`, in `inject_pywebview`). Public
methods become JS functions; public *non-callables that have a `__module__`* get descended
into. Names starting with `_` are skipped, and objects can opt out with
`_serializable = False`.

- **`Api` must keep every public attribute a `str`, `bool` or `None`.** Anything richer goes
  behind an underscore. `tests/test_app_wiring.py` enforces this by replicating pywebview's
  own recursion predicate over `dir(Api())` and asserting the walked set is **empty** — so a
  new rich public attribute fails the suite instead of failing at launch, in a webview, with
  no traceback anywhere Python can see.
- **The window reference is `Api._window`, set via `Api._attach_window()`.** It was
  `api.window`, and that single missing underscore hung the app on every launch. pywebview
  guards its own `DOM`, `EventContainer` and `state` with `_serializable = False`, but
  **`Window.native` is unguarded** (`webview/window.py:182`) — under the winforms backend
  that is the .NET `Form`, so the walk went
  `Form.AccessibilityObject` → `.Bounds` (a `System.Drawing.Rectangle`) → `.Empty` (a static
  `Rectangle`) → `.Empty` → … to the recursion limit. pythonnet returns a fresh wrapper on
  every `getattr`, so pywebview's `id(obj)` dedup in `exposed_objects` never fires.
- **Why that is fatal rather than noisy:** the walk sits *between* injecting the pywebview
  scaffolding and running `finish.js`, and `finish.js` is what dispatches `pywebviewready`
  (`webview/js/finish.js:9`). `web/app.js` boots the whole UI from that event. So the window
  opened, painted nothing useful, never called `start()`, and had to be killed from Task
  Manager. **A window that opens and does nothing is this bug; look at the bridge, not at the
  MCP server.**
- `web/app.js` now sets a 30 s watchdog that banners "The Python bridge did not initialise"
  if `pywebviewready` never arrives. It only helps when the bridge alone is broken — a walk
  that blocks the UI thread outright leaves nothing able to paint.

## Commands

```bat
.venv\Scripts\activate
python -m pytest                        REM 288 tests, ~42s
python -m spice_mcp_app                 REM the desktop app; --folder fixtures --debug
python -m spice_mcp_app --file fixtures\wrong_value_lowpass.asc   REM one circuit, pre-checked
python -m spice_mcp_app.launch fixtures\wrong_value_lowpass.asc --no-ltspice
python -m spice_mcp_server              REM stdio server; sits waiting for a client
python scripts\install_context_menu.py  REM right-click verb; --status / --uninstall
python scripts\probe_tool_calling.py    REM 3-stage gateway check; --prompted-json for the
                                        REM other tool mode. Costs a few tokens.
python scripts\app_smoke.py             REM headless end-to-end, no window; costs money
```

`scripts\list_bedrock_models.py` is a **leftover from the direct-Bedrock detour** and cannot
work here: it imports boto3 and calls the Bedrock control plane with AWS credentials that do
not exist on this machine. It is the only boto3 importer left in the repo and nothing imports
it. It stays only because deleting a tracked file needs the user's say-so — ask before
removing it, and do not treat it as a live diagnostic. The gateway equivalent of "what can I
reach?" is `scripts\probe_tool_calling.py`.

Prefix Python invocations with `PYTHONIOENCODING=utf-8` — LTspice output contains `Ω`, `µ`
and `°`, which crash a cp1252 console.

`.mcp.json` registers the server so it can be driven from Claude Code directly.

## State: Steps 0–8, the Explorer launcher, and the gateway migration; 288 tests pass

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
`config.py` (env/`.env`, gateway URL + model + credential *label*, no secret), `session.py`
(the spec'd log, atomic write per turn), `llm.py` (`GatewayClient`, MCP→Anthropic tool
mapping, bounded agent loop),
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
  minus `_LLM_ONLY_ENV` — `TAMU_API_KEY`, `SPICE_MCP_MODEL`, `SPICE_MCP_BASE_URL`,
  `SPICE_MCP_TOOL_MODE` — plus everything matching the `AWS_` prefix.

  **A real leak lives here.** When the transport briefly moved to direct Bedrock, the explicit
  `TAMU_API_KEY` deny entry was dropped in favour of the `AWS_` prefix rule, which *looked*
  like it covered "the credential". It did not: the gateway token then travelled into the MCP
  server — a process that knows nothing about LLMs and has no use for it — and every test
  still passed. `_LLM_ONLY_ENV` now imports `TOKEN_ENV_VAR` from `config.py` rather than
  re-typing the name, so a rename cannot silently miss a copy, and
  `tests/test_config.py::test_the_server_is_never_given_the_gateway_token` asserts it through
  the *built* environment rather than by reading the deny-list. The `AWS_` prefix rule is kept
  as belt-and-braces — the app has no AWS identity, but a shell may carry those for unrelated
  work and a netlist parser has no business seeing them; `AWS_ROLE_ARN` is in the test on
  purpose because it is named nowhere in `mcp_client.py`.

### Session log
`./sessions/<uuid>.json`, flushed after every turn: `session_id`, `started_at`, `model`,
`circuit_file`, `turns[]` (each `role`, `text`, optional `tool_calls`, `input_tokens`,
`output_tokens`), `total_input_tokens`, `total_output_tokens`, `resolved`. The Messages API's
`usage` already uses `input_tokens`/`output_tokens`, so `Turn.from_usage` renames nothing —
what it still does, and must keep doing, is **warn when an assistant turn arrives without
usage**. It reads through `usage_value()` rather than converting `usage` to a dict up front,
which is what keeps an absent count distinguishable from a zero and the warning reachable.
`usage` also carries `cache_read_input_tokens`/`cache_creation_input_tokens`; caching is off
and the schema is fixed by the external comparison, so those are deliberately dropped.

### Testing philosophy — keep this
`tests/test_checks.py` asserts the **exact set** of checks each fixture produces. Equality
rather than membership is deliberate: it makes the suite a false-positive guard. A check
that cries wolf trains both the user and the model to ignore the output, so a new check
misfiring on the known-good circuit must fail the suite loudly.

Log-parser test samples are **real captured LTspice output**, never invented. The messages
are inconsistent enough that plausible-looking fabricated samples would test the wrong thing.

The suite is offline and free: `tests/test_llm.py` drives the agent loop with a `FakeClient`
returning canned Messages-API responses. Live-model checks live in `scripts/`, not in
`pytest`, so running the tests never costs money or depends on the gateway being reachable.

`tests/conftest.py`'s **autouse `isolated_credential_environment`** is what makes that true.
It deletes `config.TOKEN_ENV_VAR` first — the developer's own `.env` holds a live
`TAMU_API_KEY`, so without that a mis-wired test could reach the live gateway and spend money,
and a "credentials are missing" test would pass on a fresh checkout while failing here. It
then strips every `AWS_*` and `SPICE_MCP_*` variable, points `HOME`/`USERPROFILE` at
`tmp_path`, and neuters `load_dotenv`. The `AWS_*` sweep is kept even though the app no longer
touches AWS: `tests/test_config.py` asserts those variables are *inert*, and that assertion
only means something if the fixture is not itself supplying them. `LTSPICE_EXE` is deliberately
left alone — it is not an LLM setting and the `needs_ltspice` tests depend on it.

### Fixtures pull in different directions on purpose
`no_dc_path.asc` is caught by the static check but **simulates fine** (exit 0, solves `.op`
"by inspection"). `source_conflict.asc` is the mirror: **statically clean**, genuinely fails
to simulate. `wrong_value_lowpass.asc` passes both and is out of spec by 10× — only circuit
*reasoning* catches it. `RCLP.asc` also simulates "successfully" despite `R1` dangling and
`Vin` driving nothing. Neither pass subsumes the other; that's the argument for two stages.

## Acceptance bar — met, except three items needing the user

`scripts\app_smoke.py` drives the real `Api` headlessly through nine stages on a temp copy of
`wrong_value_lowpass.asc`: diagnosis → **approval-gate probe** (the model is asked to write
unilaterally; the gate refuses and the file stays byte-identical) → approval → byte fidelity →
re-simulate → model confirmation → session-log audit.

It passed end to end on the gateway's **OpenAI-shaped** route (16 turns, 41 996 in / 1 572
out, schema keys exactly as specified, no null counts, totals matching). It has **not been
re-run since the move to `/v1/messages`**, so that result no longer certifies the current
transport. Run `probe_tool_calling.py` (a few tokens) first, then `app_smoke.py`. Stages 4/5
are prompt-sensitive and may need re-tuning against Opus 5's phrasing; a re-tune is fine,
**a refusal that does not happen is not**.

What *is* verified on the current transport, offline: a real `tool_use` block → the approval
gate in `Api._tool_executor` → MCP → LTspice → the exact planted value coming back in the
`tool_result` → a final answer built from it. The provenance harness plants a fresh unusual
value in a temp copy of the schematic each run, so a hallucinated answer cannot pass; the
planted value is never hardcoded into `pytest`. Both tool modes pass, role alternation is
correct in both, and the gate refuses an unapproved apply while leaving the file
byte-identical.

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
feature was the natural next move rather than a detour. Note that the walkthrough was
**not runnable at all** until the bridge-walk hang was fixed (see "The pywebview bridge
walk") — every launch produced a dead window. The launcher itself was never the problem.
Steps 1–3 below have since been reached automatically: launching on
`fixtures\wrong_value_lowpass.asc` reaches `MCP server ready with 7 tools` and a
`check_netlist_static` on the preselected circuit with no clicking, so what remains is the
part that genuinely needs eyes.

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

Automated up to that point: 288 tests, plus a headless launcher run verified to reach window
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
