# SPICE MCP client

Debug LTspice circuits by talking to an LLM about the **netlist**, not a screenshot.

An MCP server exposes the circuit as structured text — components, nodes, values,
directives, static checks, simulation logs — so the model reasons over topology instead of
pixels. That is cheaper in input tokens and gives far more reliable answers, because the
model can see that `R1` connects to `NC_01` rather than squinting at a wire that looks
close enough to a pin.

Milestone 1 of a broader effort to build MCP servers for EDA tools. LTspice is the open
sandbox for validating the approach before Cadence-class tools.

## Requirements

- **Windows.** LTspice has no native Linux build, and batch invocation is Windows-only in
  v1. Reading and statically checking an existing `.net`/`.cir` works anywhere; converting
  a `.asc` or running a simulation needs the executable.
- **LTspice** (developed against 26.0.2). Auto-detected at the per-user
  `%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe` and the older `Program Files\LTC`
  paths. Override with `LTSPICE_EXE` if yours is elsewhere.
- **Python 3.11+** (developed on 3.13).

## Setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

To pick the environment back up in a later session, that middle line is the one you want:

```bat
.venv\Scripts\activate
```

(In Git Bash it is `source .venv/Scripts/activate`; in PowerShell,
`.venv\Scripts\Activate.ps1`.)

- **AWS credentials with Bedrock access.** The LLM is Claude on **Amazon Bedrock**, so
  auth is IAM-based — there is no API key for this app to hold. Anything the standard AWS
  chain can find works: `aws configure`, `AWS_PROFILE`, `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` (plus `AWS_SESSION_TOKEN` for temporary credentials), an instance
  role, or an `AWS_BEARER_TOKEN_BEDROCK` if your organisation issued one instead.

Then copy `.env.example` to `.env`:

```bat
copy .env.example .env
```

Every setting in it is optional — if your credentials already resolve, the defaults
(`us.anthropic.claude-opus-5` in `us-east-1`) work as shipped. `.env` is git-ignored, and
**no credential ever enters the app's config object**, so there is nothing for a stray debug
print to leak: the AWS SDK reads the environment itself and the app only records *which
style* of credential it found.

## Choosing a model

`SPICE_MCP_MODEL` is a Bedrock **inference profile id**, not a display name — e.g.
`us.anthropic.claude-opus-5`. The `us.` prefix routes across US regions; `global.` is the
same model without the geographic restriction and is billed about **10% cheaper**, which is
worth noting in any cost write-up so the number is not read as base pricing.

Profile availability is **region-scoped** and gated on model access being granted to your
account, so list what you actually have before relying on anything:

```bat
python scripts\list_bedrock_models.py
```

That prints the Claude inference profiles and Anthropic foundation models visible in the
resolved region, and flags whether the app's default is among them. It calls Bedrock's
control plane only, so it is **free**. Like the other scripts it lives outside the app's
dependency tree — `pip install -r scripts\requirements.txt` is enough to run it.

The app needs the model to support **tool use**. To prove a real round trip end to end:

```bat
python scripts\probe_tool_calling.py                                  REM native
python scripts\probe_tool_calling.py <model id> --prompted-json       REM the fallback
```

Three stages — a plain completion with non-zero usage, a genuine `tool_use` block with
parsed input, and a `tool_result` reply that closes the loop with `stop_reason: "end_turn"`.
This one sends real requests, so it costs real money on your AWS account. Run
`list_bedrock_models.py` first if either credentials or model access is in doubt.

Set `SPICE_MCP_AWS_REGION` in `.env` to move regions; it falls back to `AWS_REGION`, then
`AWS_DEFAULT_REGION`, then `us-east-1`.

## Running the app

```bat
python -m spice_mcp_app --folder fixtures
```

Open a folder, pick a circuit, and describe the symptom. The model has the seven tools
below and will read, check and simulate on its own. When it proposes a value change you get
a before/after diff with **Apply** and **Reject**; applying writes the `.asc` and
immediately re-simulates, because a fix that was never verified is not a finished fix.

`--debug` turns up logging. The header shows running token totals; **Export log** saves the
session JSON wherever you want it.

`--file <path>` opens a single circuit: it selects it and runs the static checks on load, and
its folder populates the sidebar. That is what the Explorer entry below uses.

## Launch from Explorer

Registering a right-click entry removes the three manual steps between "something looks wrong"
and "ask about it":

```bat
python scripts\install_context_menu.py              REM install
python scripts\install_context_menu.py --status      REM what is actually registered
python scripts\install_context_menu.py --uninstall   REM remove
```

Then right-click any `.asc` and choose **Debug with SPICE MCP**. It opens the schematic in
LTspice *and* the client, with that circuit already selected and its static checks already on
screen.

**On Windows 11 this lives in the legacy menu** — right-click, then **Show more options** (or
Shift+right-click). The new compact menu is populated only by packaged (MSIX) apps implementing
the `IExplorerCommand` COM interface; there is no registry setting that promotes a classic verb
into it.

Details worth knowing:

- **Per-user, no administrator rights, reversible.** Everything goes under
  `HKCU\Software\Classes\SystemFileAssociations\.asc\shell\`, which *adds* a verb. It creates
  no ProgID, never touches `HKCU\Software\Classes\.asc`, and stays clear of `shell\open`, so
  LTspice's own association and what double-clicking a schematic does are unchanged. The verb
  is marked `NeverDefault` so it can never be promoted to the double-click action.
- **The absolute paths are baked into the registry**, because the registry cannot give a
  command a working directory. Move or rename the repo, or rebuild `.venv`, and the entry
  stops working until you re-run the installer. `--status` diagnoses exactly that.
- It runs `pythonw.exe`, so no console window flashes. The cost is that a failure before the
  window exists has nowhere to print, so the launcher shows a message box for fatal errors and
  writes `launch.log` at the repo root.
- The apps come up together **when you start from the schematic**. Opening LTspice on its own
  from the Start menu will not summon the client — nothing watches for it.

## Editing in LTspice at the same time

The circuit is bound once, at launch. Nothing polls LTspice, and the client only ever reads the
`.asc` **from disk**, so:

- **Before asking**, save in LTspice (Ctrl+S) — unsaved GUI edits are invisible here. The app
  says so once per session when it notices LTspice running.
- **After applying a fix**, LTspice will not notice the external edit. It holds its own copy and
  will write it back over yours if you save from the GUI. Use **File ▸ Revert** to reload the
  patched file. The app warns about this in the conversation whenever LTspice is running.

The write itself is never blocked on this — you asked for the fix and the file is yours. The
warning exists so a fix cannot quietly disappear later.

## Running the MCP server

Normally the desktop app spawns it. To run it standalone (it speaks MCP over stdio, so it
will just sit there waiting for a client — that is correct):

```bat
python -m spice_mcp_server
```

`SPICE_MCP_LOG_LEVEL=DEBUG` turns up the diagnostics, which go to **stderr**; stdout is
the protocol wire and must stay clean.

The repo ships a `.mcp.json`, so the server can also be driven straight from Claude Code
against the fixtures without the app existing yet.

Nothing about the server is Claude Code specific — it is a plain MCP stdio server and any
MCP host can spawn it. `.mcp.json` is just one host's registration file. Its `command` is
relative to the repo root, so a host that launches from somewhere else needs either an
absolute path to `.venv\Scripts\python.exe` or the repo set as the working directory. The
real constraints are the server's own: Windows, and LTspice for anything touching a `.asc`.

## Tools

| Tool | What it does |
| --- | --- |
| `read_netlist(path)` | Structured view of a circuit: components, nodes, values, nets, directives, subcircuits. Accepts `.asc` (converted via `LTspice -netlist`), `.net`, `.cir`. |
| `check_netlist_static(path)` | Findings with no simulation: missing ground, floating pins, isolated sections, no DC path to ground, duplicate refs, missing/malformed values, the `M`-means-milli trap, missing analysis directive. |
| `run_simulation(path, timeout_s)` | Runs `-b` in a scratch dir, kills the process tree on timeout, returns the parsed log. |
| `read_sim_log(path)` | Parses a `.log`: singular/over-defined matrices, undefined models (with netlist line number), convergence and timestep failures, floating nodes, missing `.include`/`.lib`, `.measure` results. |
| `patch_component_value(asc_path, ref, new_value, apply)` | Changes one component's value in a `.asc`. **Previews by default** — returns the diff without writing. |
| `diff_netlist(before_path, after_path)` | Compares two circuits by reference designator: value changes, rewires, added/removed parts, net and directive changes. |
| `export_netlist(path, out_path)` | Writes a real SPICE netlist to a path you choose. Refuses to overwrite. |

### Never trust the exit code

Two LTspice behaviours, both verified against 26.0.2, make the obvious success signals
useless:

- **A failed run can exit 0.** `ERROR: Node n1 is floating and connected to current
  source I1` exits 0. A missing ground exits 1. There is no consistent mapping.
- **A `.raw` file is written even when the analysis failed.** The over-defined-matrix
  failure still produced a complete-looking waveform file.

So `succeeded` is derived from the log text alone. `returncode` is carried in the result
for diagnostics only, and is documented as such in the tool schema so the model doesn't
reach for it.

The log messages themselves are not uniform either — of the five distinct failures the
parser handles, only two carry an `ERROR:`/`WARNING:` prefix, and one arrives in a
compile-style `path(line): message` form. A grep for `ERROR` would miss the two most
serious failures outright.

## The `.asc` vs `.net` distinction

`.asc` is the schematic and the **source of truth** — it carries the GUI coordinates, so a
fix written back there stays usable in LTspice. `.net` is the flattened netlist. Prefer
pointing tools at the `.asc`.

Beware: LTspice's *Export Netlist* menu item (and `-PCBnetlist`) writes an **ExpressPCB**
netlist, not SPICE. The `RCLP.net` in this repo is one of those. The server detects the
format, and falls back to converting the sibling `.asc` when there is one rather than
misparsing PCB connectivity as a circuit.

## Your files are never written to

LTspice writes `-netlist` and `-b` output *next to the input file*, which means running it
on your schematic silently overwrites your own `.net`. Every invocation therefore stages
the input into a scratch directory under `%TEMP%\spice_mcp_work\` first, and all
artifacts land there. `tests/test_ltspice.py` pins this down — it is a regression test for
a real incident, not a hypothetical.

## The model cannot edit your schematic on its own

Exactly one tool writes to your files, and the approval for it is registered by the
**Apply** button, not by the model. If the model calls `patch_component_value` with
`apply=true` off its own bat, the app refuses the call and tells it to show you a diff
instead. Approvals are single-use and specific to one file, one reference designator and one
value — approving `C1 → 100n` does not authorise `R1`, or a second write of the same value.
`tests/test_llm.py` asserts the refused call never reaches the server; the end-to-end script
also asks the model to bypass the gate and checks the file comes back byte-identical.

The gate sits on the executor, below both tool-calling modes, so `prompted_json` buys the
model no extra reach — the same test drives the same unapproved write through both.

When a write does happen it is surgical: the one `SYMATTR Value` line belonging to that
component is rewritten, reusing the file's own indentation and line terminator. Encoding,
line endings and every other byte are preserved, so the schematic still opens in the LTspice
GUI — which is the whole point of patching the `.asc` rather than a netlist.

## Tests

```bat
python -m pytest
```

225 tests, ~40s. Parser and static-check tests run without LTspice installed; schematic
tests skip automatically if the executable is not found. Nothing in the suite calls the
network, so running it costs nothing — the live-model checks are the two scripts,
`probe_tool_calling.py` and `app_smoke.py` (a headless nine-stage end-to-end run against
a temp copy of `wrong_value_lowpass.asc`).

Staying free is enforced, not hoped for: an autouse fixture strips every `AWS_*` and
`SPICE_MCP_*` variable, points `HOME`/`USERPROFILE` at a temp directory so `~/.aws` cannot
be found, and neuters `.env` loading. Otherwise the AWS credential chain would happily
succeed from somewhere no test mentions — making a "credentials are missing" test pass on a
bare laptop and fail on a configured one, or letting a mis-wired test spend real money.

The fixture matrix in `tests/test_checks.py` asserts the **exact** set of checks each
circuit produces. Equality rather than membership is deliberate: it makes the suite a
false-positive guard, so a new check that misfires on the known-good circuit fails loudly
instead of quietly training everyone to ignore the output.

`fixtures/` holds deliberately broken circuits, each documenting its own bug in a `TEXT`
comment:

| Fixture | Bug | Caught by |
| --- | --- | --- |
| `good_lowpass.asc` | None — exists so checks can be proven silent on a correct circuit. | — |
| `floating_node.asc` | `R2`'s right pin dangles. | static |
| `missing_ground.asc` | Loop is closed but there is no node 0. | static + sim (singular matrix) |
| `no_dc_path.asc` | `Vmid`/`Vout` reach ground only through capacitors. | static only |
| `source_conflict.asc` | Two ideal sources of different value in parallel. | sim only (over-defined matrix) |
| `wrong_value_lowpass.asc` | Correct topology, `C1` off by 10×. | neither |

The last three are the interesting ones, because they pull in different directions:

- `no_dc_path.asc` is flagged by the static check but **simulates fine** — LTspice exits
  0, solves the `.op` "by inspection", and only warns. The DC operating point is not
  actually determined, and the simulator doesn't care.
- `source_conflict.asc` is the mirror image: **statically clean**, but the simulation
  genuinely fails.
- `wrong_value_lowpass.asc` passes both. It is out of spec by 10× and nothing but circuit
  reasoning will catch it. That is the fixture that tests whether the model understands
  the circuit rather than just parsing errors.

Together they're the argument for the two-stage design: neither pass subsumes the other.

`RCLP.asc` is fixture 0 and genuinely broken two ways — `R1` has a floating pin and is not
in the signal path, so `Vin` never drives `Vout`. It is worth running yourself, because
**it simulates "successfully"**: the `.ac` analysis completes with no warning at all. Only
the static check notices that the circuit cannot possibly do what it looks like it does.

## Layout

```
spice_mcp_server/   MCP server. Knows nothing about LLMs.
  models.py         pydantic return models — these ARE the tool output schemas
  netlist.py        SPICE parsing; ExpressPCB detection
  checks.py         static checks
  logparse.py       .log parsing — where the real diagnosis usually lives
  ltspice.py        exe discovery, staged -netlist / -b invocation, timeouts
  asc.py            byte-preserving .asc value patching
  diff.py           before/after circuit comparison
spice_mcp_app/      desktop UI + LLM client + MCP client
  config.py         model/region/credential-style resolution; holds no credential
  session.py        the token log; atomic write after every turn
  llm.py            Bedrock client, MCP→Anthropic tool mapping, agent loop
  mcp_client.py     stdio client; holds one server subprocess open
  api.py            the JS bridge — and the approval gate
  launch.py         the Explorer entry point: opens LTspice, then the window
  web/              index.html, style.css, app.js
spice_mcp_launch.py repo-root shim the registry command points at
scripts/            standalone utilities, own requirements.txt
  install_context_menu.py   adds/removes the right-click verb (HKCU, no admin)
fixtures/           deliberately broken circuits
sessions/           per-session token logs (git-ignored)
```

The two-process split is the point: the server never learns what an LLM is, so it can be
reused by any MCP host and later retargeted at other EDA tools.

## Session logs

Each debug session writes `sessions/<uuid>.json` with per-turn token counts. Cost
comparison against the old screenshot workflow is done **externally** — the app only
records, it does not analyse. With Bedrock the per-call charges also land in CloudWatch and
Cost Explorer, so the log and the bill can be reconciled against each other.

```json
{
  "session_id": "…", "started_at": "…", "model": "…", "circuit_file": "…\\RCLP.asc",
  "turns": [{"role": "assistant", "text": "…", "tool_calls": [],
             "input_tokens": 4211, "output_tokens": 96}],
  "total_input_tokens": 41996, "total_output_tokens": 1572, "resolved": true
}
```

The file is rewritten atomically after **every** turn, so a crash mid-session still leaves a
complete log. `tool_calls` is present only on turns that made them. `resolved` is the
**Mark resolved** button — it is how you tell, later, which sessions actually ended in a fix.

The schema is fixed by that external comparison, so it deliberately does **not** grow keys:
Bedrock also reports `cache_read_input_tokens` and `cache_creation_input_tokens`, and since
prompt caching is not enabled they are dropped rather than logged. An assistant turn that
arrives with no usage at all is logged as zero **and warned about** — a silently null count
would under-report cost without anything visibly failing.

## Notes for future work

- `mcp` must stay `>=2.1`. The 2.x API (`MCPServer`, return-annotation-derived schemas) is
  a rewrite; most tutorials online target v1 and will not run.
- **Never send a `thinking` parameter.** Adaptive thinking is on by default for Opus 5;
  `budget_tokens` is rejected with a 400, and explicitly *disabling* thinking makes the model
  occasionally write a tool call into its visible text rather than emitting a `tool_use`
  block — which is the exact failure this transport was migrated to fix.
- `.gitattributes` exempts `*.asc`/`*.asy`/`*.net` from EOL normalisation. `RCLP.asc` is
  bare-LF while `RCLP.net` is CRLF, and `text=auto` would rewrite schematic bytes.
- `-I<path>` must be the **last** LTspice argument, after the filename, with no space.
  Placed earlier, LTspice hangs forever instead of erroring.
