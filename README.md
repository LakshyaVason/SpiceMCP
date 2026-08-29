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

Then copy `.env.example` to `.env` and fill in your key:

```bat
copy .env.example .env
```

`.env` is git-ignored. **No key ever goes in source.**

## Choosing a model

The TAMU AI Chat proxy is OpenAI-compatible. List what it actually offers:

```bat
python scripts\list_tamu_models.py
```

This script deliberately lives outside the app's dependency tree — it needs only
`requests` (`pip install -r scripts\requirements.txt`) so it can be run without the full
app environment. Put your chosen model id in `SPICE_MCP_MODEL` in `.env`.

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

## Tools

| Tool | What it does |
| --- | --- |
| `read_netlist(path)` | Structured view of a circuit: components, nodes, values, nets, directives, subcircuits. Accepts `.asc` (converted via `LTspice -netlist`), `.net`, `.cir`. |
| `check_netlist_static(path)` | Findings with no simulation: missing ground, floating pins, isolated sections, no DC path to ground, duplicate refs, missing/malformed values, the `M`-means-milli trap, missing analysis directive. |
| `run_simulation(path, timeout_s)` | Runs `-b` in a scratch dir, kills the process tree on timeout, returns the parsed log. |
| `read_sim_log(path)` | Parses a `.log`: singular/over-defined matrices, undefined models (with netlist line number), convergence and timestep failures, floating nodes, missing `.include`/`.lib`, `.measure` results. |

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

## Tests

```bat
python -m pytest
```

Parser and static-check tests run without LTspice installed; schematic tests skip
automatically if the executable is not found.

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
spice_mcp_app/      desktop UI + LLM client + MCP client
scripts/            standalone utilities, own requirements.txt
fixtures/           deliberately broken circuits
sessions/           per-session token logs (git-ignored)
```

The two-process split is the point: the server never learns what an LLM is, so it can be
reused by any MCP host and later retargeted at other EDA tools.

## Session logs

Each debug session writes `sessions/<uuid>.json` with per-turn token counts. TAMU returns
OpenAI-shaped `prompt_tokens`/`completion_tokens`; these are mapped to
`input_tokens`/`output_tokens` on the way in so the log schema stays stable. Cost
comparison against the old screenshot workflow is done **externally** — the app only
records, it does not analyse.

## Notes for future work

- `mcp` must stay `>=2.1`. The 2.x API (`MCPServer`, return-annotation-derived schemas) is
  a rewrite; most tutorials online target v1 and will not run.
- `.gitattributes` exempts `*.asc`/`*.asy`/`*.net` from EOL normalisation. `RCLP.asc` is
  bare-LF while `RCLP.net` is CRLF, and `text=auto` would rewrite schematic bytes.
- `-I<path>` must be the **last** LTspice argument, after the filename, with no space.
  Placed earlier, LTspice hangs forever instead of erroring.
