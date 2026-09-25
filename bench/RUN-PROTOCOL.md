# Run protocol: where the input tokens actually are

Written for the Windows box, because the darwin machine has no `TAMU_API_KEY` and no
LTspice — and without the preload the request shape is not comparable with the bench rows.

Every command is listed with its cost. **Run the free ones first**: if step 2 surprises you,
stop and paste it back rather than spending money on the rest.

One rule that matters: **one question per invocation.** `--question` is `action="append"`
and asks them in a single session, so a second question in the same run would inherit the
first one's history and stop being a controlled measurement.

```bat
.venv\Scripts\activate
set PYTHONIOENCODING=utf-8
```

## Step 1 — the suite (free)

```bat
python -m pytest
```

Expect green. The instrumentation is confined to `scripts/measure_turn.py`; two tests fail
on darwin for platform reasons (`test_launcher`, the case-insensitive path check) and should
pass here.

## Step 2 — transport, then the free breakdown (a few tokens, then free)

```bat
python scripts\probe_tool_calling.py --prompted-json

python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto --breakdown --label head-breakdown --json
```

Read three things out of the breakdown before going further:

- **`messages=[]:`** — a prediction being tested. The old `--breakdown` sent `messages=[]`
  to `count_tokens`, which the Messages API should reject. The line says what actually
  happened; either way the run continues, because the passes now send a one-character probe.
- **`reconstruct:` vs `+ history`** — should be `+0`. Anything else means the cumulative
  passes and the real reconstruction disagree, which would make every earlier bench row
  suspect.
- **`UNATTRIBUTED`** — the headline number. Reconstruction minus the sum of the segments.
  Expect roughly +1000 if the residual this investigation is chasing is real.

## Step 3 — the payload itself (one turn, ~2–4k input tokens)

```bat
python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto ^
  --dump-request bench\dump-head --label head --json
```

**`bench\dump-head\01-create.json` is the deliverable of the whole investigation.** Open it
and read `request.system` and `request.messages` by eye. Paths are already replaced with
`<CIRCUIT_DIR>`, `<HOME>` and `<REPO>`, and the run refuses to write a dump containing the
gateway token — so the file is safe to paste back whole.

The `usage` in that same file is the billed number for that exact text. That pairing is the
only thing that can settle the question.

## Step 4 — the three-row comparison (one turn each)

```bat
python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto --force-tools none ^
  --dump-request bench\dump-head-zero --label head-zero-tools --json

python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto ^
  --force-tools read_netlist,check_netlist_static --label head-two-tools --json

python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto --force-tools all --label head-all-tools --json
```

`--force-tools` only changes what a request *offers*. `Api._tools` stays all seven and the
approval gate in `Api._tool_executor` is untouched.

## Step 5 — history accumulation (one turn)

```bat
python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto --history-probe --label head-history --json
```

Five priced snapshots: after selecting, after selecting the *same* circuit again, after one
live turn, after selecting again, and after applying the patch if the turn proposed one.
Stage 2 is the one to watch — if the request grew, a second sidebar click is billed forever.

## Step 6 — effort, kept separate (four turns)

```bat
for %%E in (auto light medium hard) do python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort %%E --label head-effort-%%E --json
```

Input and output are reported in their own columns. No input saving may be attributed to
effort; this is what shows that rather than asserts it.

## Step 7 — the baseline row, same instrument (one turn)

```bat
git stash push scripts\measure_turn.py
git checkout experiment/concise-token-optimization
git checkout experiment/adaptive-context-optimization -- scripts\measure_turn.py

python scripts\measure_turn.py --circuit RCLP.asc ^
  --question "why is my circuit not getting any gain" ^
  --tool-mode prompted_json --effort auto ^
  --dump-request bench\dump-concise --label concise --json

git checkout -- scripts\measure_turn.py
git checkout experiment/adaptive-context-optimization
git stash pop
```

The old `bench/after.json` was recorded by a different version of the harness. Measuring the
previous branch with *this* instrument is what makes the two rows subtractable.

## What to paste back

`bench\head-breakdown.json`, `bench\head.json`, `bench\head-zero-tools.json`,
`bench\head-two-tools.json`, `bench\head-all-tools.json`, `bench\head-history.json`, the four
`bench\head-effort-*.json`, `bench\concise.json`, and
**`bench\dump-head\01-create.json`** — that last one above all.
