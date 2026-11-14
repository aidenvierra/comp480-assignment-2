# MinHash reuse for an agent (COMP 480/580, Assignment 2)

MinHash-based **task-level** and **content-level** reuse for the
[Spartacus](https://github.com/sahiljoshi515/Spartacus) agent, evaluated on the Assignment 2
datasets with each dataset's own `verify.py`. Neither the harness nor the datasets are modified.

| File | What it does |
|---|---|
| `minhash.py` | 3-word shingles, 128 MinHash functions (seed 480), banded LSH (32 × 4) |
| `policy.py` | Task-level reuse: MinHash candidates, then a validation key and a decisive-word check |
| `content.py` | Content-level reuse: near-duplicate tool outputs become a reference plus an exact line diff; quoted replies become a reference |
| `run.py` | Runs one configuration over the datasets, verifies every answer, logs metrics, enforces a spend cap |
| `plots.py` | Offline threshold sweep (no model calls), summary table and figures |
| `test_minhash.py` | Unit tests for all of the above (offline; a stub model replaces the API) |

## Setup

```sh
git clone <this repo> a2 && cd a2
git clone https://github.com/sahiljoshi515/Spartacus
git clone <assignment repo> Assignment2_COMP_580
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`run.py` and the tests look for `Spartacus/` and `Assignment2_COMP_580/` next to this file;
set `SPARTACUS_DIR` / `A2_DIR` to use other locations.

The model is `gemini-3.5-flash-lite` through OpenRouter. Put the OpenRouter key in
`~/.config/spartacus/env` (Spartacus's key file) under the OpenAI name, because requests go
through Spartacus's OpenAI-compatible backend pointed at OpenRouter:

```sh
mkdir -p ~/.config/spartacus && echo 'OPENAI_API_KEY=sk-or-...' >> ~/.config/spartacus/env
```

## Reproducing the experiments

```sh
.venv/bin/python -m unittest test_minhash                     # offline tests, no API calls
.venv/bin/python plots.py sweep --out results/sweep.csv       # threshold sweep, no API calls
for c in baseline task content both; do
  .venv/bin/python run.py --config $c --out results/full --jobs 8 --trace
done
.venv/bin/python plots.py report --results results/full --out results/figures --sweep results/sweep.csv
```

Defaults: datasets `tickets, bugs, hosts, configs, code, threads, revisions, logs`; a random
sample of 8 incoming tasks per dataset (`--n 8 --seed 480`), solved in stream (`seq`) order;
task-reuse threshold calibrated per dataset from `history.jsonl` (0.5 on all eight); prices
$0.30 / $2.50 per million input / output tokens; one shared spend cap (`--cap`; $8 was used) kept in
`results/full/spend.json`. A rerun skips tasks already logged. `--fake` runs everything with a
stub model for free.

Configurations: `baseline` (plain agent), `task` (task-level reuse), `content` (content-level
reuse), `both`. Same model, prompt, tools and turn limit (30) in all four; sub-agents are off
in all four, because a spawned sub-agent would bypass the content layer.

## Two OpenRouter adjustments (in `run.use_openrouter`)

Spartacus has no OpenRouter backend, so `run.py` points its OpenAI backend at OpenRouter and
adjusts two things at the wire, without editing Spartacus's files:

1. Tool results are sent as `{"result": text}`, the shape Spartacus's native Gemini backend
   uses. Sent as raw text, OpenRouter forwarded only a JSON object found inside the text:
   a read of `INSTRUCTIONS.md` reached the model as its example answer and nothing else.
2. When Gemini returns `MALFORMED_FUNCTION_CALL` (reproducible for a given context, no
   tokens billed), that one request is retried with `tool_choice: "required"`.

## What must never happen

Nothing reads `hidden/`. `policy.py` and `content.py` see only the task, `history.jsonl`,
`INSTRUCTIONS.md` and tool outputs. `run.py` and `plots.py` call each dataset's `verify.py` to
grade answers, and only verified agent answers are indexed for later reuse.
