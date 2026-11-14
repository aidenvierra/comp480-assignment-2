# MinHash reuse for the Spartacus agent — design

Date: 2026-11-14. Course: COMP 480/580 Assignment 2 (due 2026-12-04).
Sources: `Assignment_2_480_580_Fall2026.pdf`, `Assignment2_COMP_580/` (datasets, verifiers),
`Spartacus/` (agent harness). Both repos are cloned beside this doc and stay unmodified.

## Intent

Show that MinHash can cut an agent's input tokens, tool calls, model requests and wall time
while keeping verified accuracy, using two mechanisms: **task-level reuse** (skip the agent when
an incoming task is a verified duplicate) and **content-level reuse** (fold near-duplicate tool
outputs into a reference plus exact differences). The work is graded: MinHash 20, task reuse 25,
content reuse 25, experiments 15, report 10, code quality/reproducibility 5. Graders will run the
code on held-out tasks, so it must generalise and must never read `hidden/`.

The report is out of scope for now; experiments must still produce everything it needs.

## Goals

- `minhash.py`: MinHash (3-word shingles, 128 hash functions, own implementation) plus LSH.
- `policy.py`: task-level reuse that never reuses a hard negative's answer.
- `content.py`: content-level reuse that never drops answer-critical numbers, ids or config values.
- `run.py`: runs the four configurations, verifies, logs metrics, enforces a spend cap.
- `plots.py`: tables and plots from the logs, plus the offline threshold sweep.
- `test_minhash.py`: small tests of the MinHash maths and the reuse safety rules.

## Non-goals

- New tasks, new verifiers, any edit to Spartacus or the dataset repo.
- Maximising reuse. An unsafe reuse is a failure even if it saves tokens.
- Semantic (embedding) similarity — mentioned only as the "improvement" in the report.

## Anti-goals (technically works, still a failure)

- Any read of `hidden/` (labels, pairs, content.json, history_meta.json) by policy, content or
  agent. Verifiers may read it; our code only calls `verify.py`'s `grade` to gate indexing.
- Hard-coded answers or per-task-id special cases.
- Spending more than the $10 budget. Default cap **$6**.
- Prompts or model differing between configurations.

## Constraints

- Model: `gemini-3.5-flash-lite` via **OpenRouter** (user's key, in `~/.config/spartacus/env`).
  Spartacus has no OpenRouter backend. `run.py` points the OpenAI backend at OpenRouter at runtime
  (`spartacus.models.PROVIDERS["openai"]["base"] = "https://openrouter.ai/api/v1"`, key as
  `OPENAI_API_KEY`, model `openai:google/gemini-3.5-flash-lite`). Exact OpenRouter model id and
  whether multi-turn tool calls work without echoed reasoning data are **verified in the smoke
  test first**; if they break, stop and report before working around it.
- Python 3.12 available; numpy 2.5 available and allowed. No other new dependencies
  (matplotlib for plots only).
- Fixed seeds everywhere (MinHash seed 480).
- Same `Harness` settings in all configurations: default tools, default system prompt, a fixed
  task prompt, `persist=False`, a fixed `max_turns` (30), and `enable_subagents=False`
  *(builder's call: a spawned child is built from core tools only, so it would silently bypass
  content reuse; disabling it everywhere keeps the four configs comparable. Stated in the report.)*

## Module boundaries *(builder's call)*

Graders may drive the three graded files from their own runner, so all reuse logic lives
behind self-contained APIs in those files; `run.py` is thin wiring:
- `policy.TaskPolicy(dataset, env_dir, threshold=None)`: `lookup(task) -> (answer, info) | None`,
  `index(task, answer)`. An unknown dataset (no validation rule) never reuses — fail-safe.
- `content.ContentCache()`: `process(tool, args, output) -> str`; and
  `content.wrap_tools(workdir, cache) -> [Tool]` building the wrapped `read_file`/`bash`/`grep`.
- `minhash`: `signature(text, mask_numbers=False)`, `estimate`, `LSHIndex`.

## Datasets and scale

Eight datasets: the six corpus sets (`hosts`, `configs`, `code`, `threads`, `revisions`, `logs`)
plus `tickets` and `bugs`. Code still supports all ten (held-out grading).

Live runs use the **first N incoming tasks per dataset in `seq` order** (default N = 25, so
200 tasks × 4 configs). A prefix, not a sample, so duplicates of earlier tasks keep their source.
N is a flag. Order of work for cost: smoke run of ~5 tasks/dataset, baseline only → measured
$/task → choose N so all four configs fit under the cap → full runs. The task-level threshold
sweep runs offline over all incoming tasks (no model calls).

## How one task runs (run.py)

1. Build a fresh temp workdir outside both repos containing: `INSTRUCTIONS.md`, `history.jsonl`,
   `task.json` (copy of `incoming/<tid>.json`), any other top-level env files (`db.json` for
   citations), and for corpus datasets the contents of `env/scenarios/<task.scenario>/` copied to
   the workdir root (giving `inventory/`, `config/`, `adapters/`, `inbox/`, `policy/`, `logs/`).
   Scenario id normalisation: case-insensitive, dash optional (`FLT1646` = `FLT-1646`).
2. If task reuse is on, ask `policy` first. A hit writes the reused answer to `answer.json`,
   no agent call.
3. Otherwise run `Harness(workdir, model=..., persist=False, max_turns=30,
   extra_tools=content_tools or None)` with the fixed prompt: *"Solve the task in task.json by
   following INSTRUCTIONS.md. Write your answer to answer.json."*
4. Grade with the dataset's `verify.grade(tid, answer)` (import the module; it never raises).
5. If the agent produced it and it passed, `policy.index(task, answer)` — only verified answers
   enter the store. Reused answers are not re-indexed.
6. Append one JSON line to `results/<config>/<dataset>.jsonl`: `tid, seq, passed, reused,
   reused_from, similarity, tool_calls, model_requests, input_tokens, output_tokens, wall_s,
   policy_s, content_s, content_hits, chars_in, chars_out, est_cost`.

Metrics come from Harness `on_event`: `"assistant"` events count model requests and carry
`usage.input/output`; `"tool_start"` counts tool calls. Wall time covers steps 2–5 including
MinHash work. Tasks within a dataset run sequentially in `seq` order (the store is online);
datasets may run in parallel threads. The log line also stores the `answer`, so a resumed run
can rebuild the policy store; a rerun skips tids already logged.

**Spend cap *(builder's call)*:** one total across every invocation (smoke + all four configs),
kept in a ledger file `results/spend.json` that each run reads and adds to. Cost = tokens ×
per-million prices (flags). N is chosen from the smoke run so all four configs fit under the
cap before any full run starts; if the cap is still hit mid-config, the run stops cleanly and
analysis uses only tids completed in all four configs (same-tasks rule). The estimate is not
the bill, so the user should also set a credit limit on the OpenRouter key.

CLI shape: `python run.py --config {baseline,task,content,both} --datasets tickets,hosts,...
--n 25 --cap 6.0 [--threshold 0.5]`. A `--sweep` mode in `plots.py` does the offline sweep.

## minhash.py

- Normalise: lower-case, alphanumeric tokens. Option `mask_numbers` turns digit runs into `0`
  (used for content similarity, where files differ only in numbers; off for task text).
- Shingles: consecutive 3-token tuples; texts under 3 tokens give one shingle of all tokens.
  Each shingle → 64-bit int via `hashlib.blake2b(digest_size=8)`.
- 128 hash functions `h_i(x) = (a_i·x + b_i) mod (2^61 − 1)` with `a_i, b_i` from
  `random.Random(480)`. Signature = per-function minimum, vectorised in numpy.
- `estimate(sig_a, sig_b)` = fraction of equal components. An `exact_jaccard` helper for tests
  and the report.
- `LSHIndex(bands=32, rows=4)`: `add(key, sig)`, `query(sig)` → candidate keys from shared band
  buckets. 32×4 gives a recall knee near J ≈ 0.42, below the lowest duplicate band we want to
  catch, so the threshold (not LSH) decides.

## policy.py — task-level reuse (approach B: similarity plus a field check)

Store per dataset: one record per verified item (all `history.jsonl` rows at start, plus
verified agent answers), each with its text signature, its validation key, and its answer.

Decision for an incoming task:
1. Text = the dataset's similarity text (tickets: subject+body; bugs: title+body+stacktrace
   fields; docs: title+body; citations: raw; corpus: subject+body). Signature, LSH query.
2. Keep candidates with estimated J ≥ threshold (default per dataset, see below).
3. **Validation gate** (rules quoted from each `INSTRUCTIONS.md`, normalised the same way):
   - tickets: `product`, `error_code` (`E-1042` = `e1042`), `platform`.
   - bugs: `exception`, `top_frame` (dotted = path form, no line numbers), version series
     (major.minor).
   - docs: identifier from title (upper-cased) and status class (status word → class).
   - citations: DOI, title, author (surname, initial) pairs, venue abbreviation, year.
   - corpus: `scenario`, `kind`, `target`.
   The public helpers in `assignment2/datasets/common.py` (`norm_code`, `norm_frame`,
   `norm_version`, `norm_doi`, `norm_title`, `norm_author`) implement exactly these published
   rules and may be imported; nothing from `hidden/`.
4. **Critical-word guard**: reject if a dataset's answer-deciding words appear in one text but
   not the other — ticket cue terms (the italic cues in the resolution guide), docs status words.
   Word lists are short constants in `policy.py` taken from `INSTRUCTIONS.md`.
5. Best surviving candidate wins; reuse its answer. Docs special case: reusing a history record
   gives `{"category": record.category, "duplicate_of": record.hid}`; reusing a verified task
   answer copies it as is.
6. No survivor → run the agent.

Thresholds *(builder's call)*: set **without labels**, so nothing is tuned on the test set.
Default 0.5 for every dataset; a per-dataset value may be derived only from `history.jsonl`
itself (history pairs that pass the validation gate but have different answers — the "history
twins" — mark the similarity band that is unsafe) and is computed at policy construction, so it
also works on held-out data. Paraphrase duplicates (J 0.15–0.37) are deliberately left to the
agent: a miss costs time, a wrong reuse costs accuracy.

**Offline sweep** (`plots.py --sweep`, report-only, never feeds back into `policy.py`): for each
dataset and threshold in 0.1…1.0, replay all incoming tasks against history only, score each
reuse with `verify.grade`, and report reuse rate, reuse precision, and false reuses, with and
without the validation gate (approach C, gate only, as a reference point). No model calls.

## content.py — content-level reuse (approach B: near-copies as differences)

Wrap `read_file`, `bash` and `grep`: `run.py` builds Spartacus's core tools, then passes
same-named replacements in `extra_tools` whose `run` calls the original and passes the output
through `ContentCache.process(tool, args, output)`. One cache per task (one trajectory).

`process`:
1. Outputs under ~400 chars, and error outputs, pass through untouched.
2. Exact repeat of an earlier output → `[reuse] identical to <earlier ref>`.
3. Else signature with `mask_numbers=True`, LSH query over earlier outputs, best candidate with
   estimated J ≥ 0.8 (tunable).
4. Line diff against that candidate on the **raw** text (difflib), so every changed number, id
   and value is kept. For `read_file`, compare with line-number prefixes stripped and emit
   changed lines with their line numbers.
5. Emit `[reuse] same as <ref> except:` followed by the differing lines (each marked with its
   line number and old → new where it is a replacement, added or removed otherwise).
6. If the folded text is not at most 50 % of the original, return the original.
7. Store every original output (folded or not) so later outputs can reference it.

`<ref>` is human-readable: `inventory/web-01.txt` for reads, the command for bash.

Multi-file splitting of a single output (e.g. `cat *.txt`) is out of scope unless the smoke run
shows the agent reads files that way (that was approach C; revisit then).

## Approaches considered

Task-level:
- A. Similarity threshold only — rejected: hard negatives reach J 0.93, paraphrase duplicates
  sit at 0.15–0.37, so no threshold is safe.
- **B. MinHash candidates + validation fields + critical-word guard — chosen by the user.**
- C. Validation fields only — kept as a reference curve in the sweep, not the method.

Content-level:
- A. Exact repeats only — safe but small savings; corpus files are near-copies, not copies.
- **B. MinHash near-copy detection + exact line diff — chosen by the user.** Matches the
  `[reuse]` behaviour the datasets' `INSTRUCTIONS.md` already describe to the agent.
- C. B plus splitting multi-file outputs — deferred pending smoke-run evidence.

## Decided by the user

- Use B for both mechanisms.
- Budget $10 total, be conservative; smoke run first and size the full runs from it.
- Subset by first N per dataset is acceptable; default N = 25.
- Six files is fine (`minhash.py`, `policy.py`, `content.py`, `run.py`, `plots.py`,
  `test_minhash.py`).

## Left to the builder

- Exact diff text format, small constants (400-char floor, 0.8 content threshold, 50 % savings
  floor) — tune on smoke-run trajectories, report final values.
- Thread-level parallelism across datasets.
- Plot styling. Required outputs: accuracy vs input tokens per config; threshold sweep
  (reuse rate and false reuses vs threshold, per dataset); per-dataset bar charts of the five
  metrics; MinHash overhead vs time saved.
- Held-out grading: we don't know the graders' runner. The module APIs above are the
  assumption; if the course publishes an interface, adapt to it.
- Where the submission repo lives: proposed as a new git repo in this folder holding the six
  files, `results/`, and a README with exact commands; the two cloned repos sit beside it,
  ignored, with clone instructions.
