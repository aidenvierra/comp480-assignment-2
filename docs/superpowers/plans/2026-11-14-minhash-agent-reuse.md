# MinHash Agent Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MinHash task-level and content-level reuse to the Spartacus agent, then measure four configurations on eight datasets within a $6 spend cap.

**Architecture:** Three graded modules (`minhash.py`, `policy.py`, `content.py`) expose self-contained APIs; `run.py` wires them into an unmodified Spartacus `Harness` (tool override via `extra_tools`, metrics via `on_event`, OpenRouter via the OpenAI backend's base URL), grades with each dataset's unmodified `verify.py`, and logs one JSON line per task; `plots.py` turns logs into tables and plots and runs the label-scored threshold sweep for the report only.

**Tech Stack:** Python 3.12, numpy, matplotlib (plots only), stdlib `unittest`, `difflib`, `hashlib`.

**Spec:** `docs/superpowers/specs/2026-11-14-minhash-agent-reuse-design.md`

## Global Constraints

- Never read anything under `hidden/`, from our code or the agent's workdir. Only `verify.grade` (called by `run.py` and `plots.py`) may.
- Do not modify `Spartacus/` or `Assignment2_COMP_580/`.
- Model `gemini-3.5-flash-lite` via OpenRouter, same model and prompt in every config.
- Task prompt, verbatim: `Solve the task in task.json by following INSTRUCTIONS.md. Write your answer to answer.json.`
- Harness: `persist=False`, `max_turns=30`, `enable_subagents=False`, default system prompt; `extra_tools` only in content/both configs.
- MinHash: 3-word shingles, 128 hash functions, seed 480, LSH 32 bands × 4 rows.
- Spend: one global ledger `results/spend.json`, default cap $6.00 across all runs.
- Default task threshold 0.5; content threshold 0.8; content savings floor: folded ≤ 50 % of original; content pass-through floor 400 chars.
- Datasets: `tickets, bugs, hosts, configs, code, threads, revisions, logs`; first N by `seq` (default 25).
- Dependencies: numpy, matplotlib only (in `requirements.txt`, installed into `.venv`).

## Review Focus

1. Docs reuse must emit `duplicate_of` = the matched history `hid` (not the history record's own answer, which has no such key) — test in Task 3.
2. Scenario ids in tasks are unnormalised (`FLT1646`, `flt-1798`); the workdir builder must still find `env/scenarios/FLT-1646` — test in Task 5.
3. A content fold must keep every changed number (e.g. `used_pct=83` vs `used_pct=61`) even though similarity masks digits — test in Task 4.
4. An agent that never writes `answer.json` (turn limit, HTTP error) must produce a failed row, not crash the run — test in Task 5.
5. A resumed run after hitting the cap must skip logged tids and rebuild the policy store from logged verified answers — test in Task 5.

---

## File structure (repo root: `Assignment2_COMP_480/`)

| File | Responsibility |
|---|---|
| `minhash.py` | tokenising, shingling, signatures, estimate, `LSHIndex` |
| `policy.py` | `TaskPolicy`: per-dataset validation rules, store, lookup, index, history-derived threshold |
| `content.py` | `ContentCache.process`, `wrap_tools` |
| `run.py` | workdir builder, OpenRouter wiring, per-task run, metrics, ledger, resume, CLI |
| `plots.py` | offline sweep, summary tables, plots |
| `test_minhash.py` | unittest suite for all of the above (one file, sections per module) |
| `README.md`, `requirements.txt`, `.gitignore` | setup and exact commands |

Path constants (in `run.py` and `test_minhash.py`, overridable by env vars `SPARTACUS_DIR`, `A2_DIR`): `Spartacus/` and `Assignment2_COMP_580/` beside the files. `DATASETS = A2_DIR/assignment2/datasets`.

---

### Task 1: Repo scaffold and `minhash.py`

**Files:** Create `.gitignore`, `requirements.txt`, `minhash.py`, `test_minhash.py`

**Interfaces — Produces:**
- `tokens(text: str, mask_numbers: bool = False) -> list[str]` — lower-case `[a-z0-9]+`; with mask, a token that is all hex and contains a digit becomes `"0"`, otherwise digit runs become `"0"` (same rule as `common.content_tokens`).
- `shingles(text, mask_numbers=False) -> set[str]` — 3-token windows joined by space; fewer than 3 tokens → one shingle of all tokens; empty text → empty set.
- `signature(text, mask_numbers=False) -> numpy.ndarray` (uint64, length `NUM_PERM=128`); empty set → all `2**64-1`.
- `estimate(a, b) -> float` — fraction of equal components.
- `exact_jaccard(text_a, text_b, mask_numbers=False) -> float` (empty/empty = 1.0).
- `class LSHIndex(bands=32, rows=4)`: `add(key, sig)`, `query(sig) -> set` of keys, `__len__`.

- [ ] **Step 1:** `git init` in `Assignment2_COMP_480/`; `.gitignore` lists `Spartacus/`, `Assignment2_COMP_580/`, `.venv/`, `__pycache__/`, `.DS_Store`, `results/work/`. `requirements.txt`: `numpy`, `matplotlib`. Create `.venv` and install. Run: `.venv/bin/python -c "import numpy, matplotlib"` → no error.
- [ ] **Step 2: Write failing tests** in `test_minhash.py` class `TestMinHash`:
  - `test_identical_texts_estimate_one`: `estimate(signature(t), signature(t)) == 1.0`.
  - `test_estimate_close_to_exact`: two texts built from a 200-word vocabulary with ~70 % overlap; `abs(estimate - exact_jaccard) < 0.12`.
  - `test_deterministic`: `signature(t)` equal across two calls (`numpy.array_equal`).
  - `test_mask_numbers`: `exact_jaccard("used_pct=83 on /", "used_pct=61 on /", mask_numbers=True) == 1.0` and `< 1.0` without mask; `tokens("req 3fa9c2", True) == ["req", "0"]`.
  - `test_short_and_empty`: `shingles("a b") == {"a b"}`; `shingles("") == set()`.
  - `test_lsh_finds_near_duplicate`: index 20 random texts plus `t`; query `t` with one word changed returns `t`'s key; a disjoint text's query does not.
- [ ] **Step 3:** Run `.venv/bin/python -m unittest test_minhash -v` → fails (no module).
- [ ] **Step 4: Implement `minhash.py`.** Shingle → int via `int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")`. Hash family `h_i(x) = (a_i * x + b_i) mod P`, `P = 2**61 - 1`, `a_i ∈ [1, P)`, `b_i ∈ [0, P)` from `random.Random(480)`. Do the arithmetic without uint64 overflow: reduce `x` mod `P` first and compute in Python ints or with numpy `object` dtype only if fast enough; acceptable alternative: split into 32-bit halves. Must handle ~3 000 shingles × 128 in < 50 ms. LSH bucket key = `(band_index, bytes of the band slice)`.
- [ ] **Step 5:** Run tests → PASS. Time `signature` on a 3 000-shingle text and note it.
- [ ] **Step 6:** Commit (`.gitignore requirements.txt minhash.py test_minhash.py`, plus `docs/`): `feat: MinHash signatures and LSH index`.

### Task 2: `policy.py` — keys, texts, answers per dataset

**Files:** Create `policy.py`; extend `test_minhash.py`

**Interfaces — Produces:**
- `dataset_name(env_dir) -> str` — parent dir name of `env/`, suffix after first `-` stripped (`tickets-d25` → `tickets`).
- `RULES: dict[str, Rule]` for all ten datasets, `Rule` = namedtuple `(text, key, answer)` of callables:
  - `text(rec) -> str`: tickets `subject+body`; bugs `title+body+stacktrace`; docs `title+body`; citations `raw`; corpus (hosts, configs, code, threads, revisions, logs) `subject+body`.
  - `key(rec) -> tuple | None` (None = cannot validate → never reuse):
    tickets `(norm_code(product), norm_code(error_code), norm_code(platform))`;
    bugs `(exception.lower(), norm_frame(top_frame), norm_version(version))`;
    docs `(identifier upper-cased, status class)` — identifier via `\b(CHG|INC|POL)-\d{3,4}\b|\bREL-\d+\.\d+\b` on title (case-insensitive), class via status words (below), `REL` → class `"none"`;
    citations `(norm_doi(doi), norm_title(title), tuple(norm_author(a) for a in authors), venue.lower(), str(year))` from `rec["citation"]`;
    corpus `(norm_code(scenario), kind.lower(), norm_code(target or ""))`.
  - `answer(rec) -> dict` for a **history** record: tickets `{"resolution_id"}`; bugs `{"component","root_cause"}`; docs `{"category": rec["category"], "duplicate_of": rec["hid"]}`; citations `rec["verdict"]`; corpus `rec["answer"]`.
- Normalisers `norm_code, norm_frame, norm_version, norm_doi, norm_title, norm_author` re-implemented in `policy.py` with the same semantics as `assignment2/datasets/common.py` (keeps `policy.py` importable without the dataset repo).
- `critical_terms(instructions_md: str) -> dict[str, list[str]]` — parsed from `INSTRUCTIONS.md`: `{"cues": [...]}` from every `story mentions *X*`, and `{"status": {class_word_list...}}` from every line containing `Status words:` (category name from the backticked name at line start, words split on `,` / ` and `, `(none)` ignored). Docs status class = the category whose words appear (word-stem match: normalised text contains the word with a trailing `d`/`ed`/`s` stripped).

- [ ] **Step 1: Failing tests** (`TestPolicyRules`), using real files under `DATASETS` (env only):
  - `test_normalisers_match_common`: for sample values (`"E-1042"`, `"lattice/net/http.py::fetch:12"`, `"v3.1.4"`, `"https://doi.org/10.5555/X."`, `"Raman, P."`), each `policy.norm_*` equals `assignment2.datasets.common.norm_*`.
  - `test_every_history_record_has_key_and_answer`: for each of the ten datasets, every `history.jsonl` row gives a non-None `key` and a dict `answer` with the keys listed in that dataset's `INSTRUCTIONS.md` Output example.
  - `test_docs_answer_has_duplicate_of_hid` (Review Focus 1).
  - `test_critical_terms_tickets`: parsed cues include `"clock"`, `"office network"`, `"password reset"`; docs status words include `"approved"` under `change-approved`.
  - `test_dataset_name`: `.../tickets-d25/env` → `tickets`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement as specified.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: per-dataset validation rules for task reuse`.

### Task 3: `policy.py` — `TaskPolicy` lookup and index

**Files:** Modify `policy.py`; extend `test_minhash.py`

**Interfaces:**
- Consumes: Task 1 `signature, estimate, LSHIndex`; Task 2 `RULES, critical_terms, dataset_name`.
- Produces: `class TaskPolicy(env_dir, threshold=None)` with
  - `.dataset`, `.threshold` (float), `.store_size`
  - `lookup(task: dict) -> tuple[dict, dict] | None` — `(answer, info)`, `info = {"source": hid_or_tid, "similarity": float, "candidates": int}`.
  - `index(task: dict, answer: dict) -> None` — adds a verified agent answer (key `task["tid"]`; stored answer returned verbatim on reuse).
  - `last_info: dict` — candidate count and best rejected reason of the last lookup (`"no_candidate" | "below_threshold" | "key_mismatch" | "critical_word"`), for logging.

Behaviour: unknown dataset → `lookup` always None. Store built from `history.jsonl` at construction. Decision = LSH candidates → estimate ≥ threshold → key equal → critical-word guard (a cue or status word present in exactly one of the two normalised texts → reject) → highest estimate wins.
Threshold when `threshold is None`: `max(0.5, min(0.9, t_max + 0.05))` where `t_max` is the largest estimated similarity among history pairs with equal key and different answer (0 if none); pairs come from LSH self-query, not all-pairs.

- [ ] **Step 1: Failing tests** (`TestTaskPolicy`), built from real history rows (copy a history row, rename `hid`→`tid`, drop answer fields):
  - `test_exact_copy_reused`: tickets row → `lookup` returns that row's answer, `info["source"] == hid`, similarity 1.0.
  - `test_platform_flip_rejected`: same row with `platform` changed → None, `last_info["reason"] == "key_mismatch"`.
  - `test_cue_flip_rejected`: tickets row whose body contains the cue `"office network"` with that phrase replaced by `"clock"` → None (`critical_word` or `below_threshold`).
  - `test_corpus_scenario_mismatch_rejected`: hosts row with `scenario` changed to another existing scenario → None.
  - `test_docs_status_flip_rejected`: docs row with `approved` replaced by `rejected` everywhere → None.
  - `test_index_then_reuse`: an incoming task not matching history → None; after `index(task, {"x": 1})`, a copy of it with a new tid → `({"x": 1}, info)` with `info["source"] == original tid`.
  - `test_unknown_dataset_never_reuses`: temp `env/` under a dir named `mystery` with a copied history → None.
  - `test_threshold_bounds`: for each of the ten datasets `0.5 <= TaskPolicy(env).threshold <= 0.9`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS; print each dataset's derived threshold and construction time (note for the report).
- [ ] **Step 5:** Commit `feat: TaskPolicy with validated MinHash reuse`.

### Task 4: `content.py`

**Files:** Create `content.py`; extend `test_minhash.py`

**Interfaces:**
- Consumes: Task 1 `signature(text, mask_numbers=True), estimate, LSHIndex`; Spartacus `spartacus.tools.core_tools(workdir) -> list[Tool]`, `spartacus.tools.Tool(name, spec, run)`.
- Produces:
  - `class ContentCache(threshold=0.8, min_chars=400, max_ratio=0.5)`: `process(tool: str, args: dict, output: str) -> str`; counters `hits`, `chars_in`, `chars_out`, `seconds` (time spent inside `process`).
  - `wrap_tools(workdir: str, cache: ContentCache) -> list[Tool]` — same-named `read_file`, `bash`, `grep` whose `run(**kw)` calls the core tool and returns `cache.process(name, kw, output)`; specs copied unchanged.

Output format (exact):
```
[reuse] identical to <ref>
```
```
[reuse] same as <ref> except (<k> of <n> lines differ):
L<i>: <new line>    (was: <old line>)
L<i>+: <added line>
L<i>-: <removed line>
```
`<ref>`: `read_file` → the `path` arg; `bash` → `bash: <command, first 80 chars>`; `grep` → `grep <regex> in <pattern>`. For `read_file` strip the `N\t` prefix before diffing and number lines by the **new** file. Outputs starting with `ERROR:`/`BLOCKED:` or shorter than `min_chars` pass through and are still stored. Exact-repeat check by `hashlib.sha1` of the output, before MinHash. Every original output is stored (fold or not). Fold only when `len(folded) <= max_ratio * len(output)`, else return original. `chars_in += len(output)`, `chars_out += len(returned)`.

- [ ] **Step 1: Failing tests** (`TestContent`), using two real host files from `hosts/env/scenarios/FLT-1000/inventory/`:
  - `test_short_output_passthrough`: 100-char output returned unchanged.
  - `test_exact_repeat`: same long output twice → second is `"[reuse] identical to a.txt"`.
  - `test_near_duplicate_folded_keeps_numbers` (Review Focus 3): read file A then file B (formatted like `read_file` output) → B's result starts with `"[reuse] same as"`, is ≤ 50 % of B's length, and every line of B that is absent from A appears verbatim in the result.
  - `test_unrelated_not_folded`: host file then an unrelated long text → unrelated returned unchanged.
  - `test_insufficient_savings_returns_original`: B = A with every third line changed → returned unchanged.
  - `test_wrap_tools_round_trip`: temp workdir with A and B; wrapped `read_file` twice gives folded second result; names are `{"read_file","bash","grep"}`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement with `difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes()`; a `replace` block of unequal length emits paired lines then `+`/`-` for the rest.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: content-level reuse of near-duplicate tool outputs`.

### Task 5: `run.py` — runner, metrics, ledger, resume (tested offline with a fake model)

**Files:** Create `run.py`; extend `test_minhash.py`

**Interfaces:**
- Consumes: `TaskPolicy` (Task 3), `ContentCache`, `wrap_tools` (Task 4); Spartacus `Harness`, `spartacus.models.PROVIDERS`, `spartacus.provider.complete`.
- Produces:
  - `load_tasks(dataset: str, n: int | None) -> list[dict]` — incoming sorted by `seq`, first n.
  - `build_workdir(dataset: str, task: dict, dest: str) -> str` — copies `INSTRUCTIONS.md`, `history.jsonl`, every other top-level *file* in `env/` (not `incoming/`, not `scenarios/`), writes `task.json`; for tasks with `scenario`, copies the contents of the matching `env/scenarios/<dir>` (match by `norm_code`) into `dest`.
  - `use_openrouter()` — sets `PROVIDERS["openai"]["base"] = "https://openrouter.ai/api/v1"`; requires `OPENAI_API_KEY` (loaded by Spartacus from `~/.config/spartacus/env`; README tells the user to put the OpenRouter key there as `OPENAI_API_KEY`).
  - `MODEL = "openai:google/gemini-3.5-flash-lite"` (confirm id in Task 6).
  - `load_verifier(dataset: str) -> module` — the dataset's `verify.py` loaded by path.
  - `class Ledger(path)`: `.spent` (float), `add(dollars)` (writes file atomically), `over(cap) -> bool`.
  - `run_task(dataset, task, config, policy, workroot, prices) -> dict` — returns the log row: `tid, seq, config, dataset, passed, reused, reused_from, similarity, policy_reason, tool_calls, model_requests, input_tokens, output_tokens, wall_s, policy_s, content_s, content_hits, chars_in, chars_out, est_cost, answer, error`.
  - `main(argv)` CLI: `--config {baseline,task,content,both}`, `--datasets` (comma list, default the eight), `--n 25`, `--cap 6.0`, `--threshold` (float, optional), `--price-in`, `--price-out` ($ per million tokens), `--out results`, `--fake` (stub model), `--jobs 4` (datasets in parallel threads).
- Grading: import `<dataset>/verify.py` via `importlib.util.spec_from_file_location`; `passed = not module.shortfall(tid, answer)`; missing/invalid `answer.json` → `answer=None`, graded (fails), row still written (Review Focus 4). Harness exceptions caught into `error`.
- Resume: before running a dataset, read existing `results/<config>/<dataset>.jsonl`; skip logged tids; for configs with task reuse, re-`index` every logged row with `passed and not reused`.
- Ledger check before each task; stop the dataset cleanly when `over(cap)`.
- `--fake`: replace `spartacus.provider.complete` with a stub that, on first call, returns a `write_file` call writing `{"fake": true}` to `answer.json` with usage `{input: 1000, output: 50}`, then a final text reply.

- [ ] **Step 1: Failing tests** (`TestRunner`, all offline):
  - `test_load_tasks_sorted_prefix`: `load_tasks("tickets", 5)` returns 5 tasks with increasing `seq`, the smallest five.
  - `test_build_workdir_corpus_unnormalised_id` (Review Focus 2): a hosts task whose `scenario` lacks the dash → workdir has `task.json`, `INSTRUCTIONS.md`, `history.jsonl`, `inventory/*.txt`, and no `incoming/`, `scenarios/` or `hidden/`.
  - `test_build_workdir_citations_has_db`: `db.json` present.
  - `test_fake_run_counts`: `run_task` with fake model, baseline → `model_requests == 2`, `tool_calls == 1`, `input_tokens == 2000`, `passed is False`, `error is None`.
  - `test_missing_answer_is_failed_row` (Review Focus 4): stub that replies with text only → row `passed False`, `answer None`.
  - `test_task_reuse_skips_agent`: config `task`, task = copy of a tickets history row → `reused True`, `model_requests == 0`, `passed True`.
  - `test_ledger_and_resume` (Review Focus 5): `main([... "--fake", "--n", "3", "--cap", "0.000001" ...])` writes ≤ 1 row per dataset and stops; rerun with large cap completes to 3 rows without duplicate tids.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run full suite → PASS. Also `python run.py --config both --datasets tickets,hosts --n 3 --fake --out /tmp/x` exits 0.
- [ ] **Step 5:** Commit `feat: experiment runner with spend ledger`.

### Task 6: `plots.py` — sweep, tables, plots

**Files:** Create `plots.py`; extend `test_minhash.py`

**Interfaces:**
- Consumes: `TaskPolicy` (with explicit `threshold`), `run.load_tasks`, `run.load_verifier`.
- Produces CLI:
  - `python plots.py sweep --out results/sweep.csv` — for each of the eight datasets and threshold in `0.1, 0.2, …, 1.0`, two modes (`gated` = full policy, `similarity_only` = threshold alone, reusing the best candidate's answer), over **all** incoming tasks against history only (no `index`): columns `dataset, threshold, mode, tasks, reused, correct_reuses, false_reuses, reuse_rate, precision`. Correctness via `verify.shortfall`. No model calls.
  - `python plots.py report --results results/full --out results/figures` — writes `summary.csv` (per config × dataset: accuracy, mean tool calls, model requests, input tokens, wall s, policy s, content s, reuse rate) and figures: `efficiency_vs_accuracy.png`, `threshold_sweep.png`, `per_dataset_metrics.png`, `overhead_vs_savings.png`. Only tids present in all four configs are counted.

- [ ] **Step 1: Failing tests** (`TestPlots`): `test_sweep_rows` — `sweep(["tickets"], [0.5, 1.0])` returns 4 rows; gated `false_reuses == 0` at both thresholds; at 1.0 `reused ≥ 1` (exact copies exist). `test_report_on_fake_results` — run `run.main` with `--fake` for all four configs into a temp dir, then `report` writes `summary.csv` with 4 × datasets rows and the four PNGs.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement (matplotlib `Agg` backend).
- [ ] **Step 4:** Run → PASS. Run the real sweep; if gated mode shows any false reuse on any dataset, stop and fix `policy.py` before Task 7.
- [ ] **Step 5:** Commit `feat: offline threshold sweep and report plots`.

### Task 7: Smoke run with the real model (USER CHECKPOINT)

No new files except possibly constant tweaks in `content.py`.

- [ ] **Step 1:** Ask the user to put the OpenRouter key in `~/.config/spartacus/env` as `OPENAI_API_KEY=...` and to set a credit limit on the key. Wait.
- [ ] **Step 2:** Confirm the model id and price: `GET https://openrouter.ai/api/v1/models` (no key needed), find the gemini 3.5 flash-lite entry, record `id` and `pricing.prompt/completion` (per token → convert to per million). Set `MODEL` and default `--price-in/--price-out` in `run.py`.
- [ ] **Step 3:** One task: `python run.py --config baseline --datasets tickets --n 1 --out results/smoke`. Expected: row with `model_requests ≥ 2` and no `error`. If a 400 mentions reasoning/thought signatures, STOP and report to the user before any workaround.
- [ ] **Step 4:** `python run.py --config baseline --n 5 --out results/smoke` (40 tasks). Report: accuracy, mean input tokens and $ per task per dataset, ledger total.
- [ ] **Step 5:** Inspect how the agent reads files (from a debug dump of tool calls: add `--trace` flag writing `results/<out>/trace/<tid>.jsonl` of tool name, args, output length). If it mostly uses `bash cat *`, raise with the user whether to add multi-file splitting (spec approach C).
- [ ] **Step 6:** `python run.py --config content --datasets hosts,logs --n 3 --out results/smoke` — check `content_hits > 0` and accuracy not below baseline on those tasks; tune content constants only if a fold dropped information.
- [ ] **Step 7:** Compute projected cost of all four configs at N=25 (task configs: assume reuse rate from Task 6's sweep at the chosen threshold). Present N options under the remaining cap to the user; wait for their N.
- [ ] **Step 8:** Commit `chore: model id, prices, smoke-run tuning`.

### Task 8: Full runs (USER CHECKPOINT before spending) and README

**Files:** Create `README.md`; results under `results/full/`

- [ ] **Step 1:** Show the user projected spend for the chosen N and the remaining ledger balance; wait for go-ahead.
- [ ] **Step 2:** Run in this order, each `--n <N> --out results/full`: `baseline`, `task`, `content`, `both`. After each, report accuracy and ledger total.
- [ ] **Step 3:** `python plots.py report --results results/full --out results/figures`.
- [ ] **Step 4:** Write `README.md`: setup (clone both repos beside, `.venv`, key file), test command, the exact sweep/run/report commands used (with N, prices, seed 480), file map, and what each config means.
- [ ] **Step 5:** Fresh-clone check: in a temp dir, clone this repo plus the two repos, run `python -m unittest test_minhash` and one `--fake` run. Expected PASS / exit 0.
- [ ] **Step 6:** Commit `docs: README and experiment results`.
