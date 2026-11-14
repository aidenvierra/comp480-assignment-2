"""Run the Spartacus agent on dataset tasks under one configuration, verify, and log metrics.

    python run.py --config both --datasets tickets,hosts --n 25

Configurations: baseline (plain agent), task (task-level reuse), content (content-level
reuse), both. Each task gets a fresh working directory holding only what the agent may see.
One JSON line per task goes to <out>/<config>/<dataset>.jsonl; a rerun skips logged tasks.
All runs share one spend ledger, <out>/spend.json, and stop at --cap dollars.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SPARTACUS_DIR = os.environ.get("SPARTACUS_DIR", os.path.join(HERE, "Spartacus"))
A2_DIR = os.environ.get("A2_DIR", os.path.join(HERE, "Assignment2_COMP_580"))
DATASETS = os.path.join(A2_DIR, "assignment2", "datasets")
sys.path.insert(0, SPARTACUS_DIR)

from spartacus import Harness, models, provider  # noqa: E402

import content  # noqa: E402
import policy  # noqa: E402

MODEL = "openai:google/gemini-3.5-flash-lite"   # OpenRouter id, routed through the OpenAI backend
OPENROUTER = "https://openrouter.ai/api/v1"
PRICES = (0.30, 2.50)                           # $ per million input / output tokens (OpenRouter, 2026-11-14)
PROMPT = "Solve the task in task.json by following INSTRUCTIONS.md. Write your answer to answer.json."
MAX_TURNS = 30
DEFAULT_DATASETS = ["tickets", "bugs", "hosts", "configs", "code", "threads", "revisions", "logs"]
CONFIGS = {"baseline": (False, False), "task": (True, False), "content": (False, True), "both": (True, True)}


# ---------------------------------------------------------------------------
# tasks and working directories
# ---------------------------------------------------------------------------
def env_dir(dataset):
    return os.path.join(DATASETS, dataset, "env")


def load_tasks(dataset, n):
    """The first n incoming tasks in stream order (all of them when n is None)."""
    folder = os.path.join(env_dir(dataset), "incoming")
    tasks = [read_json(os.path.join(folder, name)) for name in os.listdir(folder)]
    return sorted(tasks, key=lambda t: t["seq"])[:n]


def build_workdir(dataset, task, dest):
    """A directory with the instructions, history, reference files, task.json and, for a
    corpus task, its scenario's files. Nothing from incoming/ or hidden/."""
    os.makedirs(dest, exist_ok=True)
    source = env_dir(dataset)
    for name in os.listdir(source):
        if os.path.isfile(os.path.join(source, name)):
            shutil.copy(os.path.join(source, name), dest)
    if task.get("scenario"):
        scenarios = os.path.join(source, "scenarios")
        match = next(d for d in os.listdir(scenarios) if policy.norm_code(d) == policy.norm_code(task["scenario"]))
        shutil.copytree(os.path.join(scenarios, match), dest, dirs_exist_ok=True)
    with open(os.path.join(dest, "task.json"), "w") as handle:
        json.dump(task, handle, indent=2)
    return dest


_verifiers = {}


def load_verifier(dataset):
    """The dataset's own verify.py, imported by path and left unmodified."""
    if dataset not in _verifiers:
        path = os.path.join(DATASETS, dataset, "verify.py")
        spec = importlib.util.spec_from_file_location("verify_" + dataset, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _verifiers[dataset] = module
    return _verifiers[dataset]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def use_openrouter():
    """Point Spartacus's OpenAI-compatible backend at OpenRouter (key: OPENAI_API_KEY)."""
    models.PROVIDERS["openai"]["base"] = OPENROUTER


def fake_model(model, system, messages, tools=None):
    """A free stand-in for tests: writes a placeholder answer, then stops."""
    usage = {"input": 1000, "output": 50}
    if tools and not any(m["role"] == "tool" for m in messages):
        call = {"name": "write_file", "args": {"path": "answer.json", "content": '{"fake": true}'},
                "signature": None}
        return {"text": "", "tool_calls": [call], "usage": usage}
    return {"text": "done", "tool_calls": [], "usage": usage}


class Ledger:
    """Dollars spent across every run, kept on disk so the cap holds across invocations."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.spent = read_json(path)["spent"] if os.path.exists(path) else 0.0

    def add(self, dollars):
        with self.lock:
            self.spent += dollars
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path + ".tmp", "w") as handle:
                json.dump({"spent": self.spent}, handle)
            os.replace(self.path + ".tmp", self.path)

    def over(self, cap):
        return self.spent >= cap


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------
def run_task(dataset, task, config, task_policy, workroot, prices, model=MODEL, trace=False):
    """Solve one task (by reuse or by the agent), verify it, and return its log row."""
    use_task, use_content = CONFIGS[config]
    start = time.perf_counter()
    row = {"tid": task["tid"], "seq": task["seq"], "config": config, "dataset": dataset,
           "reused": False, "reused_from": None, "similarity": None, "policy_reason": None,
           "tool_calls": 0, "model_requests": 0, "input_tokens": 0, "output_tokens": 0,
           "policy_s": 0.0, "content_s": 0.0, "content_hits": 0, "chars_in": 0, "chars_out": 0,
           "answer": None, "error": None}
    work = build_workdir(dataset, task, tempfile.mkdtemp(dir=workroot))
    steps = []
    try:
        hit = None
        if use_task:
            tick = time.perf_counter()
            hit = task_policy.lookup(task)
            row["policy_s"] = time.perf_counter() - tick
            row["policy_reason"] = task_policy.last_info["reason"]
        if hit:
            row["answer"], info = hit
            row.update(reused=True, reused_from=info["source"], similarity=info["similarity"])
        else:
            cache = content.ContentCache() if use_content else None

            def on_event(kind, payload):
                if kind == "assistant":
                    usage = payload.get("usage") or {}
                    row["model_requests"] += 1
                    row["input_tokens"] += usage.get("input", 0)
                    row["output_tokens"] += usage.get("output", 0)
                elif kind == "tool_start":
                    row["tool_calls"] += 1
                elif kind == "tool_end" and trace:
                    call = payload["call"]
                    steps.append({"tool": call["name"], "args": call["args"], "chars": len(str(payload["result"]))})

            Harness(work, model=model, persist=False, max_turns=MAX_TURNS, enable_subagents=False,
                    on_event=on_event, extra_tools=content.wrap_tools(work, cache) if cache else None).run(PROMPT)
            row["answer"] = _read_answer(work)
            if cache:
                row.update(content_s=cache.seconds, content_hits=cache.hits,
                           chars_in=cache.chars_in, chars_out=cache.chars_out)
    except Exception as error:  # a failed task is a row, never a crashed run
        row["error"] = "%s: %s" % (type(error).__name__, error)
        row["answer"] = row["answer"] or _read_answer(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    row["passed"] = not load_verifier(dataset).shortfall(task["tid"], row["answer"])
    if use_task and row["passed"] and not row["reused"]:
        tick = time.perf_counter()
        task_policy.index(task, row["answer"])  # only verified answers enter the store
        row["policy_s"] += time.perf_counter() - tick
    row["wall_s"] = time.perf_counter() - start
    row["est_cost"] = (row["input_tokens"] * prices[0] + row["output_tokens"] * prices[1]) / 1e6
    if trace:
        row["trace"] = steps
    return row


def _read_answer(work):
    try:
        return read_json(os.path.join(work, "answer.json"))
    except (OSError, ValueError):
        return None


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_jsonl(path):
    """The rows of a JSON-lines file; none when it does not exist yet."""
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# one dataset, then the command line
# ---------------------------------------------------------------------------
def run_dataset(dataset, opts, ledger):
    use_task, _ = CONFIGS[opts.config]
    log = os.path.join(opts.out, opts.config, dataset + ".jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    done = read_jsonl(log)
    tasks = load_tasks(dataset, opts.n)
    task_policy = policy.TaskPolicy(env_dir(dataset), threshold=opts.threshold) if use_task else None
    if task_policy:  # a resumed run rebuilds the store from answers it already verified
        by_tid = {t["tid"]: t for t in tasks}
        for row in done:
            if row["passed"] and not row["reused"] and row["tid"] in by_tid:
                task_policy.index(by_tid[row["tid"]], row["answer"])
    logged = {row["tid"] for row in done}
    workroot = tempfile.mkdtemp(prefix="minhash-agent-")
    for task in tasks:
        if task["tid"] in logged:
            continue
        if ledger.over(opts.cap):
            print("[%s] spend cap $%.2f reached; stopping" % (dataset, opts.cap))
            break
        row = run_task(dataset, task, opts.config, task_policy, workroot,
                       (opts.price_in, opts.price_out), trace=opts.trace)
        ledger.add(row["est_cost"])
        with open(log, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print("[%s/%s] %s %s%s in=%d tools=%d %.1fs  spent $%.4f" % (
            opts.config, dataset, row["tid"], "PASS" if row["passed"] else "FAIL",
            " (reused)" if row["reused"] else "", row["input_tokens"], row["tool_calls"],
            row["wall_s"], ledger.spent), flush=True)
    shutil.rmtree(workroot, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n", type=int, default=25, help="first N tasks per dataset, by seq")
    parser.add_argument("--cap", type=float, default=6.0, help="total $ across all runs")
    parser.add_argument("--threshold", type=float, help="task-reuse threshold (default: calibrated)")
    parser.add_argument("--price-in", type=float, default=PRICES[0], help="$ per million input tokens")
    parser.add_argument("--price-out", type=float, default=PRICES[1], help="$ per million output tokens")
    parser.add_argument("--out", default=os.path.join(HERE, "results"))
    parser.add_argument("--jobs", type=int, default=4, help="datasets run in parallel")
    parser.add_argument("--trace", action="store_true", help="log each tool call's name, args, size")
    parser.add_argument("--fake", action="store_true", help="stub model, no API calls")
    opts = parser.parse_args(argv)

    if opts.fake:
        provider.complete = fake_model
    else:
        use_openrouter()
    ledger = Ledger(os.path.join(opts.out, "spend.json"))
    datasets = opts.datasets.split(",")
    with ThreadPoolExecutor(max_workers=max(1, min(opts.jobs, len(datasets)))) as pool:
        list(pool.map(lambda d: run_dataset(d, opts, ledger), datasets))
    print("total spent: $%.4f" % ledger.spent)


if __name__ == "__main__":
    main()
