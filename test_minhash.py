"""Tests for minhash.py, policy.py, content.py, run.py and plots.py.

Run: .venv/bin/python -m unittest test_minhash -v
Uses only the public env/ side of the datasets (never hidden/), plus verify.py where noted.
"""

import json
import os
import random
import sys
import unittest

import numpy as np

import content
import minhash
import policy

HERE = os.path.dirname(os.path.abspath(__file__))
A2_DIR = os.environ.get("A2_DIR", os.path.join(HERE, "Assignment2_COMP_580"))
DATASETS = os.path.join(A2_DIR, "assignment2", "datasets")
ALL_DATASETS = ["tickets", "bugs", "docs", "citations",
                "hosts", "configs", "code", "threads", "revisions", "logs"]
SPARTACUS_DIR = os.environ.get("SPARTACUS_DIR", os.path.join(HERE, "Spartacus"))
sys.path[:0] = [A2_DIR, SPARTACUS_DIR]


def env(dataset):
    return os.path.join(DATASETS, dataset, "env")


def history(dataset):
    with open(os.path.join(env(dataset), "history.jsonl")) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def instructions(dataset):
    with open(os.path.join(env(dataset), "INSTRUCTIONS.md")) as handle:
        return handle.read()


def random_text(rng, vocab, n):
    return " ".join(rng.choice(vocab) for _ in range(n))


class TestMinHash(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)
        self.vocab = ["w%d" % i for i in range(200)]

    def test_identical_texts_estimate_one(self):
        text = random_text(self.rng, self.vocab, 300)
        self.assertEqual(minhash.estimate(minhash.signature(text), minhash.signature(text)), 1.0)

    def test_estimate_close_to_exact(self):
        words = random_text(self.rng, self.vocab, 400).split()
        a = " ".join(words[:340])
        b = " ".join(words[60:])
        exact = minhash.exact_jaccard(a, b)
        self.assertGreater(exact, 0.5)
        self.assertLess(abs(minhash.estimate(minhash.signature(a), minhash.signature(b)) - exact), 0.12)

    def test_deterministic(self):
        text = random_text(self.rng, self.vocab, 100)
        self.assertTrue(np.array_equal(minhash.signature(text), minhash.signature(text)))

    def test_mask_numbers(self):
        a, b = "used_pct=83 on / root", "used_pct=61 on / root"
        self.assertEqual(minhash.exact_jaccard(a, b, mask_numbers=True), 1.0)
        self.assertLess(minhash.exact_jaccard(a, b), 1.0)
        self.assertEqual(minhash.tokens("req 3fa9c2", mask_numbers=True), ["req", "0"])

    def test_short_and_empty(self):
        self.assertEqual(minhash.shingles("a b"), {"a b"})
        self.assertEqual(minhash.shingles(""), set())

    def test_lsh_finds_near_duplicate(self):
        index = minhash.LSHIndex()
        texts = [random_text(self.rng, self.vocab, 150) for _ in range(20)]
        for i, text in enumerate(texts):
            index.add(i, minhash.signature(text))
        words = texts[7].split()
        words[75] = "changed"
        self.assertIn(7, index.query(minhash.signature(" ".join(words))))
        other = " ".join("z%d" % i for i in range(150))
        self.assertEqual(index.query(minhash.signature(other)), set())
        self.assertEqual(len(index), 20)


class TestPolicyRules(unittest.TestCase):
    def test_normalisers_match_common(self):
        from assignment2.datasets import common
        for name, value in [("norm_code", "E-1042"), ("norm_frame", "lattice/net/http.py::fetch:12"),
                            ("norm_version", "v3.1.4 (build 7)"), ("norm_doi", "https://doi.org/10.5555/X."),
                            ("norm_title", "Link Prediction, Revisited!"), ("norm_author", "Raman, P.")]:
            self.assertEqual(getattr(policy, name)(value), getattr(common, name)(value), name)

    def test_every_history_record_has_key_and_answer(self):
        record_keys = {"tickets": {"resolution_id"}, "bugs": {"component", "root_cause"},
                       "docs": {"category", "duplicate_of"}, "citations": {"doi", "status", "wrong_fields"}}
        for dataset in ALL_DATASETS:
            rule, terms = policy.RULES[dataset], policy.critical_terms(instructions(dataset))
            for record in history(dataset):
                self.assertIsNotNone(rule.key(record, terms), (dataset, record["hid"]))
                answer = rule.answer(record)
                self.assertIsInstance(answer, dict)
                self.assertTrue(answer)
                if dataset in record_keys:
                    self.assertEqual(set(answer), record_keys[dataset], dataset)

    def test_docs_answer_has_duplicate_of_hid(self):
        record = history("docs")[0]
        self.assertEqual(policy.RULES["docs"].answer(record),
                         {"category": record["category"], "duplicate_of": record["hid"]})

    def test_docs_status_class_matches_history_category(self):
        terms = policy.critical_terms(instructions("docs"))
        for record in history("docs"):
            self.assertEqual(policy.RULES["docs"].key(record, terms)[1], record["category"], record["hid"])

    def test_critical_terms(self):
        cues = policy.critical_terms(instructions("tickets"))["cues"]
        for cue in ("clock", "office network", "password reset"):
            self.assertIn(cue, cues)
        status = policy.critical_terms(instructions("docs"))["status"]
        self.assertIn("approved", status["change-approved"])
        self.assertEqual(status["release-notes"], [])

    def test_dataset_name(self):
        self.assertEqual(policy.dataset_name("/x/datasets/tickets-d25/env"), "tickets")
        self.assertEqual(policy.dataset_name("/x/datasets/hosts/env/"), "hosts")


def as_task(record, tid="T-test", **changes):
    """A history record re-shaped as an incoming task, optionally with fields changed."""
    task = {k: v for k, v in record.items() if k != "hid"}
    task.update(tid=tid, **changes)
    return task


class TestTaskPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickets = policy.TaskPolicy(env("tickets"))

    def test_exact_copy_reused(self):
        record = history("tickets")[3]
        answer, info = self.tickets.lookup(as_task(record))
        self.assertEqual(answer, {"resolution_id": record["resolution_id"]})
        self.assertEqual(info["source"], record["hid"])
        self.assertEqual(info["similarity"], 1.0)

    def test_platform_flip_rejected(self):
        record = history("tickets")[3]
        other = "android" if record["platform"].lower() != "android" else "windows"
        self.assertIsNone(self.tickets.lookup(as_task(record, platform=other)))
        self.assertEqual(self.tickets.last_info["reason"], "key_mismatch")

    def test_cue_flip_rejected(self):
        record = next(r for r in history("tickets") if "office network" in r["body"])
        task = as_task(record, body=record["body"].replace("office network", "clock"))
        self.assertIsNone(self.tickets.lookup(task))
        self.assertIn(self.tickets.last_info["reason"], ("critical_word", "below_threshold"))

    def test_reworded_cue_flip_rejected(self):
        # The cue phrases are "plan change" and "duplicate batch"; neither appears verbatim.
        body = ("Every time I upload a file I get 'E-3307: storage quota exceeded'.\n\n%s\n\n"
                "I've already tried waiting a day, which changed nothing.")
        source = {"hid": "H-x", "product": "MeshMail", "error_code": "E-3307", "platform": "macos",
                  "subject": "E-3307: quota exceeded on macOS", "resolution_id": "R-21",
                  "body": body % "A duplicate import run doubled every record in the account."}
        task = dict(source, tid="T-x", body=body % "Yesterday's plan upgrade is not reflected yet.")
        terms = policy.critical_terms(instructions("tickets"))
        self.assertNotEqual(policy.decisive_words(source["body"], terms), policy.decisive_words(task["body"], terms))

    def test_corpus_scenario_mismatch_rejected(self):
        records = history("hosts")
        record = records[0]
        other = next(r["scenario"] for r in records if policy.norm_code(r["scenario"]) != policy.norm_code(record["scenario"]))
        hosts = policy.TaskPolicy(env("hosts"))
        self.assertIsNotNone(hosts.lookup(as_task(record)))
        self.assertIsNone(hosts.lookup(as_task(record, scenario=other)))

    def test_docs_status_flip_rejected(self):
        record = next(r for r in history("docs") if r["category"] == "change-approved" and "approved" in r["body"])
        flipped = as_task(record, body=record["body"].replace("approved", "rejected"))
        docs = policy.TaskPolicy(env("docs"))
        self.assertIsNotNone(docs.lookup(as_task(record)))
        self.assertIsNone(docs.lookup(flipped))

    def test_index_then_reuse(self):
        tickets = policy.TaskPolicy(env("tickets"))
        task = as_task(history("tickets")[0], tid="T-new", product="BrandNewProduct")
        self.assertIsNone(tickets.lookup(task))
        tickets.index(task, {"resolution_id": "R-01"})
        answer, info = tickets.lookup(dict(task, tid="T-newer"))
        self.assertEqual(answer, {"resolution_id": "R-01"})
        self.assertEqual(info["source"], "T-new")

    def test_unknown_dataset_never_reuses(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mystery = os.path.join(tmp, "mystery", "env")
            os.makedirs(mystery)
            for name in ("history.jsonl", "INSTRUCTIONS.md"):
                shutil.copy(os.path.join(env("tickets"), name), mystery)
            self.assertIsNone(policy.TaskPolicy(mystery).lookup(as_task(history("tickets")[0])))

    def test_threshold_bounds(self):
        for dataset in ALL_DATASETS:
            threshold = policy.TaskPolicy(env(dataset)).threshold
            self.assertTrue(0.5 <= threshold <= 0.9, (dataset, threshold))

    def test_ungated_mode_ignores_key(self):
        record = history("tickets")[3]
        ungated = policy.TaskPolicy(env("tickets"), gated=False)
        other = "android" if record["platform"].lower() != "android" else "windows"
        self.assertIsNotNone(ungated.lookup(as_task(record, platform=other)))


def numbered(text):
    """Text formatted the way Spartacus's read_file returns it."""
    return "\n".join("%d\t%s" % (n, line) for n, line in enumerate(text.splitlines(), 1))


class TestContent(unittest.TestCase):
    def setUp(self):
        folder = os.path.join(DATASETS, "hosts", "env", "scenarios", "FLT-1000", "inventory")
        names = sorted(os.listdir(folder))[:2]
        self.a, self.b = (open(os.path.join(folder, n)).read() for n in names)
        self.cache = content.ContentCache()

    def read(self, path, text):
        return self.cache.process("read_file", {"path": path}, numbered(text))

    def test_short_output_passthrough(self):
        self.assertEqual(self.cache.process("bash", {"command": "ls"}, "x" * 100), "x" * 100)

    def test_exact_repeat(self):
        self.read("a.txt", self.a)
        self.assertEqual(self.read("a.txt", self.a), "[reuse] identical to a.txt")

    def test_near_duplicate_folded_keeps_numbers(self):
        self.read("a.txt", self.a)
        folded = self.read("b.txt", self.b)
        self.assertTrue(folded.startswith("[reuse] same as a.txt except"), folded[:80])
        self.assertLessEqual(len(folded), 0.5 * len(numbered(self.b)))
        a_lines = set(self.a.splitlines())
        for line in self.b.splitlines():
            if line not in a_lines:
                self.assertIn(line, folded)
        self.assertEqual(self.cache.hits, 1)

    def test_unrelated_not_folded(self):
        self.read("a.txt", self.a)
        unrelated = "\n".join("line %s about something else entirely" % w for w in "abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(self.read("u.txt", unrelated), numbered(unrelated))

    def test_insufficient_savings_returns_original(self):
        self.read("a.txt", self.a)
        lines = self.a.splitlines()
        changed = "\n".join(line + " changed" if i % 3 == 0 else line for i, line in enumerate(lines))
        self.assertEqual(self.read("c.txt", changed), numbered(changed))

    def test_quoted_reply_folded_to_reference(self):
        inbox = os.path.join(DATASETS, "threads", "env", "scenarios", "MBX-1000", "inbox")
        names = sorted(os.listdir(inbox))
        texts = {n: open(os.path.join(inbox, n)).read() for n in names}
        reply = next(n for n in names if "\n> " in texts[n] and len(texts[n]) > 600)
        for name in names[:names.index(reply)]:
            self.read(name, texts[name])
        result = self.read(reply, texts[reply])
        self.assertIn("[reuse] quoted text of", result)
        self.assertLess(len(result), len(numbered(texts[reply])))
        own_lines = [l for l in texts[reply].splitlines() if l.strip() and not l.startswith(">")]
        for line in own_lines:
            self.assertIn(line, result)

    def test_wrap_tools_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (("a.txt", self.a), ("b.txt", self.b)):
                with open(os.path.join(tmp, name), "w") as handle:
                    handle.write(text)
            wrapped = {tool.name: tool for tool in content.wrap_tools(tmp, self.cache)}
            self.assertEqual(set(wrapped), {"read_file", "bash", "grep"})
            self.assertEqual(wrapped["read_file"].run(path="a.txt"), numbered(self.a))
            self.assertTrue(wrapped["read_file"].run(path="b.txt").startswith("[reuse] same as a.txt"))


def text_only_model(model, system, messages, tools=None):
    """A model that answers in prose and never writes answer.json."""
    return {"text": "I think it is R-01.", "tool_calls": [], "usage": {"input": 500, "output": 10}}


class TestRunner(unittest.TestCase):
    PRICES = (0.10, 0.40)

    def setUp(self):
        import tempfile
        import run
        self.run = run
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, run.provider, "complete", run.provider.complete)
        run.provider.complete = run.fake_model

    def test_openrouter_wraps_tool_output_as_json(self):
        # OpenRouter hands Gemini only the JSON object it finds inside raw tool text.
        self.addCleanup(setattr, self.run.provider, "_openai_wire", self.run.provider._openai_wire)
        self.addCleanup(self.run.models.PROVIDERS["openai"].update, dict(self.run.models.PROVIDERS["openai"]))
        self.run.use_openrouter()
        self.run.use_openrouter()  # idempotent: never double-wrapped
        text = 'Write answer.json:\n{"resolution_id": "R-07"}'
        wire = self.run.provider._openai_wire("sys", [
            {"role": "user", "text": "go"},
            {"role": "assistant", "text": "", "tool_calls": [{"name": "read_file", "args": {}}]},
            {"role": "tool", "name": "read_file", "text": text}])
        self.assertEqual(json.loads(wire[-1]["content"]), {"result": text})
        self.assertEqual(self.run.models.PROVIDERS["openai"]["base"], self.run.OPENROUTER)

    def test_openrouter_retries_malformed_function_call(self):
        self.addCleanup(setattr, self.run.provider, "_post", self.run.provider._post)
        self.addCleanup(setattr, self.run.provider, "_openai_wire", self.run.provider._openai_wire)
        self.addCleanup(self.run.models.PROVIDERS["openai"].update, dict(self.run.models.PROVIDERS["openai"]))
        replies = [{"choices": [{"finish_reason": "error", "native_finish_reason": "MALFORMED_FUNCTION_CALL",
                                 "message": {"content": None}}]},
                   {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}]
        bodies = []
        self.run.provider._post = lambda url, body, headers, retries=5: bodies.append(body) or replies.pop(0)
        self.run.use_openrouter()
        reply = self.run.provider._post("u", {"tools": [{}]}, {})
        self.assertEqual(reply["choices"][0]["message"]["content"], "ok")
        self.assertNotIn("tool_choice", bodies[0])
        self.assertEqual(bodies[1]["tool_choice"], "required")

    def test_load_tasks_sorted_prefix(self):
        tasks = self.run.load_tasks("tickets", 5)
        every = sorted(t["seq"] for t in self.run.load_tasks("tickets", None))
        self.assertEqual([t["seq"] for t in tasks], every[:5])

    def test_load_tasks_seeded_sample(self):
        sample = self.run.load_tasks("tickets", 8, seed=480)
        self.assertEqual(len(sample), 8)
        self.assertEqual([t["tid"] for t in sample], [t["tid"] for t in self.run.load_tasks("tickets", 8, seed=480)])
        self.assertEqual([t["seq"] for t in sample], sorted(t["seq"] for t in sample))
        self.assertNotEqual(sample, self.run.load_tasks("tickets", 8))

    def test_build_workdir_corpus_unnormalised_id(self):
        task = dict(self.run.load_tasks("hosts", 1)[0], scenario="flt1000")
        work = self.run.build_workdir("hosts", task, os.path.join(self.tmp.name, "w"))
        files = set(os.listdir(work))
        self.assertTrue({"task.json", "INSTRUCTIONS.md", "history.jsonl", "inventory"} <= files)
        self.assertFalse(files & {"incoming", "scenarios", "hidden"})
        self.assertTrue(os.listdir(os.path.join(work, "inventory")))

    def test_build_workdir_citations_has_db(self):
        task = self.run.load_tasks("citations", 1)[0]
        work = self.run.build_workdir("citations", task, os.path.join(self.tmp.name, "w"))
        self.assertIn("db.json", os.listdir(work))

    def test_fake_run_counts(self):
        task = self.run.load_tasks("tickets", 1)[0]
        row = self.run.run_task("tickets", task, "baseline", None, self.tmp.name, self.PRICES)
        self.assertEqual((row["model_requests"], row["tool_calls"], row["input_tokens"]), (2, 1, 2000))
        self.assertIs(row["passed"], False)
        self.assertIsNone(row["error"])
        self.assertEqual(row["answer"], {"fake": True})

    def test_missing_answer_is_failed_row(self):
        self.run.provider.complete = text_only_model
        task = self.run.load_tasks("tickets", 1)[0]
        row = self.run.run_task("tickets", task, "baseline", None, self.tmp.name, self.PRICES)
        self.assertIs(row["passed"], False)
        self.assertIsNone(row["answer"])

    def test_task_reuse_skips_agent(self):
        task_policy = policy.TaskPolicy(env("tickets"))
        task = next(t for t in self.run.load_tasks("tickets", None) if task_policy.lookup(t))
        row = self.run.run_task("tickets", task, "task", task_policy, self.tmp.name, self.PRICES)
        self.assertTrue(row["reused"])
        self.assertEqual(row["model_requests"], 0)
        self.assertTrue(row["passed"])

    def test_ledger_and_resume(self):
        out = os.path.join(self.tmp.name, "results")
        base = ["--config", "baseline", "--datasets", "tickets", "--n", "3", "--fake", "--out", out]
        self.run.main(base + ["--cap", "0.000001"])
        log = os.path.join(out, "baseline", "tickets.jsonl")
        self.assertEqual(len(self.run.read_jsonl(log)), 1)
        self.run.main(base + ["--cap", "100"])
        tids = [row["tid"] for row in self.run.read_jsonl(log)]
        self.assertEqual(len(tids), 3)
        self.assertEqual(len(set(tids)), 3)
        self.assertGreater(self.run.Ledger(os.path.join(out, "spend.json")).spent, 0)


class TestPlots(unittest.TestCase):
    def test_sweep_rows(self):
        import plots
        rows = plots.sweep(["tickets"], [0.5, 1.0])
        self.assertEqual(len(rows), 4)
        gated = [r for r in rows if r["mode"] == "gated"]
        self.assertTrue(all(r["false_reuses"] == 0 for r in gated))
        self.assertGreaterEqual(next(r for r in gated if r["threshold"] == 1.0)["reused"], 1)

    def test_report_on_fake_results(self):
        import tempfile
        import plots
        import run
        self.addCleanup(setattr, run.provider, "complete", run.provider.complete)
        with tempfile.TemporaryDirectory() as tmp:
            results, figures = os.path.join(tmp, "results"), os.path.join(tmp, "figures")
            for config in run.CONFIGS:
                run.main(["--config", config, "--datasets", "tickets,hosts", "--n", "3",
                          "--fake", "--out", results])
            sweep_csv = os.path.join(tmp, "sweep.csv")
            plots.write_csv(plots.sweep(["tickets", "hosts"], [0.5, 1.0]), sweep_csv)
            plots.report(results, figures, sweep_csv)
            summary = plots.read_csv(os.path.join(figures, "summary.csv"))
            self.assertEqual(len(summary), 4 * 3)  # 4 configs x (2 datasets + overall)
            for name in plots.FIGURES:
                self.assertTrue(os.path.getsize(os.path.join(figures, name)) > 0, name)


if __name__ == "__main__":
    unittest.main()
