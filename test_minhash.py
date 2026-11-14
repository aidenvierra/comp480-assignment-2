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

import minhash
import policy

HERE = os.path.dirname(os.path.abspath(__file__))
A2_DIR = os.environ.get("A2_DIR", os.path.join(HERE, "Assignment2_COMP_580"))
DATASETS = os.path.join(A2_DIR, "assignment2", "datasets")
ALL_DATASETS = ["tickets", "bugs", "docs", "citations",
                "hosts", "configs", "code", "threads", "revisions", "logs"]
sys.path.insert(0, A2_DIR)


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


if __name__ == "__main__":
    unittest.main()
