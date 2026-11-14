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


if __name__ == "__main__":
    unittest.main()
