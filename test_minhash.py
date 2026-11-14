"""Tests for minhash.py, policy.py, content.py, run.py and plots.py.

Run: .venv/bin/python -m unittest test_minhash -v
Uses only the public env/ side of the datasets (never hidden/), plus verify.py where noted.
"""

import random
import unittest

import numpy as np

import minhash


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


if __name__ == "__main__":
    unittest.main()
