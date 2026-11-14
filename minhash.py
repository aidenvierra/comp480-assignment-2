"""MinHash over word 3-shingles, and an LSH index for finding near-duplicates fast.

A text becomes a set of 3-word shingles. Each of 128 hash functions maps every shingle
to a number and keeps the minimum; two signatures agree on a component with probability
equal to the Jaccard similarity of the sets, so the fraction of equal components
estimates it. LSH splits the signature into bands and buckets each band, so a query
only compares against items that share at least one band with it.
"""

import hashlib
import random
import re

import numpy as np

NUM_PERM = 128
SHINGLE = 3
SEED = 480
PRIME = (1 << 61) - 1      # Mersenne prime, larger than any 32-bit shingle hash
EMPTY = np.iinfo(np.uint64).max

# h_i(x) = (a_i * x + b_i) mod PRIME. With x, a, b < 2**32 the product never overflows uint64.
_rng = random.Random(SEED)
_A = np.array([_rng.randrange(1, 1 << 32) for _ in range(NUM_PERM)], dtype=np.uint64)
_B = np.array([_rng.randrange(0, 1 << 32) for _ in range(NUM_PERM)], dtype=np.uint64)

_WORD = re.compile(r"[a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_HEX = re.compile(r"[0-9a-f]+")


def tokens(text, mask_numbers=False):
    """Lower-case alphanumeric words. With mask_numbers, numbers and hex ids become "0"."""
    words = _WORD.findall(str(text).lower())
    if mask_numbers:
        words = ["0" if _HEX.fullmatch(w) and any(c.isdigit() for c in w) else _DIGITS.sub("0", w)
                 for w in words]
    return words


def shingles(text, mask_numbers=False):
    """The set of consecutive 3-word windows; a shorter non-empty text is one shingle."""
    words = tokens(text, mask_numbers)
    if 0 < len(words) < SHINGLE:
        return {" ".join(words)}
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def _hash32(shingle):
    return int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=4).digest(), "big")


def signature(text, mask_numbers=False):
    """128 minimum hash values of the text's shingles (all EMPTY for an empty text)."""
    items = shingles(text, mask_numbers)
    if not items:
        return np.full(NUM_PERM, EMPTY, dtype=np.uint64)
    x = np.array([_hash32(s) for s in items], dtype=np.uint64)
    return ((np.outer(x, _A) + _B) % PRIME).min(axis=0)


def estimate(sig_a, sig_b):
    """Estimated Jaccard similarity: the share of equal signature components."""
    return float(np.mean(sig_a == sig_b))


def exact_jaccard(text_a, text_b, mask_numbers=False):
    """True Jaccard of the shingle sets, for tests and the report."""
    a, b = shingles(text_a, mask_numbers), shingles(text_b, mask_numbers)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


class LSHIndex:
    """Banded LSH: items sharing any band of `rows` components become candidates.

    32 bands x 4 rows puts the 50 % detection point near Jaccard (1/32) ** (1/4) = 0.42.
    """

    def __init__(self, bands=32, rows=4):
        assert bands * rows == NUM_PERM
        self.bands, self.rows = bands, rows
        self.buckets = [{} for _ in range(bands)]
        self.size = 0

    def _keys(self, sig):
        return [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def add(self, key, sig):
        for bucket, band in zip(self.buckets, self._keys(sig)):
            bucket.setdefault(band, []).append(key)
        self.size += 1

    def query(self, sig):
        found = set()
        for bucket, band in zip(self.buckets, self._keys(sig)):
            found.update(bucket.get(band, ()))
        return found

    def __len__(self):
        return self.size
