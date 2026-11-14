"""Task-level reuse: answer an incoming task from a verified one, but only when it is safe.

MinHash finds earlier tasks whose text is similar. Similarity alone is not enough: a hard
negative reads almost the same and has a different answer. So a candidate is reused only
when all three hold:
  1. estimated Jaccard >= threshold,
  2. its validation key equals the task's (the fields each dataset's INSTRUCTIONS.md says
     must match, normalised the same way),
  3. no answer-deciding word (a ticket's cue, a memo's status word) appears in one text but
     not the other.
Everything here reads only env/ (history.jsonl and INSTRUCTIONS.md), never hidden/.
"""

import copy
import json
import os
import re
from collections import namedtuple

import minhash

DEFAULT_THRESHOLD = 0.5
MAX_THRESHOLD = 0.9
MARGIN = 0.05

# ---------------------------------------------------------------------------
# normalisation, following the rules published in each INSTRUCTIONS.md
# ---------------------------------------------------------------------------


def norm_text(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def norm_code(code):
    """E-1042, e1042 and 'E 1042' are one code."""
    return re.sub(r"[^a-z0-9]", "", str(code).lower())


def norm_frame(frame):
    """lattice/net/http.py::fetch and lattice.net.http.fetch are one frame; line numbers drop."""
    frame = re.sub(r"\.py::", ".", str(frame).strip().lower())
    frame = frame.replace("/", ".").replace("::", ".")
    return re.sub(r":\d+$", "", frame)


def norm_version(version):
    """v3.1.2 and 3.1.4 are both series 3.1."""
    match = re.search(r"(\d+)\.(\d+)", str(version))
    return "%s.%s" % match.groups() if match else norm_text(version)


def norm_doi(doi):
    doi = str(doi or "").strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    return re.sub(r"^doi:\s*", "", doi).rstrip(". ")


def norm_title(title):
    return norm_text(title)


def norm_author(name):
    """'Priya Raman', 'Raman, P.' and 'P Raman' -> ('raman', 'p')."""
    name = str(name).strip()
    if "," in name:
        surname, given = [part.strip() for part in name.split(",", 1)]
    else:
        parts = name.split()
        surname, given = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (name, "")
    return norm_text(surname), norm_text(given)[:1]


# ---------------------------------------------------------------------------
# answer-deciding words, parsed from INSTRUCTIONS.md so held-out variants work too
# ---------------------------------------------------------------------------
_CUE = re.compile(r"story mentions \*([^*]+)\*")
_STATUS = re.compile(r"^- `([a-z-]+)`:.*Status words: (.+?)\.?$", re.M)
_DOC_ID = re.compile(r"\b(CHG|INC|POL)-\d{3,4}\b|\bREL-\d+\.\d+\b", re.I)
_FAMILY = {"CHG": "change", "INC": "incident", "POL": "policy", "REL": "release"}


def critical_terms(instructions_md):
    """{'cues': [...], 'status': {category: [words]}} found in the instructions."""
    status = {}
    for category, words in _STATUS.findall(instructions_md):
        status[category] = [w.strip() for w in words.split(",") if w.strip() and w.strip() != "(none)"]
    return {"cues": sorted(set(_CUE.findall(instructions_md))), "status": status}


def _mentions(text, word):
    """Does normalised `text` contain `word` in any inflection (approve/approved/approval)?"""
    stem = re.sub(r"(ed|d|s)$", "", norm_text(word)) if " " not in word else norm_text(word)
    return re.search(r"\b" + re.escape(stem), text) is not None


def decisive_words(text, terms):
    """The cue and status words present in a text."""
    text = norm_text(text)
    words = terms["cues"] + [w for ws in terms["status"].values() for w in ws]
    return {w for w in words if _mentions(text, w)}


def _docs_key(record, terms):
    """(identifier, category): the family comes from the id, the class from the status word."""
    match = _DOC_ID.search(record.get("title", ""))
    if not match:
        return None
    family, body = _FAMILY[match.group(0)[:3].upper()], norm_text(record.get("body", ""))
    for category, words in terms["status"].items():
        if category.startswith(family) and (not words or any(_mentions(body, w) for w in words)):
            return match.group(0).upper(), category
    return None


def _citation_key(record, terms):
    c = record["citation"]
    return (norm_doi(c.get("doi")), norm_title(c.get("title")),
            tuple(norm_author(a) for a in c.get("authors") or []),
            norm_code(c.get("venue")), str(c.get("year")))


# ---------------------------------------------------------------------------
# per-dataset rules: which text to compare, which key must match, the answer to reuse
# ---------------------------------------------------------------------------
Rule = namedtuple("Rule", "text key answer")


def _fields(*names):
    return lambda record: "\n".join(str(record.get(n) or "") for n in names)


_corpus = Rule(_fields("subject", "body"),
               lambda r, t: (norm_code(r["scenario"]), str(r["kind"]).lower(), norm_code(r.get("target") or "")),
               lambda r: r["answer"])

RULES = {
    "tickets": Rule(_fields("subject", "body"),
                    lambda r, t: (norm_code(r["product"]), norm_code(r["error_code"]), norm_code(r["platform"])),
                    lambda r: {"resolution_id": r["resolution_id"]}),
    "bugs": Rule(_fields("title", "body", "stacktrace"),
                 lambda r, t: (str(r["exception"]).lower(), norm_frame(r["top_frame"]), norm_version(r["version"])),
                 lambda r: {"component": r["component"], "root_cause": r["root_cause"]}),
    "docs": Rule(_fields("title", "body"), _docs_key,
                 lambda r: {"category": r["category"], "duplicate_of": r["hid"]}),
    "citations": Rule(_fields("raw"), _citation_key, lambda r: r["verdict"]),
    **{name: _corpus for name in ("hosts", "configs", "code", "threads", "revisions", "logs")},
}


def dataset_name(env_dir):
    """.../datasets/tickets-d25/env -> 'tickets' (ablation copies share their base rules)."""
    return os.path.basename(os.path.dirname(os.path.normpath(env_dir))).split("-")[0]


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------
Item = namedtuple("Item", "signature key words answer")
REASONS = ("no_candidate", "below_threshold", "key_mismatch", "critical_word")  # increasing closeness


class TaskPolicy:
    """A store of verified answers for one dataset, searched with MinHash LSH.

    `gated=False` turns off the key and critical-word checks (similarity only); it exists
    for the report's threshold sweep, never for real runs.
    """

    def __init__(self, env_dir, threshold=None, gated=True):
        self.dataset = dataset_name(env_dir)
        self.rule = RULES.get(self.dataset)  # unknown dataset: never reuse
        self.gated = gated
        self.terms = critical_terms(_read(os.path.join(env_dir, "INSTRUCTIONS.md")))
        self.lsh = minhash.LSHIndex()
        self.items = {}
        self.last_info = {}
        if self.rule:
            for record in _jsonl(os.path.join(env_dir, "history.jsonl")):
                self._add(record["hid"], record, self.rule.answer(record))
        self.threshold = self._calibrate() if threshold is None else threshold

    @property
    def store_size(self):
        return len(self.items)

    def lookup(self, task):
        """(answer, info) when a verified task is safely the same item, else None."""
        self.last_info = {"reason": "unknown_dataset", "candidates": 0}
        if not self.rule:
            return None
        text = self.rule.text(task)
        sig = minhash.signature(text)
        key = self.rule.key(task, self.terms)
        words = decisive_words(text, self.terms)
        candidates = self.lsh.query(sig)
        best, reason = None, "no_candidate"
        for source in candidates:
            item = self.items[source]
            similarity = minhash.estimate(sig, item.signature)
            if similarity < self.threshold:
                failed = "below_threshold"
            elif self.gated and (key is None or item.key != key):
                failed = "key_mismatch"
            elif self.gated and item.words != words:
                failed = "critical_word"
            else:
                if best is None or similarity > best[0]:
                    best = (similarity, source)
                continue
            reason = max(reason, failed, key=REASONS.index)
        self.last_info = {"reason": "reused" if best else reason, "candidates": len(candidates)}
        if best is None:
            return None
        similarity, source = best
        return copy.deepcopy(self.items[source].answer), {
            "source": source, "similarity": similarity, "candidates": len(candidates)}

    def index(self, task, answer):
        """Add a task whose answer passed verification."""
        if self.rule:
            self._add(task["tid"], task, answer)

    def _add(self, source, record, answer):
        text = self.rule.text(record)
        sig = minhash.signature(text)
        self.items[source] = Item(sig, self.rule.key(record, self.terms),
                                  decisive_words(text, self.terms), answer)
        self.lsh.add(source, sig)

    def _calibrate(self):
        """Raise the threshold above the most similar pair of history records that pass the
        key and critical-word checks yet have different answers: that band is demonstrably
        unsafe here."""
        worst = 0.0
        for source, item in self.items.items():
            for other in self.lsh.query(item.signature):
                peer = self.items[other]
                passes_gate = (peer.key, peer.words) == (item.key, item.words)
                if other != source and passes_gate and _verdict(peer.answer) != _verdict(item.answer):
                    worst = max(worst, minhash.estimate(item.signature, peer.signature))
        return max(DEFAULT_THRESHOLD, min(MAX_THRESHOLD, worst + MARGIN))


def _verdict(answer):
    """The answer without its pointer to the record itself (docs' duplicate_of)."""
    return {k: v for k, v in answer.items() if k != "duplicate_of"}


def _read(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _jsonl(path):
    return [json.loads(line) for line in _read(path).splitlines() if line.strip()]
