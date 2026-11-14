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

import os
import re
from collections import namedtuple

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
