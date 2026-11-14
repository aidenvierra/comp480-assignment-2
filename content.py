"""Content-level reuse: show a near-duplicate tool output as a reference plus its exact diff.

Within one agent trajectory, every long tool output is remembered. When a new output is a
near-copy of an earlier one (MinHash with numbers masked, so files that differ only in
values still match), the agent gets

    [reuse] same as inventory/web-01.txt except (3 of 95 lines differ):
    L12: uptime_days = 41

Changed lines are shown in their new form (the old form is in the referenced output, which
the agent has already read); added lines are marked `L<n>+`, removed lines `L<n>-`. The
diff runs on the raw text, so every changed number, id and value is shown.

A reply that quotes an earlier message (`> ` lines) gets a second, finer fold: a quoted
block whose text appears verbatim in an earlier output becomes one reference line.

Two rules keep references safe to follow: they only ever point at an output the agent was
shown in full (never at another fold), and a request the agent repeats gets its full output,
since asking again means the earlier form was not enough.

When neither fold is clearly smaller than the original, the original is returned unchanged.
"""

import difflib
import hashlib
import re
import time
from collections import namedtuple

import minhash

Seen = namedtuple("Seen", "ref lines signature plain")
WRAPPED = ("read_file", "bash", "grep")
MIN_QUOTE_CHARS = 200
_NUMBERED = re.compile(r"^\d+\t")
_QUOTE_MARKS = re.compile(r"^(>\s?)+")
_QUOTED_RUN = re.compile(r"(?:^(?:\d+\t)?>.*\n?)+", re.M)


class ContentCache:
    """Remembers one trajectory's tool outputs and folds near-duplicates."""

    def __init__(self, threshold=0.8, min_chars=400, max_ratio=0.5):
        self.threshold, self.min_chars, self.max_ratio = threshold, min_chars, max_ratio
        self.lsh = minhash.LSHIndex()
        self.seen = []        # outputs shown in full (the only fold targets), by LSH key
        self.exact = {}       # sha1 of an output shown in full -> its ref
        self.requested = set()
        self.hits = self.chars_in = self.chars_out = 0
        self.seconds = 0.0

    def process(self, tool, args, output):
        """Return `output`, or a shorter [reuse] form of it when that is safe and worth it."""
        start = time.perf_counter()
        result = self._process(tool, args, output)
        self.seconds += time.perf_counter() - start
        self.chars_in += len(output)
        self.chars_out += len(result)
        self.hits += result != output
        return result

    def _process(self, tool, args, output):
        if len(output) < self.min_chars or output.startswith(("ERROR:", "BLOCKED:")):
            return output
        ref = _ref(tool, args)
        if ref in self.requested:  # asked again on purpose: a pointer back would loop
            return output
        self.requested.add(ref)
        digest = hashlib.sha1(output.encode()).hexdigest()
        if digest in self.exact:
            return "[reuse] identical to %s" % self.exact[digest]

        lines = [_NUMBERED.sub("", line) for line in output.splitlines()]
        sig = minhash.signature("\n".join(lines), mask_numbers=True)
        scored = [(minhash.estimate(sig, self.seen[k].signature), k) for k in self.lsh.query(sig)]
        similarity, key = max(scored, default=(0.0, None))
        if similarity >= self.threshold:
            folded = _fold(self.seen[key], lines)
            if len(folded) <= self.max_ratio * len(output):
                return folded

        result = _QUOTED_RUN.sub(lambda match: self._fold_quote(match.group(0)), output)
        if result == output:  # only text the agent saw in full may be referred to later
            self.exact[digest] = ref
            self.lsh.add(len(self.seen), sig)
            self.seen.append(Seen(ref, lines, sig, _plain(lines)))
        return result

    def _fold_quote(self, block):
        """One reference line for a quoted block that an earlier output contains verbatim."""
        text = _plain(block.splitlines())
        if len(text) < MIN_QUOTE_CHARS:
            return block
        source = next((s for s in self.seen if text in s.plain), None)
        if source is None:
            return block
        number = _NUMBERED.match(block)
        return "%s> [reuse] quoted text of %s (%d lines)\n" % (
            number.group(0) if number else "", source.ref, len(block.splitlines()))


def _fold(old, new_lines):
    """The new lines as the earlier output plus line-numbered differences."""
    diff = []
    matcher = difflib.SequenceMatcher(None, old.lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        paired = min(i2 - i1, j2 - j1)
        for k in range(paired):
            diff.append("L%d: %s" % (j1 + k + 1, new_lines[j1 + k]))
        for j in range(j1 + paired, j2):
            diff.append("L%d+: %s" % (j + 1, new_lines[j]))
        for i in range(i1 + paired, i2):
            diff.append("L%d-: %s" % (j2 + 1, old.lines[i]))
    header = "[reuse] same as %s except (%d of %d lines differ):" % (old.ref, len(diff), len(new_lines))
    return "\n".join([header] + diff)


def _plain(lines):
    """Text with line numbers, quote marks and whitespace runs removed, for verbatim search."""
    return " ".join(" ".join(_QUOTE_MARKS.sub("", _NUMBERED.sub("", line)) for line in lines).split())


def _ref(tool, args):
    """How the agent can recognise the earlier output being referred to."""
    if tool == "read_file":
        return str(args.get("path"))
    if tool == "grep":
        return "grep %s in %s" % (args.get("regex"), args.get("pattern", "*"))
    return "%s: %s" % (tool, str(args.get("command", args))[:80])


def wrap_tools(workdir, cache):
    """Spartacus's read_file, bash and grep, with outputs passed through `cache`.

    Passed to Harness(extra_tools=...), they replace the built-ins of the same name.
    Imported here so the cache itself does not depend on Spartacus being on the path.
    """
    from spartacus import tools

    def wrap(core):
        return tools.Tool(core.name, core.spec,
                          lambda **kwargs: cache.process(core.name, kwargs, core.run(**kwargs)))

    return [wrap(t) for t in tools.core_tools(workdir) if t.name in WRAPPED]
