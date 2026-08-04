"""Carving text into checkable claims.

Verification happens per claim, not per document, so this module decides what a
claim is. Two rules drive the design.

First, offsets are preserved. Every claim records where it sits in the original
answer, which is what lets the HTML report highlight the sentence in place instead
of reprinting a detached copy of it.

Second, not every sentence is a claim. Headings, questions, code, table rules and
bare fragments assert nothing, and scoring them produces noise that drags the
faithfulness number around without telling anyone anything useful. They are skipped
rather than scored.
"""

from __future__ import annotations

import re

from footnote.models import Claim

__all__ = ["segment", "split_sentences"]

# Words that end in a full stop without ending a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "rev",
        "hon",
        "gen",
        "col",
        "lt",
        "sgt",
        "capt",
        "inc",
        "ltd",
        "co",
        "corp",
        "plc",
        "llc",
        "dept",
        "univ",
        "inst",
        "assn",
        "bros",
        "vs",
        "etc",
        "eg",
        "ie",
        "cf",
        "al",
        "ca",
        "approx",
        "est",
        "fig",
        "figs",
        "no",
        "nos",
        "vol",
        "vols",
        "pp",
        "ch",
        "sec",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
        "u.s",
        "u.k",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
    ]
)

_SENTENCE_END = re.compile(r"([.!?]+)([\"'’”)\]]*)(\s+|$)")
_WORD_BEFORE = re.compile(r"([A-Za-z0-9.'’-]+)$")

_BULLET = re.compile(r"^\s*(?:[-*+•]|\(?\d{1,2}[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s+")
_RULE = re.compile(r"^\s*(?:[-*_=]{3,}|\|[\s|:-]*\|)\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_URL_ONLY = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$")


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """True when the full stop at ``dot_index`` belongs to an abbreviation."""
    match = _WORD_BEFORE.search(text[:dot_index])
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # A single letter is an initial ("J. Smith") or part of "e.g." style forms.
    if len(word) == 1 and word.isalpha():
        return True
    # A decimal point: digits on both sides of the stop.
    return bool(word[-1].isdigit() and dot_index + 1 < len(text) and text[dot_index + 1].isdigit())


def split_sentences(text: str, *, offset: int = 0) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offset pairs for each sentence in ``text``.

    Abbreviation- and decimal-aware, so "grew 3.5% in Q1 vs. Q4" stays one sentence.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        dot = match.start()
        if _is_abbreviation(text, dot):
            continue
        end = match.end(2)
        chunk = text[cursor:end]
        if chunk.strip():
            spans.append((offset + cursor + len(chunk) - len(chunk.lstrip()), offset + end))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        spans.append((offset + cursor + lead, offset + len(text.rstrip())))
    return spans


def _blocks(text: str) -> list[tuple[int, int]]:
    """Split into line-level units, dropping fenced code and structural lines."""
    units: list[tuple[int, int]] = []
    in_fence = False
    paragraph: list[tuple[int, int]] = []

    def flush() -> None:
        if paragraph:
            units.append((paragraph[0][0], paragraph[-1][1]))
            paragraph.clear()

    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        cursor += len(line)
        stripped = line.strip()
        end = start + len(line.rstrip())

        if _FENCE.match(line):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped:
            flush()
            continue
        if _HEADING.match(line) or _RULE.match(line) or _TABLE_ROW.match(line):
            flush()
            continue
        bullet = _BULLET.match(line)
        if bullet:
            # A bullet is its own unit: list items rarely continue each other.
            flush()
            units.append((start + bullet.end(), end))
            continue
        paragraph.append((start + len(line) - len(line.lstrip()), end))
    flush()
    return [(s, e) for s, e in units if e > s]


def _is_claim(text: str, min_words: int) -> bool:
    """Filter out anything that asserts nothing."""
    stripped = text.strip()
    if len(stripped) < 12:
        return False
    if stripped.endswith("?"):
        return False
    if _URL_ONLY.match(stripped):
        return False
    words = re.findall(r"[A-Za-z0-9']+", stripped)
    if len(words) < min_words:
        return False
    # Needs at least one letter-bearing word; a row of figures is not a claim.
    return any(any(ch.isalpha() for ch in w) for w in words)


def segment(text: str, *, min_words: int = 4) -> list[Claim]:
    """Split ``text`` into claims, each carrying its offset in the original string.

    ``min_words`` sets the shortest run of words treated as an assertion. Four is a
    reasonable floor: "Revenue rose sharply" is checkable, "Yes" is not.
    """
    claims: list[Claim] = []
    for block_start, block_end in _blocks(text):
        body = text[block_start:block_end]
        for start, end in split_sentences(body, offset=block_start):
            sentence = text[start:end]
            if not _is_claim(sentence, min_words):
                continue
            claims.append(
                Claim(id=f"c{len(claims) + 1}", text=sentence.strip(), start=start, end=end)
            )
    return claims
