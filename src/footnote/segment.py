from __future__ import annotations

import re

from footnote.models import Claim

__all__ = ["segment", "split_sentences"]

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
    match = _WORD_BEFORE.search(text[:dot_index])
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha():
        return True
    return bool(word[-1].isdigit() and dot_index + 1 < len(text) and text[dot_index + 1].isdigit())


def split_sentences(text: str, *, offset: int = 0) -> list[tuple[int, int]]:
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
            flush()
            units.append((start + bullet.end(), end))
            continue
        paragraph.append((start + len(line) - len(line.lstrip()), end))
    flush()
    return [(s, e) for s, e in units if e > s]


def _is_claim(text: str, min_words: int) -> bool:
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
    return any(any(ch.isalpha() for ch in w) for w in words)


def segment(text: str, *, min_words: int = 4) -> list[Claim]:
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
