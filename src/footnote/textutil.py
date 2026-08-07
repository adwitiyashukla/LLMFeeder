from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "NumericMention",
    "Token",
    "content_tokens",
    "extract_numbers",
    "has_negation",
    "numbers_agree",
    "stem",
    "tokenize",
]

STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "it's",
        "they",
        "them",
        "their",
        "there",
        "here",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "i",
        "me",
        "my",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "done",
        "about",
        "into",
        "over",
        "under",
        "between",
        "out",
        "up",
        "down",
        "again",
        "further",
        "once",
        "also",
        "very",
        "just",
        "too",
        "so",
        "such",
        "own",
        "same",
        "other",
        "another",
        "each",
        "both",
        "any",
        "all",
        "some",
        "most",
        "more",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
    ]
)

NEGATIONS: frozenset[str] = frozenset(
    [
        "no",
        "not",
        "never",
        "none",
        "nor",
        "neither",
        "nothing",
        "nobody",
        "nowhere",
        "cannot",
        "can't",
        "won't",
        "wouldn't",
        "shouldn't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "without",
        "lacks",
        "lacked",
        "lacking",
        "fails",
        "failed",
        "failing",
        "declined",
        "denied",
        "denies",
        "refused",
        "unable",
        "absent",
        "excluded",
        "rejected",
    ]
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-.][A-Za-z0-9]+)*")

_WORD_NUMBERS: dict[str, float] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "dozen": 12,
}

_SUFFIXES: tuple[str, ...] = (
    "ational",
    "iveness",
    "fulness",
    "ousness",
    "ization",
    "isation",
    "ations",
    "ition",
    "ement",
    "ments",
    "ingly",
    "edly",
    "ation",
    "ment",
    "ness",
    "tion",
    "sion",
    "ence",
    "ance",
    "able",
    "ible",
    "ing",
    "ies",
    "ied",
    "est",
    "ers",
    "ed",
    "es",
    "ly",
    "er",
    "s",
)


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    norm: str
    start: int
    end: int

    @property
    def is_stopword(self) -> bool:
        return self.norm in STOPWORDS

    @property
    def is_numeric(self) -> bool:
        return any(ch.isdigit() for ch in self.norm) or self.text.lower() in _WORD_NUMBERS


def stem(word: str) -> str:
    lowered = word.lower()
    if len(lowered) <= 3 or any(ch.isdigit() for ch in lowered):
        return lowered
    for suffix in _SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
            root = lowered[: -len(suffix)]
            if len(root) > 3 and root[-1] == root[-2] and root[-1] not in "aeiou":
                root = root[:-1]
            return root
    return lowered


def tokenize(text: str, *, offset: int = 0) -> list[Token]:
    return [
        Token(
            text=m.group(0),
            norm=stem(m.group(0)),
            start=offset + m.start(),
            end=offset + m.end(),
        )
        for m in _TOKEN_RE.finditer(text)
    ]


def content_tokens(text: str, *, offset: int = 0) -> list[Token]:
    return [
        t
        for t in tokenize(text, offset=offset)
        if t.text.lower() in NEGATIONS or (not t.is_stopword and len(t.text) > 1)
    ]


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(t.group(0).lower() in NEGATIONS for t in _TOKEN_RE.finditer(lowered))


_MAGNITUDES: dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "m": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "t": 1e12,
}

_NUMBER_RE = re.compile(
    r"""
    (?P<currency>[$£€¥])?\s?
    (?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s?
    (?P<magnitude>hundred|thousand|million|billion|trillion|[kKmMbBtT]n?\b)?
    \s?
    (?P<unit>%|percent|percentage\ points?|bps|x\b)?
    """,
    re.VERBOSE,
)

_YEAR_RE = re.compile(r"^(1[89]\d{2}|20\d{2})$")

_WORD_NUMBER_RE = re.compile(
    r"\b(?P<word>" + "|".join(_WORD_NUMBERS) + r")\b"
    r"(?:\s+(?P<magnitude>hundred|thousand|million|billion|trillion))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class NumericMention:
    raw: str
    value: float
    unit: str | None
    start: int
    end: int
    is_year: bool = False
    is_ordinal: bool = False
    before: str | None = None
    after: str | None = None

    @property
    def is_quantity(self) -> bool:
        return not (self.is_year or self.is_ordinal)

    def measures_same_thing(self, other: NumericMention) -> bool:
        if self.unit and other.unit and self.unit != other.unit:
            return False
        if self.unit and other.unit and self.unit == other.unit:
            return True
        anchors = {a for a in (self.before, self.after) if a}
        theirs = {a for a in (other.before, other.after) if a}
        return bool(anchors & theirs)

    def close_to(self, other: NumericMention, *, rel_tol: float = 0.005) -> bool:
        if self.unit and other.unit and self.unit != other.unit:
            return False
        if self.value == other.value:
            return True
        scale = max(abs(self.value), abs(other.value), 1e-9)
        return abs(self.value - other.value) / scale <= rel_tol


def _normalise_unit(raw: str | None) -> str | None:
    if not raw:
        return None
    lowered = raw.strip().lower()
    if lowered in {"%", "percent"}:
        return "%"
    if lowered.startswith("percentage point"):
        return "pp"
    if lowered == "bps":
        return "bps"
    if lowered == "x":
        return "x"
    return lowered


def extract_numbers(text: str, *, offset: int = 0) -> list[NumericMention]:
    out: list[NumericMention] = []
    for m in _NUMBER_RE.finditer(text):
        digits = m.group("value")
        if not digits:
            continue
        try:
            value = float(digits.replace(",", ""))
        except ValueError:  # pragma: no cover - the pattern guarantees a number
            continue

        magnitude = (m.group("magnitude") or "").strip().lower()
        if magnitude:
            value *= _MAGNITUDES.get(magnitude, 1.0)

        unit = _normalise_unit(m.group("unit"))
        if unit is None and m.group("currency"):
            unit = "currency"

        raw = m.group(0).strip()
        is_year = bool(_YEAR_RE.match(digits)) and not magnitude and unit is None
        tail = text[m.end() : m.end() + 2].lower()
        is_ordinal = tail in {"st", "nd", "rd", "th"} and not magnitude and unit is None
        out.append(
            NumericMention(
                raw=raw + (tail if is_ordinal else ""),
                value=value,
                unit=unit,
                start=offset + m.start(),
                end=offset + m.end() + (2 if is_ordinal else 0),
                is_year=is_year,
                is_ordinal=is_ordinal,
            )
        )

    spans = [(n.start - offset, n.end - offset) for n in out]
    for m in _WORD_NUMBER_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        value = _WORD_NUMBERS[m.group("word").lower()]
        magnitude = (m.group("magnitude") or "").lower()
        if magnitude:
            value *= _MAGNITUDES.get(magnitude, 1.0)
        out.append(
            NumericMention(
                raw=m.group(0),
                value=value,
                unit=None,
                start=offset + m.start(),
                end=offset + m.end(),
            )
        )

    out.sort(key=lambda n: n.start)
    return _anchor(out, text, offset)


def _anchor(mentions: list[NumericMention], text: str, offset: int) -> list[NumericMention]:
    if not mentions:
        return mentions
    words = [
        t
        for t in tokenize(text, offset=offset)
        if not t.is_numeric and not t.is_stopword and len(t.text) > 1
    ]
    anchored: list[NumericMention] = []
    for mention in mentions:
        before = next((w.norm for w in reversed(words) if w.end <= mention.start), None)
        after = next((w.norm for w in words if w.start >= mention.end), None)
        anchored.append(
            NumericMention(
                raw=mention.raw,
                value=mention.value,
                unit=mention.unit,
                start=mention.start,
                end=mention.end,
                is_year=mention.is_year,
                is_ordinal=mention.is_ordinal,
                before=before,
                after=after,
            )
        )
    return anchored


def numbers_agree(
    claim_numbers: list[NumericMention],
    span_numbers: list[NumericMention],
) -> tuple[list[NumericMention], list[NumericMention]]:
    quantities = [n for n in claim_numbers if n.is_quantity]
    if not quantities:
        return [], []
    available = [s for s in span_numbers if s.is_quantity]

    disagreeing: list[NumericMention] = []
    unstated: list[NumericMention] = []
    for number in quantities:
        if any(number.close_to(s) for s in available):
            continue
        comparable = any(number.measures_same_thing(s) for s in available)
        (disagreeing if comparable else unstated).append(number)
    return disagreeing, unstated
