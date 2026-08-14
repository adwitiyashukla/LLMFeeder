from __future__ import annotations

import math
from dataclasses import dataclass

from llmfeeder.models import Document
from llmfeeder.segment import split_sentences
from llmfeeder.textutil import Token, content_tokens

__all__ = ["Candidate", "CorpusIndex"]

MAX_WINDOW = 3


@dataclass(frozen=True, slots=True)
class _Unit:
    doc_index: int
    start: int
    end: int
    stems: frozenset[str]


@dataclass(frozen=True, slots=True)
class Candidate:
    document: Document
    start: int
    end: int
    text: str
    score: float

    @property
    def doc_id(self) -> str:
        return self.document.doc_id


class CorpusIndex:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._units: list[_Unit] = []
        self._postings: dict[str, set[int]] = {}
        self._idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        for doc_index, document in enumerate(self.documents):
            for start, end in split_sentences(document.text):
                stems = frozenset(t.norm for t in content_tokens(document.text[start:end]))
                if not stems:
                    continue
                unit_index = len(self._units)
                self._units.append(_Unit(doc_index, start, end, stems))
                for stem_value in stems:
                    self._postings.setdefault(stem_value, set()).add(unit_index)

        total = max(len(self._units), 1)
        for term, units in self._postings.items():
            self._idf[term] = max(math.log((total + 1) / (len(units) + 1)), 0.0) + 0.1

    def __len__(self) -> int:
        return len(self._units)

    def idf(self, term: str) -> float:
        if term in self._idf:
            return self._idf[term]
        return math.log(float(len(self._units) + 1)) + 0.1

    def weights(self, tokens: list[Token]) -> dict[str, float]:
        return {t.norm: self.idf(t.norm) for t in tokens}

    def search(self, claim_text: str, *, top_k: int = 5) -> list[Candidate]:
        tokens = content_tokens(claim_text)
        if not tokens or not self._units:
            return []
        weights = self.weights(tokens)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return []

        seeds: set[int] = set()
        for term in weights:
            seeds |= self._postings.get(term, set())
        if not seeds:
            return []

        scored: list[Candidate] = []
        for seed in sorted(seeds):
            unit = self._units[seed]
            covered: set[str] = set()
            for size in range(1, MAX_WINDOW + 1):
                last = seed + size - 1
                if last >= len(self._units):
                    break
                tail = self._units[last]
                if tail.doc_index != unit.doc_index:
                    break
                covered |= tail.stems & weights.keys()
                coverage = sum(weights[t] for t in covered) / total_weight
                score = coverage * (1.0 - 0.04 * (size - 1))
                scored.append(
                    Candidate(
                        document=self.documents[unit.doc_index],
                        start=unit.start,
                        end=tail.end,
                        text=self.documents[unit.doc_index].text[unit.start : tail.end],
                        score=score,
                    )
                )

        scored.sort(key=lambda c: (-c.score, c.end - c.start, c.doc_id, c.start))
        return _dedupe(scored, top_k)


def _dedupe(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in candidates:
        if any(
            k.doc_id == candidate.doc_id and candidate.start < k.end and k.start < candidate.end
            for k in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= top_k:
            break
    return kept
