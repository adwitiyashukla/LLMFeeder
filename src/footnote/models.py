from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"

    @property
    def is_grounded(self) -> bool:
        return self is Verdict.SUPPORTED

    @property
    def rank(self) -> int:
        return {
            Verdict.CONTRADICTED: 0,
            Verdict.UNSUPPORTED: 1,
            Verdict.PARTIAL: 2,
            Verdict.SUPPORTED: 3,
        }[self]


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: str
    text: str
    page_breaks: tuple[tuple[int, int], ...] = ()

    def page_at(self, offset: int) -> int | None:
        if not self.page_breaks:
            return None
        page = self.page_breaks[0][1]
        for start, number in self.page_breaks:
            if start > offset:
                break
            page = number
        return page

    def snippet(self, start: int, end: int, context: int = 0) -> str:
        lo = max(0, start - context)
        hi = min(len(self.text), end + context)
        return self.text[lo:hi]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    doc_id: str
    start: int
    end: int
    text: str
    page: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def locator(self) -> str:
        where = f"p.{self.page} " if self.page is not None else ""
        return f"{self.doc_id} {where}chars {self.start}-{self.end}"


@dataclass(frozen=True, slots=True)
class Evidence:
    span: SourceSpan
    score: float
    method: str
    matched_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    conflict: str | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ClaimResult:
    claim: Claim
    verdict: Verdict
    score: float
    evidence: tuple[Evidence, ...] = ()
    rationale: str | None = None

    @property
    def best(self) -> Evidence | None:
        return self.evidence[0] if self.evidence else None


@dataclass
class CheckResult:
    answer: str
    claims: list[ClaimResult] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    judge: str = "lexical"
    threshold: float = 0.75
    warnings: list[str] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.score for c in self.claims) / len(self.claims)

    @property
    def grounded_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict.is_grounded)

    def counts(self) -> dict[str, int]:
        out = {v.value: 0 for v in Verdict}
        for c in self.claims:
            out[c.verdict.value] += 1
        return out

    def problems(self) -> list[ClaimResult]:
        return sorted(
            (c for c in self.claims if not c.verdict.is_grounded),
            key=lambda c: (c.verdict.rank, c.score),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "judge": self.judge,
            "threshold": self.threshold,
            "claims_total": len(self.claims),
            "claims_grounded": self.grounded_count,
            "counts": self.counts(),
            "documents": [
                {"doc_id": d.doc_id, "path": d.path, "chars": len(d.text)} for d in self.documents
            ],
            "warnings": list(self.warnings),
            "claims": [
                {
                    "id": c.claim.id,
                    "text": c.claim.text,
                    "verdict": c.verdict.value,
                    "score": round(c.score, 4),
                    "rationale": c.rationale,
                    "evidence": [
                        {
                            "doc_id": e.span.doc_id,
                            "start": e.span.start,
                            "end": e.span.end,
                            "page": e.span.page,
                            "text": e.span.text,
                            "score": round(e.score, 4),
                            "method": e.method,
                            "matched_terms": list(e.matched_terms),
                            "missing_terms": list(e.missing_terms),
                            "conflict": e.conflict,
                        }
                        for e in c.evidence
                    ],
                }
                for c in self.claims
            ],
        }
