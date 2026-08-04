"""Core data model.

Footnote's contract is that every verdict points at an exact place in an exact
source. That contract lives here: a :class:`SourceSpan` is a half-open character
range inside one loaded document, and nothing downstream is allowed to report
support without producing one.

The models are plain dataclasses on purpose. They serialise to JSON with
:func:`dataclasses.asdict`, they type-check under mypy strict, and they add no
runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """How well a claim is supported by the source corpus.

    The distinction between ``UNSUPPORTED`` and ``CONTRADICTED`` is deliberate and
    is the useful part. "The corpus does not mention this" and "the corpus says the
    opposite" are different failures with different fixes, and a similarity score
    alone cannot tell them apart.
    """

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"

    @property
    def is_grounded(self) -> bool:
        """True when the claim may be relied on without review."""
        return self is Verdict.SUPPORTED

    @property
    def rank(self) -> int:
        """Severity order, worst first. Used to sort a review queue."""
        return {
            Verdict.CONTRADICTED: 0,
            Verdict.UNSUPPORTED: 1,
            Verdict.PARTIAL: 2,
            Verdict.SUPPORTED: 3,
        }[self]


@dataclass(frozen=True, slots=True)
class Document:
    """One loaded source, normalised to text with stable character offsets.

    ``text`` is the single authority for offsets. Every :class:`SourceSpan` indexes
    into this string, so a loader must never hand back offsets computed against the
    raw bytes of a PDF or the raw markup of an HTML file.
    """

    doc_id: str
    path: str
    text: str
    #: Sorted ``(char_offset, page_number)`` pairs. Only PDFs populate this.
    page_breaks: tuple[tuple[int, int], ...] = ()

    def page_at(self, offset: int) -> int | None:
        """Return the 1-based page containing ``offset``, if the loader knew pages."""
        if not self.page_breaks:
            return None
        page = self.page_breaks[0][1]
        for start, number in self.page_breaks:
            if start > offset:
                break
            page = number
        return page

    def snippet(self, start: int, end: int, context: int = 0) -> str:
        """Return the document text in ``[start, end)``, optionally with context."""
        lo = max(0, start - context)
        hi = min(len(self.text), end + context)
        return self.text[lo:hi]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """An exact, half-open character range inside one document."""

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
        """A short human-readable citation, e.g. ``report.pdf p.4 chars 1180-1223``."""
        where = f"p.{self.page} " if self.page is not None else ""
        return f"{self.doc_id} {where}chars {self.start}-{self.end}"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of support (or contradiction) found for a claim."""

    span: SourceSpan
    #: Support strength in ``[0, 1]``. For a contradiction this stays high, because
    #: the span *is* about the claim; the verdict is what records the disagreement.
    score: float
    #: Which judge produced this, ``"lexical"`` or ``"llm"``.
    method: str
    #: Content terms from the claim that this span accounts for.
    matched_terms: tuple[str, ...] = ()
    #: Content terms from the claim that no span accounted for.
    missing_terms: tuple[str, ...] = ()
    #: Populated when a number or a negation in the claim conflicts with the span.
    conflict: str | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    """One checkable assertion carved out of the text under review."""

    id: str
    text: str
    #: Offsets into the original answer text, so the report can highlight in place.
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """The verdict for one claim, with the evidence that produced it."""

    claim: Claim
    verdict: Verdict
    score: float
    evidence: tuple[Evidence, ...] = ()
    rationale: str | None = None

    @property
    def best(self) -> Evidence | None:
        """The strongest piece of evidence, or None when nothing was found."""
        return self.evidence[0] if self.evidence else None


@dataclass
class CheckResult:
    """The outcome of verifying one piece of text against one corpus."""

    answer: str
    claims: list[ClaimResult] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    judge: str = "lexical"
    threshold: float = 0.75
    #: Non-fatal problems worth telling the user about, e.g. a skipped source.
    warnings: list[str] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        """Mean support score across claims, in ``[0, 1]``.

        This is the headline number. It is deliberately an average of per-claim
        scores rather than a pass rate, so that a barely-supported claim and a
        squarely-supported one do not count the same.
        """
        if not self.claims:
            return 0.0
        return sum(c.score for c in self.claims) / len(self.claims)

    @property
    def grounded_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict.is_grounded)

    def counts(self) -> dict[str, int]:
        """Claim count per verdict, always including every verdict as a key."""
        out = {v.value: 0 for v in Verdict}
        for c in self.claims:
            out[c.verdict.value] += 1
        return out

    def problems(self) -> list[ClaimResult]:
        """Claims needing attention, worst first."""
        return sorted(
            (c for c in self.claims if not c.verdict.is_grounded),
            key=lambda c: (c.verdict.rank, c.score),
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable summary. Document text is omitted by design."""
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
