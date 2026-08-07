from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from footnote.models import Claim, ClaimResult, Document, Evidence, SourceSpan, Verdict
from footnote.retrieve import Candidate, CorpusIndex
from footnote.textutil import (
    NumericMention,
    Token,
    content_tokens,
    extract_numbers,
    has_negation,
    numbers_agree,
    tokenize,
)

__all__ = ["LexicalJudge"]

NUMERIC_PENALTY = 0.45
UNSTATED_FIGURE_PENALTY = 0.72
POLARITY_PENALTY = 0.50
CONTRADICTION_FLOOR = 0.55
EVIDENCE_FLOOR = 0.15


@dataclass(frozen=True, slots=True)
class _Assessment:
    candidate: Candidate
    span: SourceSpan
    coverage: float
    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    conflict: str | None


class LexicalJudge:
    name = "lexical"

    def __init__(self, *, supported: float = 0.75, partial: float = 0.45) -> None:
        if not 0.0 <= partial <= supported <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= partial <= supported <= 1")
        self.supported = supported
        self.partial = partial
        self._idf: Callable[[str], float] | None = None

    def bind(self, index: CorpusIndex) -> None:
        self._idf = index.idf

    def judge(self, claim: Claim, candidates: list[Candidate]) -> ClaimResult:
        tokens = content_tokens(claim.text)
        if not tokens or not candidates:
            return ClaimResult(
                claim=claim,
                verdict=Verdict.UNSUPPORTED,
                score=0.0,
                rationale="no candidate passage in the corpus shares any content term",
            )

        idf = self._idf
        topical = [t for t in tokens if not t.is_numeric] or tokens
        weights = {t.norm: (idf(t.norm) if idf else 1.0) for t in topical}
        total = sum(weights.values())

        claim_numbers = [n for n in extract_numbers(claim.text) if not n.is_year]
        claim_negated = has_negation(claim.text)

        assessments = [
            self._assess(c, topical, weights, total, claim_numbers, claim_negated)
            for c in candidates
        ]
        assessments = [a for a in assessments if a.coverage >= EVIDENCE_FLOOR]
        if not assessments:
            return ClaimResult(
                claim=claim,
                verdict=Verdict.UNSUPPORTED,
                score=0.0,
                rationale="no passage covered enough of the claim to be worth citing",
            )

        clean = [a for a in assessments if a.conflict is None]
        conflicting = [a for a in assessments if a.conflict is not None]
        best_clean = max(clean, key=lambda a: a.score, default=None)
        best_conflict = max(conflicting, key=lambda a: a.coverage, default=None)

        chosen, verdict, rationale = self._decide(best_clean, best_conflict)
        evidence = self._evidence(chosen, assessments)
        return ClaimResult(
            claim=claim,
            verdict=verdict,
            score=round(chosen.score, 4),
            evidence=evidence,
            rationale=rationale,
        )

    def _assess(
        self,
        candidate: Candidate,
        tokens: list[Token],
        weights: dict[str, float],
        total: float,
        claim_numbers: list[NumericMention],
        claim_negated: bool,
    ) -> _Assessment:
        span_tokens = tokenize(candidate.text)
        present = {t.norm for t in span_tokens}
        wanted = {t.norm for t in tokens}

        matched = wanted & present
        missing = wanted - present
        coverage = sum(weights[t] for t in matched) / total if total else 0.0

        span = self._align(candidate, matched, weights, span_tokens)
        span_numbers = extract_numbers(candidate.text)
        disagreeing, unstated = numbers_agree(claim_numbers, span_numbers)
        polarity_ok = claim_negated == has_negation(candidate.text)

        penalty = 1.0
        conflict: str | None = None
        if disagreeing:
            penalty *= NUMERIC_PENALTY
            figures = ", ".join(n.raw for n in disagreeing)
            found = ", ".join(n.raw for n in span_numbers if n.is_quantity) or "no figure"
            conflict = f"claim states {figures}; the cited passage states {found}"
        if unstated:
            penalty *= UNSTATED_FIGURE_PENALTY
        if not polarity_ok:
            penalty *= POLARITY_PENALTY
            direction = "negates" if claim_negated else "asserts"
            other = "asserts" if claim_negated else "negates"
            detail = f"the claim {direction} what the cited passage {other}"
            conflict = f"{conflict}; {detail}" if conflict else detail

        return _Assessment(
            candidate=candidate,
            span=span,
            coverage=coverage,
            score=coverage * penalty,
            matched=tuple(sorted(matched)),
            missing=tuple(sorted(missing)),
            conflict=conflict,
        )

    def _align(
        self,
        candidate: Candidate,
        matched: set[str],
        weights: dict[str, float],
        span_tokens: list[Token],
    ) -> SourceSpan:
        document = candidate.document
        hits = [t for t in span_tokens if t.norm in matched]
        if not hits:
            return self._span(document, candidate.start, candidate.end)

        best_key: tuple[float, int, int] | None = None
        best_range: tuple[int, int] = (hits[0].start, hits[0].end)
        for i in range(len(hits)):
            covered: set[str] = set()
            for j in range(i, len(hits)):
                covered.add(hits[j].norm)
                weight = sum(weights[t] for t in covered)
                start, end = hits[i].start, hits[j].end
                key = (weight, -(end - start), -start)
                if best_key is None or key > best_key:
                    best_key = key
                    best_range = (start, end)
        start, end = best_range
        return self._span(document, candidate.start + start, candidate.start + end)

    @staticmethod
    def _span(document: Document, start: int, end: int) -> SourceSpan:
        return SourceSpan(
            doc_id=document.doc_id,
            start=start,
            end=end,
            text=document.text[start:end],
            page=document.page_at(start),
        )

    def _decide(
        self,
        best_clean: _Assessment | None,
        best_conflict: _Assessment | None,
    ) -> tuple[_Assessment, Verdict, str]:
        if best_clean is not None and best_clean.score >= self.supported:
            return (
                best_clean,
                Verdict.SUPPORTED,
                "every content term is present in the cited passage",
            )

        if (
            best_conflict is not None
            and best_conflict.coverage >= CONTRADICTION_FLOOR
            and (best_clean is None or best_conflict.coverage > best_clean.coverage)
        ):
            return (
                best_conflict,
                Verdict.CONTRADICTED,
                best_conflict.conflict or "the cited passage disagrees with the claim",
            )

        if best_clean is not None and best_clean.score >= self.partial:
            missing = ", ".join(best_clean.missing[:6]) or "none"
            return (
                best_clean,
                Verdict.PARTIAL,
                f"the passage covers most of the claim; unaccounted terms: {missing}",
            )

        fallback = best_clean or best_conflict
        assert fallback is not None
        missing = ", ".join(fallback.missing[:6]) or "none"
        return (
            fallback,
            Verdict.UNSUPPORTED,
            f"no passage accounts for the claim; unaccounted terms: {missing}",
        )

    @staticmethod
    def _evidence(chosen: _Assessment, assessments: list[_Assessment]) -> tuple[Evidence, ...]:
        ordered = [
            chosen,
            *sorted((a for a in assessments if a is not chosen), key=lambda a: -a.coverage),
        ]
        return tuple(
            Evidence(
                span=a.span,
                score=round(a.score, 4),
                method="lexical",
                matched_terms=a.matched,
                missing_terms=a.missing,
                conflict=a.conflict,
            )
            for a in ordered[:3]
        )
