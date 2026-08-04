"""The deterministic judge.

This is the default path and the baseline every other result is measured against.
It needs no API key, no model download and no network, and it returns the same
answer every time it is run.

The scoring has three parts.

*Coverage* is the IDF-weighted fraction of a claim's content terms that appear in a
candidate passage. Rare terms count for more than common ones, so a passage earns
its score by containing the specific words that make the claim what it is.

*Alignment* narrows the cited passage to the tightest character range that still
accounts for those terms, which is what turns "somewhere on page 4" into an exact
citation the report can highlight.

*Conflict detection* is what separates this from a similarity score. A passage that
covers a claim's wording but disagrees on a figure or flips its polarity is not weak
evidence of support, it is evidence of contradiction, and the two are reported
differently because they need different fixes.
"""

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

#: Penalty applied when the passage gives a comparable figure that disagrees.
NUMERIC_PENALTY = 0.45
#: Penalty applied when the claim asserts a figure the passage never mentions.
#: Milder than a disagreement, because the source is silent rather than opposed.
UNSTATED_FIGURE_PENALTY = 0.72
#: Penalty applied when the claim and the passage disagree in polarity.
POLARITY_PENALTY = 0.50
#: Coverage a conflicting passage needs before it counts as a contradiction rather
#: than as an unrelated passage that happens to share a word.
CONTRADICTION_FLOOR = 0.55
#: Coverage below which a passage is not worth citing at all.
EVIDENCE_FLOOR = 0.15


@dataclass(frozen=True, slots=True)
class _Assessment:
    """One candidate, fully scored."""

    candidate: Candidate
    span: SourceSpan
    coverage: float
    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    conflict: str | None


class LexicalJudge:
    """Deterministic claim verification. No credentials, no network."""

    name = "lexical"

    def __init__(self, *, supported: float = 0.75, partial: float = 0.45) -> None:
        if not 0.0 <= partial <= supported <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= partial <= supported <= 1")
        self.supported = supported
        self.partial = partial
        self._idf: Callable[[str], float] | None = None

    # -- public API ---------------------------------------------------------

    def bind(self, index: CorpusIndex) -> None:
        """Adopt the corpus term weighting.

        Coverage is only meaningful when rare terms outweigh common ones, and that
        weighting is a property of the corpus rather than of the judge. Binding
        keeps the judge usable standalone (uniform weights) while giving it the real
        distribution whenever a run has built an index.
        """
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
        # Figures are deliberately kept out of the coverage weighting. They are
        # already checked exactly, by value, against the passage; letting them also
        # count as unmatched vocabulary would penalise a claim twice for the same
        # discrepancy and push a plain numeric contradiction down into the
        # unsupported band, where it would be reported as "not mentioned" rather
        # than as "the source says otherwise".
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
        # Clean passages compete on the penalised score, so that a claim asserting a
        # figure the source never gives cannot reach "supported" on wording alone.
        # Conflicting passages compete on raw coverage, because the question there is
        # how squarely the passage is about the claim, not how well it supports it.
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

    # -- scoring ------------------------------------------------------------

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
        # Numbers are checked against the whole candidate window, not the narrowed
        # span: a figure stated one clause away still supports the claim.
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
            # Silence is not disagreement. The claim is weakened for asserting a
            # figure the source never gives, but it is not called a contradiction.
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
        """Narrow the candidate to the tightest range accounting for the match.

        Considers every contiguous run of matching tokens and keeps the one with
        the highest matched weight, breaking ties towards the shortest range. The
        number of matching tokens in a three-sentence window is small, so the
        quadratic scan is cheaper than the machinery needed to avoid it.
        """
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

    # -- verdict ------------------------------------------------------------

    def _decide(
        self,
        best_clean: _Assessment | None,
        best_conflict: _Assessment | None,
    ) -> tuple[_Assessment, Verdict, str]:
        """Choose the evidence to cite and the verdict it implies.

        Order matters. A passage that supports the claim outright wins even when a
        conflicting passage exists elsewhere, because a corpus is allowed to contain
        an outdated figure alongside a current one. Only when nothing supports the
        claim does a conflicting passage become the story.
        """
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
        """The cited passage first, then the next best distinct alternatives."""
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
