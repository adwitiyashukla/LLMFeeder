"""Judges: given a claim and its candidate passages, decide the verdict.

Two implementations share one interface. The lexical judge is deterministic, needs
no credentials and never sends data anywhere. The LLM judge is opt-in and activates
only when a key is present locally.

Retrieval is common to both. The LLM judge does not get to roam the corpus; it sees
the same candidate windows the lexical judge saw, which keeps token cost bounded and
keeps the two directly comparable in the evaluation harness.
"""

from __future__ import annotations

from typing import Protocol

from footnote.models import Claim, ClaimResult
from footnote.retrieve import Candidate, CorpusIndex

__all__ = ["Judge", "available_judges", "resolve_judge"]


class Judge(Protocol):
    """Scores one claim against the passages retrieved for it."""

    name: str

    def bind(self, index: CorpusIndex) -> None:
        """Adopt the corpus term weighting before judging begins."""
        ...

    def judge(self, claim: Claim, candidates: list[Candidate]) -> ClaimResult:
        """Return the verdict for ``claim`` given its candidate passages."""
        ...


def available_judges() -> tuple[str, ...]:
    return ("lexical", "llm", "auto")


def resolve_judge(
    name: str,
    *,
    supported: float = 0.75,
    partial: float = 0.45,
    model: str | None = None,
) -> Judge:
    """Build a judge by name.

    ``auto`` picks the LLM judge when a credential is available locally and falls
    back to the lexical judge otherwise, so the default path always works offline.
    """
    from footnote.judge.lexical import LexicalJudge

    lexical = LexicalJudge(supported=supported, partial=partial)

    if name == "lexical":
        return lexical
    if name in {"llm", "auto"}:
        from footnote.judge.llm import LLMJudge, credential_available

        if name == "auto" and not credential_available():
            return lexical
        return LLMJudge(fallback=lexical, model=model, supported=supported, partial=partial)
    raise ValueError(f"unknown judge '{name}'. Choose from: {', '.join(available_judges())}")
