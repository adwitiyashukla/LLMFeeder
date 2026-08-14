from __future__ import annotations

from typing import Protocol

from llmfeeder.models import Claim, ClaimResult
from llmfeeder.retrieve import Candidate, CorpusIndex

__all__ = ["Judge", "available_judges", "resolve_judge"]


class Judge(Protocol):
    name: str

    def bind(self, index: CorpusIndex) -> None: ...

    def judge(self, claim: Claim, candidates: list[Candidate]) -> ClaimResult: ...


def available_judges() -> tuple[str, ...]:
    return ("lexical", "llm", "auto")


def resolve_judge(
    name: str,
    *,
    supported: float = 0.75,
    partial: float = 0.45,
    model: str | None = None,
) -> Judge:
    from llmfeeder.judge.lexical import LexicalJudge

    lexical = LexicalJudge(supported=supported, partial=partial)

    if name == "lexical":
        return lexical
    if name in {"llm", "auto"}:
        from llmfeeder.judge.llm import LLMJudge, credential_available

        if name == "auto" and not credential_available():
            return lexical
        return LLMJudge(fallback=lexical, model=model, supported=supported, partial=partial)
    raise ValueError(f"unknown judge '{name}'. Choose from: {', '.join(available_judges())}")
