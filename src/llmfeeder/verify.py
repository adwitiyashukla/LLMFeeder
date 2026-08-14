from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from llmfeeder.corpus import load_corpus, load_text
from llmfeeder.judge import Judge, resolve_judge
from llmfeeder.models import CheckResult, Document
from llmfeeder.retrieve import CorpusIndex
from llmfeeder.segment import segment

__all__ = ["check", "check_documents", "check_files"]

DEFAULT_SUPPORTED = 0.75
DEFAULT_PARTIAL = 0.45
DEFAULT_TOP_K = 5


def check_documents(
    answer: str,
    documents: Sequence[Document],
    *,
    judge: str | Judge = "auto",
    supported: float = DEFAULT_SUPPORTED,
    partial: float = DEFAULT_PARTIAL,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
    min_words: int = 4,
    warnings: Sequence[str] = (),
) -> CheckResult:
    resolved: Judge = (
        resolve_judge(judge, supported=supported, partial=partial, model=model)
        if isinstance(judge, str)
        else judge
    )

    claims = segment(answer, min_words=min_words)
    index = CorpusIndex(list(documents))
    resolved.bind(index)

    result = CheckResult(
        answer=answer,
        documents=list(documents),
        judge=resolved.name,
        threshold=supported,
        warnings=list(warnings),
    )
    if not claims:
        result.warnings.append("no checkable claims were found in the input text")
        return result
    if not documents:
        result.warnings.append("no source documents were loaded, so nothing can be grounded")

    for claim in claims:
        candidates = index.search(claim.text, top_k=top_k)
        result.claims.append(resolved.judge(claim, candidates))
    return result


def check(
    answer: str,
    sources: Iterable[str | Path],
    *,
    judge: str | Judge = "auto",
    supported: float = DEFAULT_SUPPORTED,
    partial: float = DEFAULT_PARTIAL,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
    min_words: int = 4,
) -> CheckResult:
    documents, warnings = load_corpus(sources)
    return check_documents(
        answer,
        documents,
        judge=judge,
        supported=supported,
        partial=partial,
        top_k=top_k,
        model=model,
        min_words=min_words,
        warnings=warnings,
    )


def check_files(
    answer_path: str | Path,
    sources: Iterable[str | Path],
    **kwargs: Any,
) -> CheckResult:
    text = load_text(Path(answer_path).read_text(encoding="utf-8", errors="replace")).text
    return check(text, sources, **kwargs)
