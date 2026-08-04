"""The orchestrator: text plus sources in, verdicts out.

The pipeline is deliberately short and each step is separately testable.

    segment  ->  retrieve  ->  judge  ->  aggregate

Everything above this module is an interface (a CLI, an MCP tool, a report) and
everything below is a component (a loader, an index, a judge). This is the only
place they meet, which is what keeps the public API small enough to be worth
depending on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from footnote.corpus import load_corpus, load_text
from footnote.judge import Judge, resolve_judge
from footnote.models import CheckResult, Document
from footnote.retrieve import CorpusIndex
from footnote.segment import segment

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
    """Verify ``answer`` against already-loaded documents."""
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
    """Verify ``answer`` against files and directories on disk.

    This is the main entry point:

    >>> from footnote import check
    >>> result = check("Revenue grew 34%.", ["./sources"])
    >>> result.faithfulness  # doctest: +SKIP
    0.91
    """
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
    **kwargs: object,
) -> CheckResult:
    """Verify the contents of a file. Convenience wrapper over :func:`check`."""
    text = load_text(Path(answer_path).read_text(encoding="utf-8", errors="replace")).text
    return check(text, sources, **kwargs)  # type: ignore[arg-type]
