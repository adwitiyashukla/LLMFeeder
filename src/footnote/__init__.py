"""Footnote: claim-level source verification for generated text.

Give it a piece of writing and the documents it was supposedly derived from, and it
returns a verdict for every claim, each one pointing at an exact character range in
an exact file.

    >>> from footnote import check
    >>> result = check("Revenue grew 34% to $2.1B.", ["./sources"])   # doctest: +SKIP
    >>> result.faithfulness                                           # doctest: +SKIP
    0.91

The default judge is deterministic and runs entirely offline. The LLM judge is
opt-in and activates only when a credential is present on the local machine.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from footnote.corpus import load_corpus, load_document, load_text
from footnote.models import (
    CheckResult,
    Claim,
    ClaimResult,
    Document,
    Evidence,
    SourceSpan,
    Verdict,
)
from footnote.report import render_report, write_report
from footnote.retrieve import Candidate, CorpusIndex
from footnote.segment import segment
from footnote.verify import check, check_documents, check_files

try:
    __version__ = version("footnote-verify")
except PackageNotFoundError:  # pragma: no cover - source checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "Candidate",
    "CheckResult",
    "Claim",
    "ClaimResult",
    "CorpusIndex",
    "Document",
    "Evidence",
    "SourceSpan",
    "Verdict",
    "__version__",
    "check",
    "check_documents",
    "check_files",
    "load_corpus",
    "load_document",
    "load_text",
    "render_report",
    "segment",
    "write_report",
]
