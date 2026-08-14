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
except PackageNotFoundError:
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
