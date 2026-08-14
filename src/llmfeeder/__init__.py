from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from llmfeeder.corpus import load_corpus, load_document, load_text
from llmfeeder.models import (
    CheckResult,
    Claim,
    ClaimResult,
    Document,
    Evidence,
    SourceSpan,
    Verdict,
)
from llmfeeder.report import render_report, write_report
from llmfeeder.retrieve import Candidate, CorpusIndex
from llmfeeder.segment import segment
from llmfeeder.verify import check, check_documents, check_files

try:
    __version__ = version("llmfeeder")
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
