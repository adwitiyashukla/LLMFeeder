from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path

from footnote.models import Document

__all__ = ["SUPPORTED_SUFFIXES", "LoaderError", "load_corpus", "load_document", "load_text"]


class LoaderError(RuntimeError):
    pass


_TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".text",
        ".log",
        ".csv",
        ".tsv",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
        ".sql",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".c",
        ".h",
        ".cpp",
        ".cs",
        ".sh",
        ".css",
        ".xml",
        ".svg",
    }
)
_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_PDF_SUFFIXES = frozenset({".pdf"})

SUPPORTED_SUFFIXES: frozenset[str] = (
    _TEXT_SUFFIXES | _HTML_SUFFIXES | _JSON_SUFFIXES | _PDF_SUFFIXES
)

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)

_WS_RUN = re.compile(r"[ \t\x0b\f\r]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _WS_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


class _TextExtractor(HTMLParser):
    _BLOCK = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section",
            "article",
            "header",
            "footer",
            "blockquote",
            "pre",
            "td",
            "th",
        }
    )
    _DROP = frozenset({"script", "style", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._suppress = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._DROP:
            self._suppress += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._DROP and self._suppress:
            self._suppress -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._suppress:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _flatten_json(value: object, prefix: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_json(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _flatten_json(item, f"{prefix}[{i}]")
    else:
        yield f"{prefix}: {value}" if prefix else str(value)


def _read_json(raw: str, suffix: str) -> str:
    if suffix in {".jsonl", ".ndjson"}:
        lines: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lines.extend(_flatten_json(json.loads(line)))
            except json.JSONDecodeError:
                lines.append(line)
        return "\n".join(lines)
    try:
        return "\n".join(_flatten_json(json.loads(raw)))
    except json.JSONDecodeError as exc:
        raise LoaderError(f"invalid JSON: {exc}") from exc


def _read_pdf(path: Path) -> tuple[str, tuple[tuple[int, int], ...]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise LoaderError(
            "PDF support needs an extra. From a checkout: pip install -e '.[pdf]'"
        ) from exc

    chunks: list[str] = []
    breaks: list[tuple[int, int]] = []
    cursor = 0
    with pdfplumber.open(str(path)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            body = _normalise(page.extract_text() or "")
            breaks.append((cursor, number))
            chunks.append(body)
            cursor += len(body) + 2
    return "\n\n".join(chunks), tuple(breaks)


def load_text(text: str, doc_id: str = "input", path: str = "<memory>") -> Document:
    return Document(doc_id=doc_id, path=path, text=_normalise(text))


def load_document(path: Path, doc_id: str | None = None) -> Document:
    suffix = path.suffix.lower()
    ident = doc_id or path.name

    if suffix in _PDF_SUFFIXES:
        body, breaks = _read_pdf(path)
        return Document(doc_id=ident, path=str(path), text=body, page_breaks=breaks)

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LoaderError(str(exc)) from exc

    if suffix in _HTML_SUFFIXES:
        parser = _TextExtractor()
        parser.feed(raw)
        parser.close()
        raw = parser.text()
    elif suffix in _JSON_SUFFIXES:
        raw = _read_json(raw, suffix)
    elif suffix not in _TEXT_SUFFIXES:
        raise LoaderError(
            f"unsupported file type '{suffix or path.name}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    body = _normalise(raw)
    if not body:
        raise LoaderError("file is empty after normalisation")
    return Document(doc_id=ident, path=str(path), text=body)


def _walk(root: Path) -> Iterator[Path]:
    for entry in sorted(root.rglob("*")):
        if entry.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in entry.parts):
            continue
        if entry.suffix.lower() in SUPPORTED_SUFFIXES:
            yield entry


def load_corpus(paths: Iterable[str | Path]) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}

    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(_walk(path))
        elif path.exists():
            files.append(path)
        else:
            warnings.append(f"{path}: no such file or directory")

    for file_path in files:
        base = file_path.name
        count = seen.get(base, 0)
        seen[base] = count + 1
        doc_id = base if count == 0 else f"{base}#{count + 1}"
        try:
            documents.append(load_document(file_path, doc_id=doc_id))
        except LoaderError as exc:
            warnings.append(f"{file_path}: {exc}")

    if not documents and not warnings:
        warnings.append("no readable sources were found")
    return documents, warnings
