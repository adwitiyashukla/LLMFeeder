"""Model Context Protocol server.

Exposes verification as a tool so an agent can check its own output before handing
it to a user. The interesting case is self-correction mid-task: a model writes a
summary, calls ``verify_against_sources``, sees that one sentence is unsupported,
and rewrites that sentence rather than shipping it.

Results are returned as compact JSON rather than prose, since the caller is a model
deciding what to do next, not a person reading a report.
"""

from __future__ import annotations

from typing import Any

from footnote.corpus import load_text
from footnote.models import CheckResult
from footnote.verify import check, check_documents

__all__ = ["build_server", "serve"]


def _summarise(result: CheckResult) -> dict[str, Any]:
    """Trim the full result to what an agent needs to act on."""
    payload = result.to_dict()
    payload["needs_attention"] = [
        {
            "claim": entry.claim.text,
            "verdict": entry.verdict.value,
            "score": round(entry.score, 3),
            "why": (entry.best.conflict if entry.best and entry.best.conflict else entry.rationale),
            "closest_source": entry.best.span.locator() if entry.best else None,
            "closest_text": entry.best.span.text if entry.best else None,
        }
        for entry in result.problems()
    ]
    return payload


def build_server(default_sources: str = ".") -> Any:
    """Construct the MCP server. Kept separate from :func:`serve` for testing."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("footnote")

    @server.tool()
    def verify_against_sources(
        text: str,
        sources_dir: str = default_sources,
        judge: str = "lexical",
    ) -> dict[str, Any]:
        """Check whether every claim in `text` is supported by the documents in a folder.

        Returns a faithfulness score in [0, 1], a verdict for each claim, and the exact
        source location backing each one. Use this before presenting generated content
        that is supposed to be derived from a document set.

        Args:
            text: The passage to verify.
            sources_dir: Folder of source documents to check against.
            judge: "lexical" for the offline deterministic judge, "llm" to adjudicate
                with a language model, or "auto" to use the LLM when a key is present.
        """
        return _summarise(check(text, [sources_dir], judge=judge))

    @server.tool()
    def verify_against_text(
        text: str,
        sources: list[str],
        judge: str = "lexical",
    ) -> dict[str, Any]:
        """Check `text` against source documents supplied inline as strings.

        Use this when the sources are already in context, for example passages just
        retrieved by a search tool, and there is nothing on disk to point at.

        Args:
            text: The passage to verify.
            sources: The source documents, each as a plain string.
            judge: "lexical", "llm", or "auto".
        """
        documents = [load_text(body, doc_id=f"source-{i + 1}") for i, body in enumerate(sources)]
        return _summarise(check_documents(text, documents, judge=judge))

    return server


def serve(default_sources: str = ".") -> None:  # pragma: no cover - a blocking loop
    """Run the server on stdio until the client disconnects."""
    build_server(default_sources).run()
