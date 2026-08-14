from __future__ import annotations

import asyncio

import pytest

from llmfeeder.corpus import load_text
from llmfeeder.mcp_server import _summarise
from llmfeeder.verify import check_documents

SOURCE = "Free cash flow was 410 million dollars for the quarter."


@pytest.fixture
def payload() -> dict[str, object]:
    result = check_documents(
        "Free cash flow was 510 million dollars. The board declared a dividend.",
        [load_text(SOURCE, doc_id="report.txt")],
        judge="lexical",
    )
    return _summarise(result)


class TestSummary:
    def test_carries_the_full_result(self, payload: dict[str, object]) -> None:
        assert payload["claims_total"] == 2
        assert "faithfulness" in payload

    def test_needs_attention_lists_the_problem_claims(self, payload: dict[str, object]) -> None:
        attention = payload["needs_attention"]
        assert isinstance(attention, list)
        assert attention
        assert all(entry["verdict"] != "supported" for entry in attention)

    def test_needs_attention_explains_and_locates_each_problem(
        self, payload: dict[str, object]
    ) -> None:
        for entry in payload["needs_attention"]:
            assert entry["claim"]
            assert entry["why"]

    def test_is_json_serialisable(self, payload: dict[str, object]) -> None:
        import json

        assert json.loads(json.dumps(payload))["claims_total"] == 2


class TestServer:
    def test_server_builds_and_exposes_both_tools(self) -> None:
        pytest.importorskip("mcp")
        from llmfeeder.mcp_server import build_server

        tools = asyncio.run(build_server(".").list_tools())
        assert {t.name for t in tools} == {"verify_against_sources", "verify_against_text"}

    def test_every_tool_carries_a_description_and_a_schema(self) -> None:
        pytest.importorskip("mcp")
        from llmfeeder.mcp_server import build_server

        tools = asyncio.run(build_server(".").list_tools())
        for tool in tools:
            assert tool.description
            assert "text" in tool.inputSchema["properties"]
