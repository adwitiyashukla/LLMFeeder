from __future__ import annotations

import pytest

from footnote import Verdict, check_documents
from footnote.corpus import load_text
from footnote.judge.lexical import LexicalJudge
from footnote.models import CheckResult, Document
from footnote.retrieve import CorpusIndex
from footnote.segment import segment

SOURCE = (
    "Acme reported revenue of 2.1 billion dollars for the third quarter, "
    "an increase of 34 percent year over year.\n\n"
    "Free cash flow was 410 million dollars. The plan does not include a tram "
    "extension.\n\n"
    "The company opened two new data centres, in Dublin and in Singapore."
)


@pytest.fixture
def corpus() -> list[Document]:
    return [load_text(SOURCE, doc_id="report.txt")]


def verdict_for(text: str, corpus: list[Document]) -> CheckResult:
    return check_documents(text, corpus, judge="lexical", min_words=1)


class TestRetrieval:
    def test_index_reports_unit_count(self, corpus: list[Document]) -> None:
        assert len(CorpusIndex(corpus)) > 0

    def test_search_returns_a_window_inside_the_document(self, corpus: list[Document]) -> None:
        index = CorpusIndex(corpus)
        (best, *_) = index.search("revenue of 2.1 billion dollars")
        assert best.doc_id == "report.txt"
        assert corpus[0].text[best.start : best.end] == best.text

    def test_windows_never_straddle_documents(self) -> None:
        documents = [
            load_text("Alpha claim body.", "a.txt"),
            load_text("Beta claim body.", "b.txt"),
        ]
        index = CorpusIndex(documents)
        for candidate in index.search("claim body"):
            assert candidate.text in next(d.text for d in documents if d.doc_id == candidate.doc_id)

    def test_empty_corpus_returns_nothing(self) -> None:
        assert CorpusIndex([]).search("anything at all") == []

    def test_unseen_terms_score_as_highly_discriminating(self, corpus: list[Document]) -> None:
        index = CorpusIndex(corpus)
        assert index.idf("zzzunseenzzz") > index.idf("revenue")


class TestVerdicts:
    def test_supported(self, corpus: list[Document]) -> None:
        result = verdict_for("Free cash flow was 410 million dollars.", corpus)
        assert result.claims[0].verdict is Verdict.SUPPORTED

    def test_contradicted_by_a_different_figure(self, corpus: list[Document]) -> None:
        result = verdict_for("Free cash flow was 510 million dollars.", corpus)
        entry = result.claims[0]
        assert entry.verdict is Verdict.CONTRADICTED
        assert entry.best is not None
        assert "510" in (entry.best.conflict or "")

    def test_contradicted_by_flipped_polarity(self, corpus: list[Document]) -> None:
        result = verdict_for("The plan includes a tram extension.", corpus)
        assert result.claims[0].verdict is Verdict.CONTRADICTED

    def test_unsupported_when_the_corpus_is_silent(self, corpus: list[Document]) -> None:
        result = verdict_for("The board declared a dividend of twelve pence.", corpus)
        assert result.claims[0].verdict is Verdict.UNSUPPORTED

    def test_a_figure_the_source_omits_is_not_a_contradiction(self, corpus: list[Document]) -> None:
        result = verdict_for(
            "The company opened two new data centres and eleven regional offices.", corpus
        )
        assert result.claims[0].verdict is not Verdict.CONTRADICTED


class TestCitations:
    def test_span_offsets_resolve_in_the_source(self, corpus: list[Document]) -> None:
        result = verdict_for("Free cash flow was 410 million dollars.", corpus)
        span = result.claims[0].best.span
        assert corpus[0].text[span.start : span.end] == span.text
        assert span.text.strip()

    def test_locator_is_human_readable(self, corpus: list[Document]) -> None:
        result = verdict_for("Free cash flow was 410 million dollars.", corpus)
        assert "report.txt" in result.claims[0].best.span.locator()

    def test_evidence_is_ordered_with_the_citation_first(self, corpus: list[Document]) -> None:
        result = verdict_for("Acme reported revenue of 2.1 billion dollars.", corpus)
        evidence = result.claims[0].evidence
        assert evidence
        assert evidence[0].span == result.claims[0].best.span


class TestAggregate:
    def test_faithfulness_is_bounded(self, corpus: list[Document]) -> None:
        result = verdict_for(
            "Free cash flow was 410 million dollars. The board declared a dividend.", corpus
        )
        assert 0.0 <= result.faithfulness <= 1.0

    def test_counts_include_every_verdict(self, corpus: list[Document]) -> None:
        assert set(verdict_for("Revenue grew.", corpus).counts()) == {v.value for v in Verdict}

    def test_problems_are_worst_first(self, corpus: list[Document]) -> None:
        result = verdict_for(
            "Free cash flow was 510 million dollars. The board declared a special dividend.",
            corpus,
        )
        ranks = [c.verdict.rank for c in result.problems()]
        assert ranks == sorted(ranks)

    def test_no_claims_warns_rather_than_failing(self, corpus: list[Document]) -> None:
        result = check_documents("Yes.", corpus, judge="lexical")
        assert result.claims == []
        assert any("no checkable claims" in w for w in result.warnings)

    def test_no_sources_warns(self) -> None:
        result = check_documents("Revenue grew by a third this year.", [], judge="lexical")
        assert any("no source documents" in w for w in result.warnings)

    def test_serialises_to_json_safe_types(self, corpus: list[Document]) -> None:
        import json

        payload = verdict_for("Free cash flow was 410 million dollars.", corpus).to_dict()
        assert json.loads(json.dumps(payload))["claims_total"] == 1


class TestJudgeConfiguration:
    def test_thresholds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            LexicalJudge(supported=0.4, partial=0.9)

    def test_unknown_judge_name_is_rejected(self, corpus: list[Document]) -> None:
        from footnote.judge import resolve_judge

        with pytest.raises(ValueError, match="unknown judge"):
            resolve_judge("telepathy")

    def test_auto_falls_back_to_lexical_without_a_credential(
        self, corpus: list[Document], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.chdir("/")
        result = check_documents("Revenue grew by a third.", corpus, judge="auto")
        assert result.judge == "lexical"

    def test_the_judge_works_both_bound_and_unbound(self, corpus: list[Document]) -> None:
        judge = LexicalJudge()
        (claim,) = segment("Free cash flow was 410 million dollars.", min_words=1)
        index = CorpusIndex(corpus)
        candidates = index.search(claim.text)

        assert judge.judge(claim, candidates).verdict is Verdict.SUPPORTED
        judge.bind(index)
        assert judge.judge(claim, candidates).verdict is Verdict.SUPPORTED
