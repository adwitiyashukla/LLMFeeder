from __future__ import annotations

from pathlib import Path

import pytest

from footnote import Verdict, check_documents, render_report, write_report
from footnote.corpus import load_text
from footnote.evaluation import Metrics, demo_case, evaluate, load_dataset
from footnote.models import CheckResult, SourceSpan

SOURCE = (
    "Acme reported revenue of 2.1 billion dollars for the third quarter.\n\n"
    "Free cash flow was 410 million dollars for the quarter."
)


@pytest.fixture
def result() -> CheckResult:
    return check_documents(
        "Free cash flow was 410 million dollars. The board declared a special dividend.",
        [load_text(SOURCE, doc_id="report.txt")],
        judge="lexical",
    )


class TestReport:
    def test_is_a_complete_standalone_document(self, result: CheckResult) -> None:
        html = render_report(result)
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")

    def test_carries_no_external_assets(self, result: CheckResult) -> None:
        html = render_report(result)
        assert "src=" not in html
        assert "cdn" not in html.lower()

    def test_highlights_the_cited_span(self, result: CheckResult) -> None:
        assert "<mark" in render_report(result)

    def test_claims_are_linked_to_marks(self, result: CheckResult) -> None:
        html = render_report(result)
        assert 'data-claim="c1"' in html
        assert 'data-claims="c1"' in html

    def test_escapes_markup_in_the_sources(self) -> None:
        hostile = check_documents(
            "The tag is dangerous and should be escaped.",
            [load_text("<script>alert('x')</script> The tag is dangerous.", "x.txt")],
            judge="lexical",
        )
        assert "<script>alert" not in render_report(hostile)

    def test_writes_to_disk(self, result: CheckResult, tmp_path: Path) -> None:
        written = write_report(result, tmp_path / "nested" / "out.html")
        assert written.is_file()
        assert written.read_text(encoding="utf-8").startswith("<!doctype html>")

    def test_handles_an_empty_result(self) -> None:
        html = render_report(CheckResult(answer=""))
        assert "No checkable claims" in html


class TestModels:
    def test_verdict_severity_ordering(self) -> None:
        order = sorted(Verdict, key=lambda v: v.rank)
        assert order[0] is Verdict.CONTRADICTED
        assert order[-1] is Verdict.SUPPORTED

    def test_only_supported_counts_as_grounded(self) -> None:
        assert Verdict.SUPPORTED.is_grounded
        assert not Verdict.PARTIAL.is_grounded

    def test_invalid_span_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid span"):
            SourceSpan(doc_id="a", start=10, end=2, text="")

    def test_locator_includes_the_page_when_known(self) -> None:
        span = SourceSpan(doc_id="a.pdf", start=0, end=4, text="body", page=4)
        assert "p.4" in span.locator()

    def test_faithfulness_of_an_empty_result_is_zero(self) -> None:
        assert CheckResult(answer="").faithfulness == 0.0


class TestDataset:
    def test_bundled_dataset_loads(self) -> None:
        examples = load_dataset()
        assert len(examples) >= 50
        assert all(e.documents for e in examples)

    def test_every_verdict_is_represented(self) -> None:
        labels = {e.label for e in load_dataset()}
        assert labels == set(Verdict)

    def test_claim_ids_are_unique(self) -> None:
        ids = [e.id for e in load_dataset()]
        assert len(ids) == len(set(ids))

    def test_missing_dataset_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset("./no-such-dataset.json")

    def test_demo_case_is_runnable(self) -> None:
        answer, documents = demo_case()
        assert answer and documents


class TestMetrics:
    def test_perfect_predictions_score_one(self) -> None:
        metrics = Metrics(judge="test")
        for verdict in Verdict:
            metrics.confusion[verdict.value][verdict.value] = 5
            metrics.total += 5
            metrics.correct += 5
        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.detection()["f1"] == 1.0

    def test_detection_treats_anything_unsupported_as_positive(self) -> None:
        metrics = Metrics(judge="test")
        metrics.confusion["contradicted"]["unsupported"] = 4
        metrics.total = 4
        assert metrics.detection()["recall"] == 1.0

    def test_missed_hallucination_lowers_recall(self) -> None:
        metrics = Metrics(judge="test")
        metrics.confusion["unsupported"]["supported"] = 2
        metrics.confusion["unsupported"]["unsupported"] = 2
        metrics.total = 4
        assert metrics.detection()["recall"] == pytest.approx(0.5)

    def test_harness_runs_end_to_end(self) -> None:
        metrics = evaluate(load_dataset()[:8], judge="lexical")
        assert metrics.total == 8
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.judge == "lexical"

    def test_metrics_serialise(self) -> None:
        import json

        payload = evaluate(load_dataset()[:4], judge="lexical").to_dict()
        assert json.loads(json.dumps(payload))["examples"] == 4


class TestQualityFloor:
    def test_detection_recall_holds(self) -> None:
        assert evaluate(load_dataset(), judge="lexical").detection()["recall"] >= 0.80

    def test_detection_precision_holds(self) -> None:
        assert evaluate(load_dataset(), judge="lexical").detection()["precision"] >= 0.90

    def test_accuracy_holds(self) -> None:
        assert evaluate(load_dataset(), judge="lexical").accuracy >= 0.80
