from __future__ import annotations

from pathlib import Path

import pytest

from llmfeeder.corpus import LoaderError, load_corpus, load_document, load_text
from llmfeeder.segment import segment, split_sentences


class TestSegment:
    def test_offsets_round_trip(self) -> None:
        text = "Revenue grew sharply. The margin fell slightly."
        for claim in segment(text):
            assert text[claim.start : claim.end] == claim.text

    def test_abbreviations_do_not_split(self) -> None:
        claims = segment("The CFO resigned in Oct. 2025 after a review.")
        assert len(claims) == 1

    def test_decimals_do_not_split(self) -> None:
        claims = segment("Revenue rose 3.5 percent in the quarter.")
        assert len(claims) == 1

    def test_initials_do_not_split(self) -> None:
        assert len(segment("The report was written by J. Smith last year.")) == 1

    def test_headings_are_skipped(self) -> None:
        claims = segment("# Quarterly Results\n\nRevenue grew by a third.")
        assert len(claims) == 1
        assert claims[0].text.startswith("Revenue")

    def test_questions_are_skipped(self) -> None:
        claims = segment("Revenue grew by a third. What happened to margin?")
        assert [c.text for c in claims] == ["Revenue grew by a third."]

    def test_bullets_become_separate_claims(self) -> None:
        claims = segment("- Revenue grew by a third.\n- Margin fell by two points.")
        assert len(claims) == 2
        assert not claims[0].text.startswith("-")

    def test_code_fences_are_skipped(self) -> None:
        claims = segment("Revenue grew.\n\n```\nprint('this is not a claim at all')\n```\n")
        assert all("print" not in c.text for c in claims)

    def test_short_fragments_are_skipped(self) -> None:
        assert segment("Yes. No. Maybe.") == []

    def test_table_rows_are_skipped(self) -> None:
        claims = segment("| name | value |\n| --- | --- |\n\nRevenue grew by a third.")
        assert len(claims) == 1

    def test_claim_ids_are_unique_and_ordered(self) -> None:
        claims = segment(
            "Alpha revenue grew sharply. Beta margin fell sharply. Gamma headcount held steady."
        )
        assert [c.id for c in claims] == ["c1", "c2", "c3"]


class TestSplitSentences:
    def test_spans_cover_the_text(self) -> None:
        text = "First sentence here. Second sentence here."
        spans = split_sentences(text)
        assert [text[s:e] for s, e in spans] == ["First sentence here.", "Second sentence here."]

    def test_trailing_fragment_is_kept(self) -> None:
        spans = split_sentences("Complete one. Trailing fragment")
        assert len(spans) == 2


class TestLoaders:
    def test_load_text_normalises_whitespace(self) -> None:
        document = load_text("alpha   beta\r\n\r\n\r\n\r\ngamma")
        assert document.text == "alpha beta\n\ngamma"

    def test_plain_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "note.txt"
        path.write_text("Revenue grew by a third.", encoding="utf-8")
        assert load_document(path).text == "Revenue grew by a third."

    def test_html_markup_is_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "page.html"
        path.write_text(
            "<html><head><style>p{color:red}</style></head>"
            "<body><p>Revenue grew.</p><script>alert(1)</script></body></html>",
            encoding="utf-8",
        )
        text = load_document(path).text
        assert "Revenue grew." in text
        assert "alert" not in text
        assert "color:red" not in text

    def test_json_is_flattened_to_readable_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('{"company": {"revenue": "2.1B"}}', encoding="utf-8")
        assert "company.revenue: 2.1B" in load_document(path).text

    def test_unsupported_type_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "image.bmp"
        path.write_bytes(b"\x00\x01")
        with pytest.raises(LoaderError):
            load_document(path)

    def test_directories_are_walked(self, tmp_path: Path) -> None:
        (tmp_path / "nested").mkdir()
        (tmp_path / "a.txt").write_text("Alpha document body.", encoding="utf-8")
        (tmp_path / "nested" / "b.md").write_text("Beta document body.", encoding="utf-8")
        documents, warnings = load_corpus([tmp_path])
        assert {d.doc_id for d in documents} == {"a.txt", "b.md"}
        assert warnings == []

    def test_one_bad_file_does_not_stop_the_run(self, tmp_path: Path) -> None:
        (tmp_path / "good.txt").write_text("A perfectly readable document.", encoding="utf-8")
        (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
        documents, warnings = load_corpus([tmp_path])
        assert [d.doc_id for d in documents] == ["good.txt"]
        assert len(warnings) == 1

    def test_missing_path_is_reported(self) -> None:
        documents, warnings = load_corpus(["./definitely-not-here"])
        assert documents == []
        assert "no such file" in warnings[0]

    def test_duplicate_basenames_get_distinct_ids(self, tmp_path: Path) -> None:
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        (tmp_path / "one" / "notes.txt").write_text("First body of text.", encoding="utf-8")
        (tmp_path / "two" / "notes.txt").write_text("Second body of text.", encoding="utf-8")
        documents, _ = load_corpus([tmp_path])
        assert len({d.doc_id for d in documents}) == 2

    def test_page_lookup_without_pages_returns_none(self) -> None:
        assert load_text("body").page_at(0) is None
