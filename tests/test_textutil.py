from __future__ import annotations

import pytest

from footnote.textutil import (
    content_tokens,
    extract_numbers,
    has_negation,
    numbers_agree,
    stem,
    tokenize,
)


class TestStem:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("resigned", "resign"),
            ("resignation", "resign"),
            ("resigns", "resign"),
            ("growing", "grow"),
            ("stopped", "stop"),
            ("cats", "cat"),
            ("data", "data"),
            ("2021", "2021"),
        ],
    )
    def test_variants_collide(self, word: str, expected: str) -> None:
        assert stem(word) == expected

    def test_short_words_untouched(self) -> None:
        assert stem("was") == "was"
        assert stem("is") == "is"


class TestTokenize:
    def test_offsets_are_exact(self) -> None:
        text = "Revenue grew 34% sharply."
        for token in tokenize(text):
            assert text[token.start : token.end] == token.text

    def test_offset_shift_is_applied(self) -> None:
        tokens = tokenize("alpha beta", offset=100)
        assert tokens[0].start == 100
        assert tokens[1].start == 106

    def test_content_tokens_drop_stopwords_but_keep_negations(self) -> None:
        norms = {t.text.lower() for t in content_tokens("the drug was not effective")}
        assert "the" not in norms
        assert "not" in norms
        assert "drug" in norms


class TestNumbers:
    def test_plain_integer(self) -> None:
        (mention,) = extract_numbers("about 47 points")
        assert mention.value == 47
        assert mention.is_quantity

    def test_magnitudes_and_currency_normalise(self) -> None:
        a = extract_numbers("$2.1B")[0]
        b = extract_numbers("2.1 billion dollars")[0]
        assert a.value == b.value == pytest.approx(2.1e9)

    def test_thousands_separator(self) -> None:
        assert extract_numbers("11,400 staff")[0].value == 11400

    def test_percentages_carry_a_unit(self) -> None:
        assert extract_numbers("grew 34%")[0].unit == "%"
        assert extract_numbers("grew 34 percent")[0].unit == "%"

    def test_years_are_not_quantities(self) -> None:
        assert extract_numbers("in 2021")[0].is_quantity is False

    def test_ordinals_are_not_quantities(self) -> None:
        (mention,) = extract_numbers("in the 12th minute")
        assert mention.is_ordinal
        assert mention.is_quantity is False

    def test_spelled_out_numbers(self) -> None:
        assert extract_numbers("three participants")[0].value == 3
        assert extract_numbers("two hundred sites")[0].value == 200

    def test_one_is_not_read_as_a_count(self) -> None:
        assert extract_numbers("one of the sites") == []

    def test_anchors_point_at_the_nearest_content_word(self) -> None:
        (mention,) = extract_numbers("adds 14 kilometres")
        assert mention.before == "add"
        assert mention.after == "kilometr"


class TestNumbersAgree:
    def test_matching_figures_agree(self) -> None:
        disagreeing, unstated = numbers_agree(
            extract_numbers("revenue of $2.1B"), extract_numbers("revenue was 2.1 billion dollars")
        )
        assert not disagreeing and not unstated

    def test_a_different_figure_for_the_same_thing_disagrees(self) -> None:
        disagreeing, unstated = numbers_agree(
            extract_numbers("free cash flow was 510 million dollars"),
            extract_numbers("free cash flow was 410 million dollars"),
        )
        assert [m.value for m in disagreeing] == [510e6]
        assert not unstated

    def test_a_figure_the_source_never_gives_is_unstated_not_contradicted(self) -> None:
        disagreeing, unstated = numbers_agree(
            extract_numbers("14 kilometres of bus lanes and six new bus stations"),
            extract_numbers("adds 14 kilometres of dedicated bus lanes"),
        )
        assert not disagreeing
        assert [m.value for m in unstated] == [6]

    def test_figures_about_different_things_never_compare(self) -> None:
        disagreeing, unstated = numbers_agree(
            extract_numbers("margin of 18%"), extract_numbers("headcount of 950 people")
        )
        assert not disagreeing
        assert len(unstated) == 1

    def test_a_bare_figure_can_still_match_a_united_one(self) -> None:
        disagreeing, _ = numbers_agree(
            extract_numbers("margin of 18%"), extract_numbers("margin of 18")
        )
        assert not disagreeing

    def test_years_are_ignored(self) -> None:
        assert numbers_agree(extract_numbers("in 2021"), extract_numbers("in 2024")) == ([], [])


class TestNegation:
    @pytest.mark.parametrize(
        "text",
        ["there were no deaths", "the plan does not include a tram", "without written consent"],
    )
    def test_detects_negation(self, text: str) -> None:
        assert has_negation(text)

    def test_plain_assertion_is_not_negated(self) -> None:
        assert not has_negation("the plan includes a tram extension")
