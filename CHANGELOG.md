# Changelog

All notable changes to Footnote are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-02

First release. Claim-level source verification with exact citations.

### Added

- **Verification engine.** Text is segmented into claims, matched against an
  IDF-weighted sentence index, narrowed to the tightest supporting character range,
  and judged into one of four verdicts: supported, partial, unsupported, contradicted.
- **Numeric reconciliation.** Figures are parsed into comparable values rather than
  compared as strings, so `$2.1B`, `2.1 billion` and `2,100,000,000` are recognised as
  the same quantity, units are respected, and a claim stating `510 million` against a
  source stating `410 million` is reported as a contradiction rather than as a close
  match. Spelled-out quantities, magnitudes and currency symbols are handled. Years
  and ordinals are excluded, since they locate a statement rather than quantify it.
- **Silence is distinguished from disagreement.** A figure the source never mentions
  is reported as unstated, not contradicted. Two figures are only compared when they
  share an anchor word, which keeps "six bus stations" apart from "14 kilometres of
  bus lanes".
- **Polarity checking**, so a claim that flips a negation in the source is caught.
- **Self-contained HTML report.** One file, no server and no external assets.
  Selecting a claim scrolls to and highlights the exact characters its verdict rests
  on, inside the rendered source. Keyboard navigation with `j` and `k`.
- **Evaluation harness** with a hand-labelled dataset of 68 claims across 8 source
  corpora, reporting per-verdict precision, recall and F1 plus binary hallucination
  detection. Quality thresholds are asserted in the test suite, so a regression in
  verification quality fails CI.
- **Optional LLM judge**, off unless a credential is found locally. Bounded to
  adjudicating the retrieved passages, with a consent notice before any egress and
  verbatim re-anchoring of every quote, so an unlocatable citation is discarded rather
  than reported. No SDK dependency; the provider adapter is standard library only and
  supports OpenAI-compatible endpoints and Anthropic.
- **Model Context Protocol server**, exposing `verify_against_sources` and
  `verify_against_text` so an agent can check its own output mid-task.
- **Source loaders** for text, Markdown, HTML, JSON and JSONL with the standard
  library, and PDF with page-accurate citations behind the `pdf` extra.
- **CLI**: `check`, `eval`, `demo`, `mcp`, `version`. `--fail-under` turns the
  faithfulness score into an exit code for use in CI.
- **Python API** with a stable span invariant: every reported range satisfies
  `document.text[span.start:span.end] == span.text`.

### Known limitations

The deterministic judge is bag-of-words and therefore cannot detect entity
substitution, antonym flips, or relations that are implied but never stated. These
are enumerated with worked examples in the README, and are the cases the LLM judge
exists to cover.
