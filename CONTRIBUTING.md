# Contributing

Thanks for taking a look. Bug reports, failing examples and pull requests are all welcome.

## Setup

```bash
git clone https://github.com/adwitiyashukla/footnote.git
cd footnote
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pip install pytest pytest-cov ruff mypy
```

## Before opening a pull request

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest
footnote eval
```

CI runs all five on Python 3.11 and 3.12. A pull request that leaves any of them red will not be merged.

## The rule that matters most

**A change to the judging logic must be justified by the evaluation harness, not by intuition.**

Run `footnote eval` before and after your change and put both sets of numbers in the pull request description. Verification quality is the entire product, so a refactor that quietly costs three points of recall is a regression even if every unit test still passes. The test suite asserts floors on accuracy, detection precision and detection recall precisely so that this cannot slip through.

If your change improves quality, raise the floors in `tests/test_report_eval.py::TestQualityFloor` in the same pull request.

## Adding evaluation cases

The dataset lives in `src/footnote/data/eval.json`. Each case pairs a short corpus with claims written about it.

```jsonc
{
  "case": "short-slug",
  "documents": { "source.txt": ["paragraph one", "paragraph two"] },
  "claims": [
    { "id": "slug-1", "text": "A claim about the source.", "label": "supported" }
  ]
}
```

Labels are `supported`, `partial`, `unsupported` and `contradicted`.

Two things to keep in mind. Write the source first and the claims second, so the claims are not unconsciously shaped to be easy. And do not relabel an example because the tool disagrees with it: if the label is right, the disagreement is a real finding and belongs in the failure analysis in the README.

Cases that Footnote currently gets **wrong** are especially valuable. A reproducible failure is more useful than another example it already handles.

## Scope

Footnote does one thing: decide whether a claim is supported by a corpus, and point at where. Things that fit well are better judges, better retrieval, more source formats and better reporting. Things that do not fit are generation, retrieval-augmented answering, and anything that turns this into a general document pipeline.

## Style

Line length 100, ruff for linting and formatting, `mypy --strict` with no new `Any` in public signatures. Comments should explain why a decision was made rather than restate what the code does.
