# Contributing

This is a personal project I built while learning, so I'm not expecting a queue of pull requests. But if you found it, tried it, and something broke or gave a weird answer, I'd genuinely like to know.

## Getting set up

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

CI runs all five on Python 3.11 and 3.12, so if any of them are red here they'll be red there too.

## If you change how the scoring works

Please run `footnote eval` before and after, and put both sets of numbers in the pull request. The whole point of the tool is the quality of its judgements, so a change that quietly loses a few points of recall is a step backwards even if every test still passes. That's also why there are threshold checks in `tests/test_report_eval.py`, so a regression fails the build instead of slipping through.

If your change makes it better, raise those thresholds in the same pull request.

## Adding to the evaluation set

The dataset is `src/footnote/data/eval.json`. Each entry is a short set of source documents plus some claims written about them.

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

Two things I learned the hard way while building this set. Write the source document first and the claims second, otherwise you unconsciously write claims that are easy to get right. And don't change a label just because the tool disagrees with it. If the label is correct, the disagreement is a real finding and belongs in the failure list in the README.

Cases it currently gets **wrong** are the most useful thing you could send me.

## Scope

Footnote does one thing: decide whether a claim is backed up by a set of documents, and point at where. Better judging, better retrieval, more file formats and better reports all fit. Generating text, doing retrieval-augmented question answering, or turning this into a general document pipeline don't.

## Style

Line length 100, ruff for linting and formatting, `mypy --strict`. The code deliberately has no comments or docstrings outside the CLI and MCP server, where they're used to generate help text.
