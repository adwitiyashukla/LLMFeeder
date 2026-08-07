# Changelog

## 0.1.0

First working version.

### What it does

- Splits text into individual claims, then checks each one against a set of source documents and returns one of four verdicts: supported, partial, unsupported or contradicted.
- Every verdict points at an exact character range in an exact file, so the report can highlight the specific words the answer relies on.
- Parses numbers into real values instead of comparing them as text, so `$2.1B`, `2.1 billion` and `2,100,000,000` all count as the same figure, and a claim saying 510 million against a source saying 410 million gets caught. Handles units, currency symbols and spelled-out numbers. Ignores years and ordinals, since those locate a statement rather than measure something.
- Treats a number the source never mentions as unstated rather than contradicted. Two figures only get compared if they share a nearby word, which stops "six bus stations" being checked against "14 kilometres of bus lanes".
- Checks polarity, so a claim that flips a negation in the source gets flagged.
- Writes a single self-contained HTML report. No server and no external files, so it can be emailed or committed. Click a claim to jump to its source, or use `j` and `k`.
- Ships an evaluation harness with 68 hand-labelled claims across 8 source corpora, reporting precision, recall and F1 per verdict plus overall hallucination detection. The thresholds are asserted in the test suite so a drop in quality fails CI.
- Optional LLM judge, off unless it finds an API key locally. It only sees the passages that were already retrieved, and any quote it produces gets looked up in the source afterwards. If the quote isn't there, the citation is dropped. No SDK dependency, just a small adapter over `urllib` covering OpenAI-compatible endpoints and Anthropic.
- MCP server exposing `verify_against_sources` and `verify_against_text`, so an AI agent can check its own output before showing it to you.
- Reads txt, markdown, HTML, JSON and JSONL with no extra dependencies. PDF works with the `pdf` extra and keeps page numbers in the citations.
- Commands: `check`, `eval`, `demo`, `mcp`. `--fail-under` turns the score into an exit code so it can gate a CI build.

### Known limitations

The offline judge works on word overlap, so it can't spot a swapped name, an antonym, or a relationship that's implied but never actually written down. The README lists every case it currently gets wrong with examples. The LLM judge exists to cover those.
