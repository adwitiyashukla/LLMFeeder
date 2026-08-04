# Footnote

**Catches AI hallucinations by checking every claim against your sources, with character-exact citations.**
**0.93 F1 detection. Zero API keys.**

[![CI](https://github.com/adwitiyashukla/footnote/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/footnote/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/footnote-verify.svg)](https://pypi.org/project/footnote-verify/)
[![Python](https://img.shields.io/pypi/pyversions/footnote-verify.svg)](https://pypi.org/project/footnote-verify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

You gave a model some documents and it gave you a paragraph back. Some of that paragraph is in the documents. Some of it is not. Footnote tells you which is which, and points at the exact characters.

```
$ footnote check answer.md --sources ./reports

claims needing attention
CONTRADICTED   0.45  Free cash flow was 510 million dollars for the quarter.
                     q3-earnings.txt chars 728-772
                     claim states 510 million dollars; the cited passage states 410 million dollars
CONTRADICTED   0.28  Headcount at the end of the quarter was 12,400 employees.
                     q3-earnings.txt chars 903-948
                     claim states 12,400; the cited passage states 11,400, 10,900
UNSUPPORTED    0.14  The company announced a quarterly dividend of 12 cents per share.
                     no supporting span found

9 claims  6 supported  0 partial  1 unsupported  2 contradicted
faithfulness 0.73  (lexical judge, 1 sources)
```

Add `--report out.html` and you get a page where clicking any claim highlights the exact sentence it rests on, inside the rendered source. [See a real one.](docs/example-report.html)

---

## Why this is not a similarity score

Nearly every attribution tool reduces a claim and a candidate passage to an embedding distance. That approach cannot see the difference between these two sentences:

> Free cash flow was **410** million dollars.
> Free cash flow was **510** million dollars.

They are 98% identical as strings and near-identical as embeddings, so a similarity threshold marks the second one as supported. It is the single most common way a generated document goes wrong, and it is exactly the case similarity is blind to.

Footnote parses figures into comparable values instead. `$2.1B`, `2.1 billion` and `2,100,000,000` resolve to the same number, `34%` and `34 percent` carry the same unit, and `three participants` is compared against `two participants`. It also separates two failures that similarity collapses into one:

| Verdict | Meaning | What you do about it |
|---|---|---|
| `SUPPORTED` | every part of the claim is in the source | nothing |
| `PARTIAL` | the source backs some of it and is silent on the rest | soften or cut the unsupported clause |
| `UNSUPPORTED` | no passage addresses the claim at all | find a source or delete it |
| `CONTRADICTED` | a passage states something incompatible | **fix it, the source says otherwise** |

The last row is the one worth having. "The corpus does not mention this" and "the corpus says the opposite" need completely different responses, and a single confidence number cannot tell you which one you are looking at.

One more distinction that turns out to matter a lot. If a claim says "six new bus stations" and the source gives no station count anywhere, that is the source being **silent**, not the source **disagreeing**. Footnote only calls a figure contradicted when the passage offers a comparable figure, decided by whether the two numbers share an anchor word. Reporting silence as a contradiction is the fastest way for a verification tool to lose a reader's trust.

## Install

```bash
pip install footnote-verify              # core: CLI, engine, txt/md/html/json sources
pip install "footnote-verify[pdf]"       # + PDF sources
pip install "footnote-verify[mcp]"       # + the Model Context Protocol server
pip install "footnote-verify[all]"       # everything
```

Python 3.11 or newer. The CLI is `footnote`. Quote the brackets, or your shell will treat them as a glob.

The default path needs no API key, no model download and no network. It is deterministic: the same input gives the same verdicts every time.

## Quickstart

```bash
footnote demo --open                     # run the bundled example, open the report

footnote check answer.md --sources ./docs
footnote check answer.md -s ./docs -s ./notes.pdf --report out.html --open
echo "Revenue grew 34%." | footnote check - --sources ./docs
footnote check answer.md -s ./docs --json results.json --quiet
```

Gate a build on it:

```bash
footnote check generated-summary.md --sources ./source-of-truth --fail-under 0.9
```

Exits non-zero when the score falls short, so a docs pipeline can refuse to publish a page whose claims have drifted from the material it was written from.

## How it works

```
text ──▶ segment ──▶ retrieve ──▶ align ──▶ judge ──▶ verdicts + citations
```

**Segment.** The text is split into claims, abbreviation- and decimal-aware, so `Oct. 2025` and `3.5 percent` stay in one piece. Headings, questions, code fences, table rules and bare fragments assert nothing and are skipped rather than scored, because scoring them moves the faithfulness number around without telling anyone anything.

**Retrieve.** Sources are indexed at sentence level, with windows of one to three consecutive sentences so a claim spanning a sentence boundary can still match. Candidates are ranked by IDF-weighted coverage of the claim's content terms, so a passage earns its rank by containing the rare, discriminating words rather than by being long. Recall matters more than precision here: a span that is never retrieved can never be cited.

**Align.** The winning window is narrowed to the tightest character range that still accounts for the matched terms. This is what turns "somewhere on page 4" into offsets you can highlight.

**Judge.** Coverage is combined with two checks similarity cannot do: figures are reconciled by value, and polarity is compared so a negation flip is caught. A passage that covers the claim's wording but disagrees on a number is reported as a contradiction, not as weak support.

## Measured quality

A test suite proves the code does what it was written to do. It says nothing about whether what it was written to do actually works. So the repository ships a hand-labelled dataset and a harness that scores the real pipeline against it.

```bash
footnote eval                     # reproduces every number below
```

**68 labelled claims, 8 source corpora, deterministic judge, no API key:**

| Verdict | Precision | Recall | F1 | n |
|---|---|---|---|---|
| supported | 0.89 | 0.97 | 0.93 | 32 |
| contradicted | 0.94 | 0.75 | 0.83 | 20 |
| unsupported | 0.73 | 1.00 | 0.84 | 8 |
| partial | 0.67 | 0.50 | 0.57 | 8 |

**Hallucination detection** (positive class = anything not fully supported, which is the decision a user actually makes):

| | |
|---|---|
| precision | **0.97** |
| recall | **0.89** |
| F1 | **0.93** |
| accuracy (4-way) | 0.85 |
| macro F1 | 0.79 |

One false positive in 68. Four missed problems in 36.

These numbers are asserted in the test suite, so a change that degrades verification quality fails CI rather than passing quietly.

### Where it fails, and why

Being specific about this is more useful than a bigger headline number. All ten disagreements with the labels fall into four groups:

1. **Entity substitution** (3 cases). "data centres in Dublin and **Tokyo**" against a source saying "Dublin and **Singapore**" is scored as supported. Every content word appears in the passage, and a bag-of-words judge has no way to know one of them was swapped. This is the clearest case for the LLM judge.
2. **Antonyms and temporal flips** (2 cases). "Background tasks run **before** the response is sent" against "**after** the response has been sent". No number disagrees and no negation cue fires.
3. **Relations that are not stated** (2 cases). If a claim asserts that A was driven by B, and the source mentions A and mentions B but never links them, coverage is satisfied. Bag-of-words cannot represent the relation.
4. **Vocabulary gaps** (1 case). "Okafor **scored**" against "Okafor **took the lead** with a header". Stemming does not bridge synonyms.

The remaining two are `partial` boundary calls, which is the hardest and least consequential class.

## The LLM judge is optional

Off unless a credential is present on your machine. When it is, the model is kept on a short leash:

- It never reads the corpus. It sees the same retrieved passages the deterministic judge saw, so token cost stays bounded and the two are directly comparable in the harness.
- **A quote it cannot produce verbatim is not cited.** Every LLM answer is re-anchored by locating its quote in the passage. If the quote is not there, the citation is dropped and the claim is downgraded. A fabricated citation is impossible by construction, which is not a promise you can make about a model that is asked to emit page numbers.
- A consent notice prints before the first byte leaves the machine.
- Credentials resolve from the environment or a local `.env`, in that order, and are never written anywhere.

```bash
cp .env.example .env        # add OPENAI_API_KEY or ANTHROPIC_API_KEY
footnote check answer.md -s ./docs --judge llm --model gpt-4o-mini
```

There is no SDK dependency. The provider adapter is about a hundred lines over the standard library and speaks to OpenAI-compatible endpoints (OpenAI, Groq, Together, OpenRouter, a local Ollama) and to Anthropic.

## Model Context Protocol server

Let an agent check its own output mid-task instead of shipping an unverified answer.

```bash
pip install "footnote-verify[mcp]"
footnote mcp --sources ./docs
```

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "footnote": { "command": "footnote", "args": ["mcp", "--sources", "/path/to/docs"] }
  }
}
```

Two tools: `verify_against_sources` for a folder on disk, and `verify_against_text` for passages already in context, such as results a search tool just returned. Both return the faithfulness score, per-claim verdicts, and a `needs_attention` list ordered worst first, so the model can rewrite the specific sentence that failed.

## Python API

```python
from footnote import check, write_report

result = check("Revenue grew 34% to $2.1B.", ["./sources"])

print(result.faithfulness)          # 0.91
print(result.counts())              # {'supported': 4, 'partial': 0, ...}

for claim in result.problems():     # worst first
    print(claim.verdict, claim.claim.text)
    if claim.best:
        print("  ", claim.best.span.locator())   # report.pdf p.4 chars 1180-1223
        print("  ", claim.best.span.text)        # the exact supporting text

write_report(result, "out.html")
```

Every `SourceSpan` is a half-open character range into the loaded document, so `document.text[span.start:span.end] == span.text` always holds. That invariant is what makes the highlighting trustworthy, and it is asserted in the tests.

## CLI reference

| Command | Purpose |
|---|---|
| `footnote check TEXT -s SOURCES` | verify a file, a literal string, or `-` for stdin |
| `footnote eval` | score the judges against a labelled dataset |
| `footnote demo` | run the bundled example end to end |
| `footnote mcp` | serve the Model Context Protocol tools on stdio |

Useful flags for `check`: `--report out.html`, `--json out.json`, `--open`, `--judge lexical|llm|auto`, `--threshold`, `--fail-under`, `--top-k`, `--verbose`, `--quiet`.

## Sources it can read

Out of the box, with no extra dependencies: `.txt`, `.md`, `.html`, `.json`, `.jsonl`, `.csv`, `.yaml`, and common source files. HTML is stripped to readable text with the standard library, and JSON is flattened to `path: value` lines so prose buried inside it is still findable. PDF needs the `pdf` extra and carries page numbers through into citations.

## Limitations

- **Bag-of-words judging.** The deterministic judge cannot see synonyms, entity swaps or unstated relations. See the failure analysis above. The LLM judge covers these; the deterministic one is the free, offline, reproducible baseline.
- **The dataset is hand-built**, not a public benchmark. It is 68 claims written to span the failure modes deliberately, which makes it useful for regression testing and honest for relative comparison, but it is not a leaderboard result. The loader accepts external JSONL in the same shape if you want to run it against something larger.
- **English only.** The stemmer, stop list and negation cues are English.
- **No cross-document reasoning.** Each claim is judged against the best single window. A claim that is only true when two documents are combined will read as partial.

## Development

```bash
git clone https://github.com/adwitiyashukla/footnote.git
cd footnote
pip install -e ".[all]"
pip install pytest pytest-cov ruff mypy

ruff check src tests && mypy && pytest
footnote eval
```

CI runs lint, `mypy --strict`, the test suite and the evaluation harness on Python 3.11 and 3.12.

## License

[MIT](LICENSE)
