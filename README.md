# Footnote

**Checks whether AI-generated text is actually supported by your source documents, one claim at a time.**

[![CI](https://github.com/adwitiyashukla/footnote/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/footnote/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-live%20report-4c8dff.svg)](https://adwitiyashukla.github.io/footnote/example-report.html)

## Why I built this

I kept running into the same annoying problem. I'd give a model a few PDFs and ask it to summarise them, and I'd get back a paragraph that *looked* completely fine. Some of it was really in the documents. Some of it wasn't. And there was no way to tell which was which without going back and re-reading everything myself, which defeats the whole point.

So I wanted a tool that goes through the output sentence by sentence and tells me: this bit is in your sources and here's exactly where, this bit isn't, and this bit actually contradicts what your source says.

That's what Footnote does.

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
faithfulness 0.73  (lexical judge, 1 source)
```

Add `--report out.html` and you get a page where clicking a claim highlights the exact sentence it came from.

### [Try the live example report](https://adwitiyashukla.github.io/footnote/example-report.html)

Click any claim on the left, and the characters it rests on light up in the source on the right. Nothing to install.

## The bit I found most interesting

My first instinct was to just compare the claim and the source passage with embeddings and call anything above a threshold "supported". Then I tried it on these two sentences:

> Free cash flow was 410 million dollars.
> Free cash flow was 510 million dollars.

They're basically identical as strings, and their embeddings are nearly identical too. So a similarity threshold says the second one is fine. But it's wrong, and a wrong number is probably the most common way an AI summary goes bad.

So instead of comparing text, I parse the numbers into actual values and compare those. `$2.1B`, `2.1 billion` and `2,100,000,000` all become the same number. `34%` and `34 percent` keep the same unit. `three participants` gets compared against `two participants`.

I also ended up splitting the "this is wrong" case into two, because they turned out to need different fixes:

| Verdict | What it means | What you'd do |
|---|---|---|
| `SUPPORTED` | it's all there in the source | nothing |
| `PARTIAL` | part of it is backed up, the rest isn't mentioned | soften or cut the extra bit |
| `UNSUPPORTED` | nothing in the sources is about this at all | find a source or delete it |
| `CONTRADICTED` | a passage says something incompatible | fix it, your source disagrees |

There's one more thing I got wrong at first and had to go back and fix. If a claim says "six new bus stations" and the source never gives a station count anywhere, my first version called that a contradiction. It isn't. The source is just silent. So now a number only counts as contradicted if the passage actually offers a comparable number, which I decide by checking whether the two numbers share a nearby word. That change is in the commit history if you want to see it.

## Install

Not on PyPI yet, so install it from here:

```bash
pip install git+https://github.com/adwitiyashukla/footnote.git
```

Or clone it, which is easier if you want to poke at the code:

```bash
git clone https://github.com/adwitiyashukla/footnote.git
cd footnote
pip install -e ".[all]"
```

Needs Python 3.11 or newer. The command is `footnote`.

Optional extras: `pdf` for PDF files, `mcp` for the Model Context Protocol server, `all` for both. Without them it still handles txt, markdown, HTML and JSON, and the only dependencies are `typer` and `rich`.

## Quickstart

```bash
footnote demo --open                     # runs the built-in example and opens the report

footnote check answer.md --sources ./docs
footnote check answer.md -s ./docs -s ./notes.pdf --report out.html --open
echo "Revenue grew 34%." | footnote check - --sources ./docs
footnote check answer.md -s ./docs --json results.json --quiet
```

You can also use it in a CI pipeline:

```bash
footnote check generated-summary.md --sources ./source-of-truth --fail-under 0.9
```

That exits with an error code if the score is too low, so a docs build can refuse to publish a page whose claims have drifted away from the source material.

## How it works

```
text -> segment -> retrieve -> align -> judge -> verdicts + citations
```

**Segment.** Split the text into individual claims. This was fiddlier than I expected. You can't just split on full stops, because `Oct. 2025` and `3.5 percent` break. I also skip headings, questions, code blocks and short fragments, since those don't actually claim anything and scoring them just adds noise to the final number.

**Retrieve.** Index the source documents by sentence, then build candidate windows of one to three sentences so a claim that spans a sentence break can still match. Windows are ranked by IDF-weighted overlap with the claim's words, so rare, specific words count more than common ones. I care much more about recall than precision here, because if the right passage never gets retrieved then the claim gets marked unsupported no matter how good the rest of the pipeline is.

**Align.** Narrow the winning window down to the smallest character range that still covers the matched words. This is the bit that turns "somewhere on page 4" into offsets you can actually highlight.

**Judge.** Combine the word overlap with two checks that similarity can't do: reconcile the numbers by value, and compare polarity so a flipped negation gets caught.

## Results

Unit tests tell you the code runs. They don't tell you whether the thing actually works. So I hand-labelled a small dataset and wrote a harness that scores the real pipeline against it.

```bash
footnote eval                     # reproduces the numbers below
```

**68 labelled claims across 8 source corpora, offline judge, no API key:**

| Verdict | Precision | Recall | F1 | n |
|---|---|---|---|---|
| supported | 0.89 | 0.97 | 0.93 | 32 |
| contradicted | 0.94 | 0.75 | 0.83 | 20 |
| unsupported | 0.73 | 1.00 | 0.84 | 8 |
| partial | 0.67 | 0.50 | 0.57 | 8 |

If you collapse it down to the question a user actually cares about ("should I go and check this claim?"):

| | |
|---|---|
| precision | 0.97 |
| recall | 0.89 |
| F1 | 0.93 |
| accuracy (4 classes) | 0.85 |
| macro F1 | 0.79 |

So one false alarm out of 68, and it misses 4 of the 36 claims that had something wrong with them.

I also put these thresholds into the test suite, so if I change the scoring later and it gets worse, CI fails instead of quietly letting it slide.

## What it gets wrong

I think this section is more useful than a bigger headline number, so here are all 10 cases it disagreed with my labels on:

1. **Swapped names or places** (3 cases). "data centres in Dublin and Tokyo" against a source that says "Dublin and Singapore" comes back as supported. Every individual word is in the passage, and my offline judge just counts words, so it has no idea one got swapped.
2. **Opposites** (2 cases). "Background tasks run before the response is sent" against "after the response has been sent". No number disagrees and no negation word appears, so nothing trips.
3. **Relationships that aren't actually stated** (2 cases). If a claim says A happened because of B, and the source mentions A and mentions B but never connects them, the word overlap is satisfied anyway.
4. **Synonyms** (1 case). "Okafor scored" against "Okafor took the lead with a header". My stemmer doesn't bridge that gap.

The other 2 are `partial` borderline calls, which honestly are hard for me to label consistently myself.

Groups 1 and 2 are the main reason the optional LLM judge exists.

## Optional LLM judge

Off by default. It only turns on if it finds an API key on your machine.

The thing I was most careful about here: the model doesn't get to invent citations. It only sees the same candidate passages the offline judge saw, and whatever it quotes gets looked up in that passage afterwards. If I can't find its quote in the text, I throw the citation away and downgrade the claim. So a made-up citation can't get through, which felt important for a tool whose whole job is checking things.

```bash
cp .env.example .env        # add OPENAI_API_KEY or ANTHROPIC_API_KEY
footnote check answer.md -s ./docs --judge llm --model gpt-4o-mini
```

It also prints a notice before it sends anything, so you know when your source text is about to leave your machine.

No SDK needed. I wrote a small adapter over `urllib` that talks to OpenAI-compatible endpoints (OpenAI, Groq, Together, OpenRouter, local Ollama) and to Anthropic. It's about a hundred lines and it means the base install stays tiny.

## MCP server

This lets an AI agent check its own output before showing it to you.

```bash
pip install -e ".[mcp]"
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

Two tools: `verify_against_sources` for a folder on disk, and `verify_against_text` for passages already in the conversation. Both return the score, the per-claim verdicts, and a `needs_attention` list sorted worst first, so the model can go and fix the specific sentence that failed.

## Using it from Python

```python
from footnote import check, write_report

result = check("Revenue grew 34% to $2.1B.", ["./sources"])

print(result.faithfulness)          # 0.91
print(result.counts())              # {'supported': 4, 'partial': 0, ...}

for claim in result.problems():     # worst first
    print(claim.verdict, claim.claim.text)
    if claim.best:
        print("  ", claim.best.span.locator())   # report.pdf p.4 chars 1180-1223
        print("  ", claim.best.span.text)        # the actual supporting text

write_report(result, "out.html")
```

One rule I stuck to everywhere: a `SourceSpan` is always a real character range into the loaded document, so `document.text[span.start:span.end] == span.text` is always true. That's what makes the highlighting reliable, and there's a test for it.

## Commands

| Command | What it does |
|---|---|
| `footnote check TEXT -s SOURCES` | check a file, a string, or `-` for stdin |
| `footnote eval` | run the evaluation harness |
| `footnote demo` | run the built-in example |
| `footnote mcp` | start the MCP server |

Handy flags for `check`: `--report out.html`, `--json out.json`, `--open`, `--judge lexical|llm|auto`, `--threshold`, `--fail-under`, `--top-k`, `--verbose`, `--quiet`.

## File types it reads

With no extra dependencies: `.txt`, `.md`, `.html`, `.json`, `.jsonl`, `.csv`, `.yaml` and most plain source files. HTML gets stripped down to readable text using the standard library, and JSON gets flattened into `path: value` lines so text hidden inside it is still findable. PDFs need the `pdf` extra and carry page numbers into the citations.

## Limitations

Being upfront about these:

- **The offline judge just counts words.** It can't see synonyms, swapped names or implied relationships. See the failure section above. The LLM judge handles those, but the offline one is the free, reproducible baseline.
- **My eval set is hand-made.** It's 68 claims I wrote specifically to cover the different failure modes, not a standard benchmark. That makes it useful for catching regressions, but it isn't a leaderboard score and I don't want to pretend it is. The loader takes external JSONL in the same format if you want to run it on something bigger.
- **English only.** The stemmer, stopword list and negation words are all English.
- **One document at a time.** Each claim gets judged against the single best passage. A claim that's only true if you combine two documents will show up as partial.

## Running it locally

```bash
git clone https://github.com/adwitiyashukla/footnote.git
cd footnote
pip install -e ".[all]"
pip install pytest pytest-cov ruff mypy

ruff check src tests && mypy && pytest
footnote eval
```

CI runs lint, `mypy --strict`, the tests and the eval harness on Python 3.11 and 3.12.

## Author

Adwitiya Shukla

## License

[MIT](LICENSE)
