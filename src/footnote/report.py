from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from footnote.models import CheckResult, Document, Verdict

__all__ = ["render_report", "write_report"]

_VERDICT_LABEL = {
    Verdict.SUPPORTED: "Supported",
    Verdict.PARTIAL: "Partial",
    Verdict.UNSUPPORTED: "Unsupported",
    Verdict.CONTRADICTED: "Contradicted",
}

_CSS = """
:root {
  --bg: #0f1115; --panel: #161920; --panel-2: #1c2028; --line: #272c37;
  --text: #e6e9ef; --muted: #99a1b3; --faint: #6b7385;
  --supported: #3fb950; --partial: #d29922; --unsupported: #7d8590; --contradicted: #f85149;
  --accent: #4c8dff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
a { color: var(--accent); }
header {
  padding: 28px 32px 22px; border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, #141821, var(--bg));
}
h1 { margin: 0 0 4px; font-size: 19px; letter-spacing: -0.01em; }
h1 span { color: var(--faint); font-weight: 400; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.metrics { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; }
.metric {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; min-width: 116px;
}
.metric .k { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--faint); }
.metric .v { font-size: 22px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
.metric.score .v { font-size: 28px; }
.bar { height: 6px; border-radius: 3px; background: var(--panel-2); overflow: hidden; display: flex; margin-top: 14px; }
.bar i { display: block; height: 100%; }
main { display: grid; grid-template-columns: minmax(340px, 42%) 1fr; min-height: calc(100vh - 210px); }
.pane { padding: 22px 26px; overflow-y: auto; max-height: calc(100vh - 210px); }
.pane.sources { border-left: 1px solid var(--line); background: #12151c; }
.pane h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--faint); margin: 0 0 14px; font-weight: 600;
}
.claim {
  border: 1px solid var(--line); border-left: 3px solid var(--unsupported);
  border-radius: 9px; background: var(--panel); padding: 13px 15px;
  margin-bottom: 11px; cursor: pointer; transition: border-color .12s, background .12s;
}
.claim:hover { background: var(--panel-2); }
.claim.active { border-color: var(--accent); background: var(--panel-2); }
.claim.supported { border-left-color: var(--supported); }
.claim.partial { border-left-color: var(--partial); }
.claim.unsupported { border-left-color: var(--unsupported); }
.claim.contradicted { border-left-color: var(--contradicted); }
.claim .top { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 7px; }
.tag {
  font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 5px; white-space: nowrap;
}
.tag.supported { background: rgba(63,185,80,.14); color: var(--supported); }
.tag.partial { background: rgba(210,153,34,.14); color: var(--partial); }
.tag.unsupported { background: rgba(125,133,144,.16); color: var(--unsupported); }
.tag.contradicted { background: rgba(248,81,73,.14); color: var(--contradicted); }
.claim .score { color: var(--faint); font-size: 12px; font-variant-numeric: tabular-nums; }
.claim .text { font-size: 14px; }
.claim .why { color: var(--muted); font-size: 12.5px; margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--line); }
.claim .cite { color: var(--faint); font-size: 11.5px; margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.doc { margin-bottom: 22px; }
.doc > summary {
  cursor: pointer; color: var(--muted); font-size: 12.5px; padding: 7px 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.doc > summary::marker { color: var(--faint); }
.body {
  white-space: pre-wrap; word-wrap: break-word; font-size: 13.5px; line-height: 1.75;
  color: #c3c9d6; background: var(--panel); border: 1px solid var(--line);
  border-radius: 9px; padding: 15px 17px;
}
mark {
  background: rgba(125,133,144,.2); color: inherit; border-radius: 3px;
  padding: 1px 2px; border-bottom: 2px solid var(--unsupported); scroll-margin: 90px;
}
mark.supported { background: rgba(63,185,80,.16); border-bottom-color: var(--supported); }
mark.partial { background: rgba(210,153,34,.16); border-bottom-color: var(--partial); }
mark.contradicted { background: rgba(248,81,73,.16); border-bottom-color: var(--contradicted); }
mark.active { background: rgba(76,141,255,.3); border-bottom-color: var(--accent); color: #fff; }
.empty { color: var(--faint); font-size: 13px; font-style: italic; }
.warn {
  background: rgba(210,153,34,.09); border: 1px solid rgba(210,153,34,.3);
  border-radius: 8px; padding: 10px 13px; margin-bottom: 14px;
  color: var(--partial); font-size: 12.5px;
}
footer { padding: 16px 32px; border-top: 1px solid var(--line); color: var(--faint); font-size: 12px; }
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; }
  .pane { max-height: none; }
  .pane.sources { border-left: none; border-top: 1px solid var(--line); }
}
"""

_JS = """
const claims = document.querySelectorAll('.claim');
function select(el) {
  document.querySelectorAll('.claim.active').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('mark.active').forEach(n => n.classList.remove('active'));
  el.classList.add('active');
  const id = el.dataset.claim;
  const marks = document.querySelectorAll('mark[data-claims~="' + id + '"]');
  marks.forEach(m => m.classList.add('active'));
  if (marks.length) {
    const holder = marks[0].closest('details');
    if (holder && !holder.open) holder.open = true;
    marks[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}
claims.forEach(el => el.addEventListener('click', () => select(el)));
document.addEventListener('keydown', e => {
  if (e.key !== 'j' && e.key !== 'k') return;
  const list = Array.from(claims);
  const current = document.querySelector('.claim.active');
  let i = current ? list.indexOf(current) : -1;
  i = e.key === 'j' ? Math.min(i + 1, list.length - 1) : Math.max(i - 1, 0);
  if (list[i]) { select(list[i]); list[i].scrollIntoView({ block: 'nearest' }); }
});
if (claims.length) select(claims[0]);
"""


@dataclass(frozen=True, slots=True)
class _Mark:
    start: int
    end: int
    verdict: Verdict
    claim_ids: tuple[str, ...]


def _collect_marks(result: CheckResult) -> dict[str, list[_Mark]]:
    raw: dict[str, list[tuple[int, int, Verdict, str]]] = {}
    for entry in result.claims:
        best = entry.best
        if best is None:
            continue
        raw.setdefault(best.span.doc_id, []).append(
            (best.span.start, best.span.end, entry.verdict, entry.claim.id)
        )

    marks: dict[str, list[_Mark]] = {}
    for doc_id, spans in raw.items():
        merged: list[_Mark] = []
        for start, end, verdict, claim_id in sorted(spans):
            if merged and start < merged[-1].end:
                last = merged[-1]
                merged[-1] = _Mark(
                    start=last.start,
                    end=max(last.end, end),
                    verdict=last.verdict if last.verdict.rank <= verdict.rank else verdict,
                    claim_ids=(*last.claim_ids, claim_id),
                )
            else:
                merged.append(_Mark(start, end, verdict, (claim_id,)))
        marks[doc_id] = merged
    return marks


def _render_document(document: Document, marks: list[_Mark]) -> str:
    pieces: list[str] = []
    cursor = 0
    for mark in marks:
        start = max(cursor, min(mark.start, len(document.text)))
        end = max(start, min(mark.end, len(document.text)))
        if start > cursor:
            pieces.append(html.escape(document.text[cursor:start]))
        ids = " ".join(mark.claim_ids)
        pieces.append(
            f'<mark class="{mark.verdict.value}" data-claims="{html.escape(ids)}">'
            f"{html.escape(document.text[start:end])}</mark>"
        )
        cursor = end
    pieces.append(html.escape(document.text[cursor:]))
    return "".join(pieces)


def _claims_html(result: CheckResult) -> str:
    if not result.claims:
        return '<p class="empty">No checkable claims were found in the input text.</p>'
    out: list[str] = []
    for entry in result.claims:
        best = entry.best
        cite = ""
        if best is not None:
            cite = f'<div class="cite">{html.escape(best.span.locator())}</div>'
        why = ""
        detail = (best.conflict if best and best.conflict else None) or entry.rationale
        if detail:
            why = f'<div class="why">{html.escape(detail)}</div>'
        out.append(
            f'<div class="claim {entry.verdict.value}" data-claim="{html.escape(entry.claim.id)}">'
            f'<div class="top"><span class="tag {entry.verdict.value}">'
            f"{_VERDICT_LABEL[entry.verdict]}</span>"
            f'<span class="score">{entry.score:.2f}</span></div>'
            f'<div class="text">{html.escape(entry.claim.text)}</div>'
            f"{why}{cite}</div>"
        )
    return "\n".join(out)


def _sources_html(result: CheckResult) -> str:
    marks = _collect_marks(result)
    if not result.documents:
        return '<p class="empty">No source documents were loaded.</p>'
    cited = [d for d in result.documents if marks.get(d.doc_id)]
    others = [d for d in result.documents if not marks.get(d.doc_id)]

    out: list[str] = []
    for document in cited:
        count = len(marks[document.doc_id])
        label = "citation" if count == 1 else "citations"
        out.append(
            f'<details class="doc" open><summary>{html.escape(document.doc_id)} '
            f"({count} {label})</summary>"
            f'<div class="body">{_render_document(document, marks[document.doc_id])}</div>'
            f"</details>"
        )
    for document in others:
        out.append(
            f'<details class="doc"><summary>{html.escape(document.doc_id)} '
            f"(no citations)</summary>"
            f'<div class="body">{html.escape(document.text)}</div></details>'
        )
    return "\n".join(out)


def _metrics_html(result: CheckResult) -> str:
    counts = result.counts()
    total = max(len(result.claims), 1)
    segments = "".join(
        f'<i style="width:{counts[v.value] / total * 100:.4f}%;background:var(--{v.value})"></i>'
        for v in Verdict
        if counts[v.value]
    )
    cards = [
        ("Faithfulness", f"{result.faithfulness:.2f}", "metric score"),
        ("Claims", str(len(result.claims)), "metric"),
        ("Supported", str(counts["supported"]), "metric"),
        ("Partial", str(counts["partial"]), "metric"),
        ("Unsupported", str(counts["unsupported"]), "metric"),
        ("Contradicted", str(counts["contradicted"]), "metric"),
        ("Judge", html.escape(result.judge), "metric"),
    ]
    body = "".join(
        f'<div class="{cls}"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v, cls in cards
    )
    return f'<div class="metrics">{body}</div><div class="bar">{segments}</div>'


def render_report(result: CheckResult, *, title: str = "Footnote report") -> str:
    warnings = "".join(f'<div class="warn">{html.escape(w)}</div>' for w in result.warnings)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    docs = len(result.documents)
    payload = html.escape(json.dumps(result.to_dict()), quote=True)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(title)} <span>/ claim-level source verification</span></h1>
  <div class="sub">{len(result.claims)} claims checked against {docs} source
    {"document" if docs == 1 else "documents"}, generated {generated}</div>
  {_metrics_html(result)}
</header>
<main>
  <section class="pane claims">
    <h2>Claims, click to locate, or press j and k</h2>
    {warnings}
    {_claims_html(result)}
  </section>
  <section class="pane sources">
    <h2>Sources, highlighted spans are the cited evidence</h2>
    {_sources_html(result)}
  </section>
</main>
<footer>
  Generated by <a href="https://github.com/adwitiyashukla/footnote">Footnote</a>.
  Every highlight is an exact character range in the source file, not a paraphrase.
</footer>
<script type="application/json" id="footnote-data">{payload}</script>
<script>{_JS}</script>
</body>
</html>
"""


def write_report(
    result: CheckResult,
    path: str | Path = "footnote-report.html",
    *,
    title: str = "Footnote report",
) -> Path:
    destination = Path(path)
    if destination.parent != Path():
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(result, title=title), encoding="utf-8")
    return destination
