from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfeeder.models import Document, Verdict

if TYPE_CHECKING:
    from rich.console import Console

__all__ = [
    "EvalExample",
    "Metrics",
    "demo_case",
    "evaluate",
    "load_dataset",
    "render_metrics",
]


@dataclass(frozen=True, slots=True)
class EvalExample:
    id: str
    claim: str
    label: Verdict
    documents: tuple[Document, ...]
    case: str


def _documents(raw: dict[str, Any], case: str) -> tuple[Document, ...]:
    out: list[Document] = []
    for doc_id, body in raw.items():
        text = "\n\n".join(body) if isinstance(body, list) else str(body)
        out.append(Document(doc_id=doc_id, path=f"<{case}>/{doc_id}", text=text))
    return tuple(out)


def _bundled_path() -> Path:
    return Path(str(files("llmfeeder").joinpath("data/eval.json")))


def load_dataset(path: str | Path | None = None) -> list[EvalExample]:
    source = Path(path) if path else _bundled_path()
    if not source.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")

    raw = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        cases = parsed["cases"] if isinstance(parsed, dict) else parsed

    examples: list[EvalExample] = []
    for case in cases:
        name = str(case["case"])
        documents = _documents(case["documents"], name)
        for item in case["claims"]:
            examples.append(
                EvalExample(
                    id=str(item.get("id") or f"{name}-{len(examples) + 1}"),
                    claim=str(item["text"]),
                    label=Verdict(str(item["label"]).lower()),
                    documents=documents,
                    case=name,
                )
            )
    return examples


@dataclass
class Metrics:
    judge: str
    total: int = 0
    correct: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    mistakes: list[dict[str, str]] = field(default_factory=list)
    misattributed: int = 0

    def __post_init__(self) -> None:
        if not self.confusion:
            self.confusion = {g.value: {p.value: 0 for p in Verdict} for g in Verdict}

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def _counts(self, label: str) -> tuple[int, int, int]:
        true_positive = self.confusion[label][label]
        false_negative = sum(self.confusion[label].values()) - true_positive
        false_positive = sum(self.confusion[g][label] for g in self.confusion) - true_positive
        return true_positive, false_positive, false_negative

    def per_label(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for verdict in Verdict:
            label = verdict.value
            tp, fp, fn = self._counts(label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            out[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(sum(self.confusion[label].values())),
            }
        return out

    @property
    def macro_f1(self) -> float:
        scored = [m for m in self.per_label().values() if m["support"]]
        return sum(m["f1"] for m in scored) / len(scored) if scored else 0.0

    def detection(self) -> dict[str, float]:
        tp = fp = fn = tn = 0
        for gold, row in self.confusion.items():
            gold_positive = gold != Verdict.SUPPORTED.value
            for predicted, count in row.items():
                predicted_positive = predicted != Verdict.SUPPORTED.value
                if gold_positive and predicted_positive:
                    tp += count
                elif not gold_positive and predicted_positive:
                    fp += count
                elif gold_positive and not predicted_positive:
                    fn += count
                else:
                    tn += count
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": float(tp),
            "false_positive": float(fp),
            "false_negative": float(fn),
            "true_negative": float(tn),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge,
            "examples": self.total,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "citation_misattribution": self.misattributed,
            "per_label": {
                k: {m: round(v, 4) for m, v in scores.items()}
                for k, scores in self.per_label().items()
            },
            "detection": {k: round(v, 4) for k, v in self.detection().items()},
            "confusion": self.confusion,
            "mistakes": self.mistakes,
        }


def evaluate(
    examples: list[EvalExample],
    *,
    judge: str = "lexical",
    model: str | None = None,
) -> Metrics:
    from llmfeeder.verify import check_documents

    metrics = Metrics(judge=judge)
    resolved_name = judge
    for example in examples:
        result = check_documents(
            example.claim, example.documents, judge=judge, model=model, min_words=1
        )
        predicted = result.claims[0].verdict if result.claims else Verdict.UNSUPPORTED
        gold = example.label
        resolved_name = result.judge

        metrics.total += 1
        metrics.confusion[gold.value][predicted.value] += 1
        if predicted is gold:
            metrics.correct += 1
        else:
            best = result.claims[0].best if result.claims else None
            metrics.mistakes.append(
                {
                    "id": example.id,
                    "case": example.case,
                    "claim": example.claim,
                    "gold": gold.value,
                    "predicted": predicted.value,
                    "cited": best.span.locator() if best else "nothing",
                }
            )
    metrics.judge = resolved_name
    return metrics


def render_metrics(metrics: Metrics, console: Console) -> None:
    from rich.table import Table

    per_label = Table(title=None, box=None, pad_edge=False, header_style="dim")
    per_label.add_column("verdict", width=14)
    per_label.add_column("precision", justify="right")
    per_label.add_column("recall", justify="right")
    per_label.add_column("f1", justify="right")
    per_label.add_column("support", justify="right")
    for label, scores in metrics.per_label().items():
        if not scores["support"]:
            continue
        per_label.add_row(
            label,
            f"{scores['precision']:.2f}",
            f"{scores['recall']:.2f}",
            f"{scores['f1']:.2f}",
            f"{int(scores['support'])}",
        )

    console.print("\n[bold]per verdict[/bold]")
    console.print(per_label)

    detection = metrics.detection()
    console.print("\n[bold]hallucination detection[/bold] [dim](positive = not supported)[/dim]")
    console.print(
        f"  precision [bold]{detection['precision']:.2f}[/bold]   "
        f"recall [bold]{detection['recall']:.2f}[/bold]   "
        f"f1 [bold]{detection['f1']:.2f}[/bold]   "
        f"[dim]missed {int(detection['false_negative'])} of "
        f"{int(detection['true_positive'] + detection['false_negative'])}[/dim]"
    )
    console.print(
        f"\naccuracy [bold]{metrics.accuracy:.2f}[/bold]   "
        f"macro f1 [bold]{metrics.macro_f1:.2f}[/bold]   "
        f"[dim]{metrics.total} labelled claims, {metrics.judge} judge[/dim]"
    )
    if metrics.mistakes:
        console.print(f"\n[dim]{len(metrics.mistakes)} disagreements with the labels[/dim]")


def demo_case() -> tuple[str, list[Document]]:
    examples = load_dataset()
    case_name = examples[0].case
    documents = list(examples[0].documents)
    answer = " ".join(e.claim for e in examples if e.case == case_name)
    return answer, documents
