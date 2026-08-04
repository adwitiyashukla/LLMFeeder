"""Command line interface.

Designed to be usable two ways: read by a person at a terminal, and read by a CI
job. ``--fail-under`` turns the faithfulness score into an exit code, so a
documentation build or a generated report can be gated on whether its claims still
match the sources it was written from.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from footnote import __version__
from footnote.models import CheckResult, Verdict

app = typer.Typer(
    name="footnote",
    help="Check whether generated text is actually supported by your sources.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

_STYLE = {
    Verdict.SUPPORTED: "green",
    Verdict.PARTIAL: "yellow",
    Verdict.UNSUPPORTED: "bright_black",
    Verdict.CONTRADICTED: "red",
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"footnote {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Footnote: claim-level source verification for generated text."""


@app.command()
def version() -> None:
    """Print the Footnote version."""
    console.print(f"footnote {__version__}")


def _read_answer(answer: str) -> str:
    """Accept a file path, a literal string, or ``-`` for standard input."""
    if answer == "-":
        return sys.stdin.read()
    path = Path(answer)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    if len(answer) < 260 and not path.exists() and ("/" in answer or "\\" in answer):
        raise typer.BadParameter(f"no such file: {answer}")
    return answer


def _render(result: CheckResult, *, verbose: bool) -> None:
    table = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    table.add_column("", width=13)
    table.add_column("score", justify="right", width=5)
    table.add_column("claim", overflow="fold")
    table.add_column("citation", overflow="fold", style="dim")

    shown = result.claims if verbose else [c for c in result.claims if not c.verdict.is_grounded]
    for entry in shown:
        best = entry.best
        citation = best.span.locator() if best else "no supporting span found"
        label = Text(entry.verdict.value.upper(), style=_STYLE[entry.verdict])
        claim_text = entry.claim.text
        if len(claim_text) > 96:
            claim_text = claim_text[:93] + "..."
        table.add_row(label, f"{entry.score:.2f}", claim_text, citation)

    if shown:
        heading = "all claims" if verbose else "claims needing attention"
        console.print(f"\n[bold]{heading}[/bold]")
        console.print(table)
    elif result.claims:
        console.print("\n[green]every claim is supported by the sources.[/green]")

    counts = result.counts()
    summary = (
        f"{len(result.claims)} claims  "
        f"[green]{counts['supported']} supported[/green]  "
        f"[yellow]{counts['partial']} partial[/yellow]  "
        f"[bright_black]{counts['unsupported']} unsupported[/bright_black]  "
        f"[red]{counts['contradicted']} contradicted[/red]"
    )
    console.print(f"\n{summary}")
    count = len(result.documents)
    console.print(
        f"faithfulness [bold]{result.faithfulness:.2f}[/bold] "
        f"[dim]({result.judge} judge, {count} source{'' if count == 1 else 's'})[/dim]"
    )


@app.command()
def check(
    answer: Annotated[
        str,
        typer.Argument(help="File to verify, a literal string, or - to read stdin."),
    ],
    sources: Annotated[
        list[str],
        typer.Option("--sources", "-s", help="Source file or directory. Repeatable."),
    ],
    judge: Annotated[
        str,
        typer.Option("--judge", "-j", help="lexical (offline), llm, or auto."),
    ] = "auto",
    report: Annotated[
        str, typer.Option("--report", "-r", help="Write an HTML report to this path.")
    ] = "",
    json_out: Annotated[
        str, typer.Option("--json", help="Write machine-readable results to this path.")
    ] = "",
    open_report: Annotated[
        bool, typer.Option("--open", help="Open the HTML report in a browser.")
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="Score at or above which a claim counts as supported.",
        ),
    ] = 0.75,
    fail_under: Annotated[
        float,
        typer.Option(
            "--fail-under", min=0.0, max=1.0, help="Exit non-zero if faithfulness falls below this."
        ),
    ] = 0.0,
    model: Annotated[
        str, typer.Option("--model", help="Model for the LLM judge, e.g. gpt-4o-mini.")
    ] = "",
    top_k: Annotated[
        int, typer.Option("--top-k", min=1, max=25, help="Candidate passages per claim.")
    ] = 5,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every claim, not just problems.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Print only the faithfulness score.")
    ] = False,
) -> None:
    """Verify a piece of text against a corpus of sources."""
    from footnote.verify import check as run_check

    text = _read_answer(answer)
    if not text.strip():
        raise typer.BadParameter("the text to verify is empty")

    result = run_check(
        text,
        sources,
        judge=judge,
        supported=threshold,
        top_k=top_k,
        model=model or None,
    )

    for warning in result.warnings:
        err.print(f"[yellow]warning:[/yellow] {warning}")

    if quiet:
        console.print(f"{result.faithfulness:.4f}")
    else:
        _render(result, verbose=verbose)

    if report:
        from footnote.report import write_report

        written = write_report(result, report)
        if not quiet:
            console.print(f"report [cyan]{written}[/cyan]")
        if open_report:
            webbrowser.open(written.resolve().as_uri())

    if json_out:
        Path(json_out).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        if not quiet:
            console.print(f"json   [cyan]{json_out}[/cyan]")

    if fail_under and result.faithfulness < fail_under:
        err.print(
            f"[red]faithfulness {result.faithfulness:.2f} is below the required "
            f"{fail_under:.2f}[/red]"
        )
        raise typer.Exit(code=1)


@app.command(name="eval")
def evaluate(
    judge: Annotated[str, typer.Option("--judge", "-j", help="lexical, llm, or auto.")] = "lexical",
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset", "-d", help="Path to a JSONL eval set. Defaults to the bundled one."
        ),
    ] = "",
    json_out: Annotated[
        str, typer.Option("--json", help="Write the full metrics to this path.")
    ] = "",
    model: Annotated[str, typer.Option("--model", help="Model for the LLM judge.")] = "",
) -> None:
    """Measure verification quality against a labelled dataset."""
    from footnote.evaluation import evaluate as run_eval
    from footnote.evaluation import load_dataset, render_metrics

    examples = load_dataset(dataset or None)
    console.print(f"evaluating [bold]{judge}[/bold] judge on {len(examples)} labelled claims...")
    metrics = run_eval(examples, judge=judge, model=model or None)
    render_metrics(metrics, console)

    if json_out:
        Path(json_out).write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
        console.print(f"json   [cyan]{json_out}[/cyan]")


@app.command()
def mcp(
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Default source directory for the tools."),
    ] = ".",
) -> None:
    """Run the Model Context Protocol server on stdio."""
    try:
        from footnote.mcp_server import serve
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        err.print("[red]the MCP server needs an extra:[/red] pip install 'footnote-verify[mcp]'")
        raise typer.Exit(code=1) from exc
    serve(default_sources=sources)


@app.command()
def demo(
    report: Annotated[
        str, typer.Option("--report", "-r", help="Where to write the demo report.")
    ] = "footnote-report.html",
    open_report: Annotated[bool, typer.Option("--open", help="Open it in a browser.")] = False,
) -> None:
    """Run the bundled example end to end and write a report.

    A zero-argument way to see what the tool does, and the command that generates
    the example report committed in the repository.
    """
    from footnote.evaluation import demo_case
    from footnote.report import write_report
    from footnote.verify import check_documents

    answer, documents = demo_case()
    result = check_documents(answer, documents, judge="lexical")
    _render(result, verbose=True)
    written = write_report(result, report, title="Footnote demo report")
    console.print(f"report [cyan]{written}[/cyan]")
    if open_report:
        webbrowser.open(written.resolve().as_uri())


if __name__ == "__main__":  # pragma: no cover
    app()
