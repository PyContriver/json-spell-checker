#!/usr/bin/env python3
"""
JSON Spell & Grammar Agent
- Pass 1 (always):    pyspellchecker  — fast, pure-Python word-level spell check
- Pass 2 (--model):   Ollama REST API — open-source LLM grammar + spelling review
- Parallel:           multiple files / entries processed concurrently via ThreadPoolExecutor
"""

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from spellchecker import SpellChecker
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from src.utils.spell import spell_check
from src.utils.llm import ollama_list_models, ollama_check, _filter_llm_result, check_ollama, OLLAMA_BASE_URL
from src.utils.io import extract_strings, load_ignore_words

console = Console()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_spell_issues(issues: list[dict]) -> None:
    if not issues:
        return
    t = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 2))
    t.add_column("Misspelled", style="red")
    t.add_column("Suggestions", style="green")
    for i in issues:
        t.add_row(i["word"], ", ".join(i["suggestions"]) or "—")
    console.print(t)


def _render_llm_result(result: dict) -> None:
    if "error" in result:
        console.print(f"      [bold red]LLM error:[/bold red] {result['error']}")
        return
    issues = result.get("issues", [])
    if not issues:
        return
    t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    t.add_column("Type",       style="cyan",  no_wrap=True)
    t.add_column("Original",   style="red")
    t.add_column("Suggestion", style="green")
    t.add_column("Explanation")
    for i in issues:
        t.add_row(
            i.get("type", ""),
            i.get("original", ""),
            i.get("suggestion", ""),
            i.get("explanation", ""),
        )
    console.print(t)
    corrected = result.get("corrected_text", "")
    if corrected:
        console.print(f"      [dim]Fixed:[/dim] [italic]{corrected}[/italic]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI agent: checks spelling & grammar of every string in one or more JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file — spell-check only:
  python3 agent.py sample.json

  # Multiple files at once:
  python3 agent.py sample.json networking_sample.json --model mistral:7b

  # Entire folder (all *.json files), parallel LLM with 8 workers:
  python3 agent.py --dir ./configs --model mistral:7b --workers 8

  # LLM only (skip spell checker):
  python3 agent.py sample.json --model llama3 --no-spellcheck

  # See what models are available in your Ollama server:
  python3 agent.py --list-models

  # Save a machine-readable report:
  python3 agent.py sample.json --model llama3 --output report.json

  # Ignore specific words (e.g. networking acronyms not in the dictionary):
  python3 agent.py sample.json --ignore BGP OSPF MPLS datacenter
  python3 agent.py sample.json --model mistral --ignore-file ignore.txt

Start Ollama (if not running):
  ollama serve
  ollama pull mistral   # or llama3.2, gemma3, phi3, qwen2.5 …
        """,
    )
    parser.add_argument("files", nargs="*", help="One or more JSON files to check")
    parser.add_argument("--dir", default=None, metavar="DIR",
                        help="Directory — check all *.json files inside it")
    parser.add_argument("--model", "-m", default="mistral:7b", metavar="MODEL",
                        help="Ollama model for AI review (default: mistral:7b)")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="Parallel LLM workers (default: 4)")
    parser.add_argument("--no-spellcheck", action="store_true",
                        help="Skip pyspellchecker pass")
    parser.add_argument("--lang", default="en", metavar="LANG",
                        help="Spell-check language (default: en)")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL, metavar="URL",
                        help=f"Ollama base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--output", "-o", default=None, metavar="PATH",
                        help="Save report to this path (default: reports/<timestamp>_report.json)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip saving the JSON report to disk")
    parser.add_argument("--list-models", action="store_true",
                        help="List models available in the running Ollama server and exit")
    parser.add_argument("--ignore", nargs="+", metavar="WORD", default=[],
                        help="Words to ignore in both spell and LLM checks (space-separated)")
    parser.add_argument("--ignore-file", default=None, metavar="FILE",
                        help="Path to a plain-text file with one ignore-word per line (# comments ok)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- List models shortcut ---
    if args.list_models:
        models = ollama_list_models(args.ollama_url)
        if models:
            console.print("[bold]Available Ollama models:[/bold]")
            for m in models:
                console.print(f"  • {m}")
        else:
            console.print(
                "[red]No models found.[/red] Is Ollama running? "
                "Start it with [bold]ollama serve[/bold] and pull models with "
                "[bold]ollama pull mistral[/bold]."
            )
        return

    # --- Build file list ---
    json_files: list[Path] = []
    for f in (args.files or []):
        p = Path(f)
        if not p.exists():
            console.print(f"[red]Error:[/red] File not found: {p}")
            sys.exit(1)
        json_files.append(p)

    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            console.print(f"[red]Error:[/red] Not a directory: {d}")
            sys.exit(1)
        json_files.extend(sorted(d.glob("*.json")))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in json_files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    json_files = unique

    if not json_files:
        console.print("[red]Error:[/red] No JSON files found. Pass file paths or use --dir.")
        sys.exit(1)

    # --- Load all JSON ---
    file_data: dict[Path, Any] = {}
    for p in json_files:
        try:
            with p.open("r", encoding="utf-8") as f:
                file_data[p] = json.load(f)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Error:[/red] Invalid JSON in {p} — {exc}")
            sys.exit(1)

    # --- Build per-file entries ---
    file_entries: dict[Path, list[tuple[str, str]]] = {
        p: extract_strings(d) for p, d in file_data.items()
    }
    all_tasks = [
        (p, field, text)
        for p in json_files
        for field, text in file_entries[p]
    ]
    total_fields = len(all_tasks)

    if total_fields == 0:
        console.print("[yellow]No string values found in any of the JSON files.[/yellow]")
        sys.exit(0)

    # --- Build ignore set ---
    try:
        ignore_words = load_ignore_words(args.ignore, args.ignore_file)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # --- Ollama pre-flight check ---
    if args.model:
        health = check_ollama(args.model, args.ollama_url)
        if not health["reachable"]:
            console.print(
                f"[red]✘ Ollama is not reachable at {args.ollama_url}.[/red]\n"
                f"  → {health['hint']}"
            )
            sys.exit(1)
        if not health["model_ready"]:
            console.print(
                f"[yellow]⚠  Model [bold]{args.model}[/bold] is not pulled.[/yellow]\n"
                f"  → {health['hint']}\n"
                f"  Available: {', '.join(health['models']) or 'none'}"
            )
            sys.exit(1)
        console.print(f"[green]✔[/green] Ollama reachable · model [bold]{args.model}[/bold] ready")

    ignore_display = ", ".join(sorted(ignore_words)) if ignore_words else "none"
    console.print(Panel(
        f"[bold]JSON Spell & Grammar Agent[/bold]\n"
        f"Files  : [cyan]{len(json_files)}[/cyan]  ({', '.join(p.name for p in json_files)})\n"
        f"Fields : [cyan]{total_fields}[/cyan] string values\n"
        f"Spell  : [cyan]{'off' if args.no_spellcheck else args.lang}[/cyan]\n"
        f"LLM    : [cyan]{args.model}[/cyan]  ×[cyan]{args.workers}[/cyan] workers\n"
        f"Ignore : [cyan]{ignore_display}[/cyan]",
        title="[bold blue]Config[/bold blue]",
        border_style="blue",
    ))

    # spell_results[file][field] / llm_results[file][field]
    spell_results: dict[Path, dict[str, list[dict]]] = {p: {} for p in json_files}
    llm_results:   dict[Path, dict[str, dict]]       = {p: {} for p in json_files}

    # --- Pass 1: pyspellchecker (fast — sequential is fine) ---
    if not args.no_spellcheck:
        checker = SpellChecker(language=args.lang)
        if ignore_words:
            checker.word_frequency.load_words(ignore_words)
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            console=console, transient=True,
        ) as prog:
            task = prog.add_task("Spell checking…", total=total_fields)
            for file_path, field, text in all_tasks:
                spell_results[file_path][field] = spell_check(checker, text, ignore_words)
                prog.advance(task)

    # --- Pass 2: Ollama — all entries across all files in parallel ---
    if args.model:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            console=console, transient=True,
        ) as prog:
            task = prog.add_task(
                f"LLM [{args.model}] ×{args.workers} workers…", total=total_fields
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        ollama_check, args.model, text, args.ollama_url, ignore_words
                    ): (file_path, field)
                    for file_path, field, text in all_tasks
                }
                for future in as_completed(futures):
                    file_path, field = futures[future]
                    llm_results[file_path][field] = _filter_llm_result(
                        future.result(), ignore_words
                    )
                    prog.advance(task)

    # --- Render report ---
    grand_flagged = 0
    all_report_rows: list[dict] = []

    for file_path in json_files:
        entries   = file_entries[file_path]
        file_flag = 0
        file_rows = []

        console.rule(f"[bold cyan]{file_path.name}[/bold cyan]")

        for field, text in entries:
            s_issues  = spell_results[file_path].get(field, [])
            l_result  = llm_results[file_path].get(field, {})
            llm_error = "error" in l_result
            llm_bad   = l_result.get("has_issues", False) or llm_error
            has_issues = bool(s_issues) or llm_bad

            if has_issues:
                file_flag += 1

            icon = "[red]✘[/red]" if has_issues else "[green]✔[/green]"
            console.print(f"\n  {icon}  [bold]{field}[/bold]")

            if has_issues:
                preview = text[:110] + ("…" if len(text) > 110 else "")
                console.print(f"      [dim]{preview}[/dim]")

            if s_issues:
                console.print("      [bold yellow]Spell:[/bold yellow]")
                _render_spell_issues(s_issues)

            if l_result and llm_bad:
                console.print("      [bold magenta]LLM:[/bold magenta]")
                _render_llm_result(l_result)

            file_rows.append({
                "file":         str(file_path),
                "path":         field,
                "value":        text,
                "spell_issues": s_issues,
                "llm_result":   l_result,
                "has_issues":   has_issues,
            })

        color = "red" if file_flag else "green"
        console.print(
            f"\n  [{color}]{file_flag} field(s) with issues[/{color}]"
            f" out of [cyan]{len(entries)}[/cyan] in {file_path.name}\n"
        )
        grand_flagged  += file_flag
        all_report_rows.extend(file_rows)

    # --- Grand summary ---
    console.rule("[bold]Grand Summary[/bold]")
    color = "red" if grand_flagged else "green"
    console.print(
        f"\n[bold]Total:[/bold] [{color}]{grand_flagged} field(s) with issues[/{color}]"
        f" across [cyan]{total_fields}[/cyan] fields in [cyan]{len(json_files)}[/cyan] file(s).\n"
    )

    # --- Save report ---
    # --output overrides the default path; --no-report disables saving entirely.
    if not args.no_report:
        if args.output:
            out = Path(args.output)
        else:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out   = reports_dir / f"{stamp}_report.json"

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(all_report_rows, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✔[/green] Report saved → [bold]{out}[/bold]\n")

    sys.exit(1 if grand_flagged else 0)


if __name__ == "__main__":
    main()
