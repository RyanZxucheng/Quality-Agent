"""
Terminal output formatting module.
Provides rich, colorized terminal output while keeping full-format file logs.
"""
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.columns import Columns
from rich.text import Text

# ── Console singleton ──────────────────────────────────────────────────────────

_console: Optional[Console] = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(highlight=True)
    return _console


# ── Logger suppression ─────────────────────────────────────────────────────────

_NOISY_LOGGERS = [
    "urllib3",
    "httpx",
    "sentence_transformers",
    "transformers",
    "requests",
    "anthropic",
    "openai",
    "httpcore",
    "asyncio",
]


def suppress_noisy_loggers():
    """Raise third-party library log levels to WARNING to reduce terminal noise."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_console_logging(verbose: bool = False, output_dir: Optional[str] = None) -> Console:
    """
    Set up terminal and file logging.

    - Terminal: RichHandler, colorized, no timestamp/module path
    - File: standard FileHandler, full format for post-hoc audit

    Args:
        verbose: set root logger to DEBUG level when True
        output_dir: log file output directory (no file when None)

    Returns:
        Console instance
    """
    console = get_console()
    root_logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO

    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # ── Terminal Handler ───────────────────────────────────────────────────
    rh = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
    )
    rh.setLevel(level)
    root_logger.addHandler(rh)

    # ── File Handler (full format) ─────────────────────────────────────────
    if output_dir:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / "medical_qa_agent.log",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(fh)

    suppress_noisy_loggers()

    return console


# ── Structured output helpers ──────────────────────────────────────────────────

def print_header(
    title: str,
    model: str = "",
    input_files: str = "",
    output_dir: str = "",
    thresholds: str = "",
    batch_info: str = "",
    llm_info: str = "",
):
    """
    Print a compact startup banner with core configuration info.

    ┌─ Medical QA Quality Assessment ─────────────────────────────────┐
    │ Input: example_qa.json  │  Model: claude-3-5-sonnet              │
    │ Output: data/output     │  Thresholds: 70/35  │  Batch: 5 (par) │
    └──────────────────────────────────────────────────────────────────┘
    """
    console = get_console()

    info_lines = []
    left_parts = []
    right_parts = []

    def _append_label_value(line: Text, label: str, value: str) -> None:
        line.append(label, style="bold")
        line.append(" ")
        line.append(value)

    if input_files:
        left_parts.append(("Input:", input_files))
    if model:
        right_parts.append(("Model:", model))

    if left_parts or right_parts:
        line = Text()
        if left_parts:
            _append_label_value(line, *left_parts[0])
            if right_parts:
                line.append("  │  ")
        if right_parts:
            _append_label_value(line, *right_parts[0])
        info_lines.append(line)

    left_parts2 = []
    right_parts2 = []
    if output_dir:
        left_parts2.append(("Output:", output_dir))
    if thresholds:
        right_parts2.append(("Thresholds:", thresholds))
    if batch_info:
        right_parts2.append(("Batch:", batch_info))

    if left_parts2 or right_parts2:
        line2 = Text()
        if left_parts2:
            _append_label_value(line2, *left_parts2[0])
            if right_parts2:
                line2.append("  │  ")
        if right_parts2:
            for i, part in enumerate(right_parts2):
                if i > 0:
                    line2.append("  │  ")
                _append_label_value(line2, *part)
        info_lines.append(line2)

    if llm_info:
        line3 = Text()
        line3.append(llm_info, style="dim")
        info_lines.append(line3)

    content = Text.assemble(
        (f"  {title}", "bold cyan"),
        "\n",
    )
    for il in info_lines:
        content.append("\n")
        content.append(Text("  ") + il)

    console.print()
    console.print(Panel(
        content,
        border_style="cyan",
        box=box.HEAVY_EDGE,
        padding=(0, 1),
    ))


def print_summary_panel(
    total: int,
    retained: int,
    discarded: int,
    retention_rate: float,
    average_score: float,
    dimension_averages: Optional[dict] = None,
):
    """
    Print a compact evaluation summary panel.

    ┌─ Evaluation Complete ─────────────────────────────────────────────┐
    │ ✓ Retained: 120/150 (80.0%)  │  Avg Score: 75.2                  │
    │   accuracy: 38.2  completeness: 36.5  consistency: 35.1  safety   │
    └──────────────────────────────────────────────────────────────────┘
    """
    console = get_console()

    status_color = "green" if retention_rate >= 0.5 else "yellow"

    summary = Text("  ")
    summary.append("✓" if retention_rate >= 0.5 else "⚠", style=f"bold {status_color}")
    summary.append(" ")
    summary.append("Retained:", style="bold")
    summary.append(f" {retained}/{total} ({retention_rate:.1%})", style=status_color)
    summary.append("  │  ")
    summary.append("Avg Score:", style="bold")
    summary.append(f" {average_score:.1f}", style="white")

    lines = [summary]

    if dimension_averages:
        dim_text = Text("  ")
        for i, (name, avg) in enumerate(dimension_averages.items()):
            color = "green" if avg >= 30 else "yellow" if avg >= 20 else "red"
            if i > 0:
                dim_text.append("  │  ")
            dim_text.append(f"{name}: {avg:.1f}", style=color)
        lines.append(Text())
        lines.append(dim_text)

    # Rebuild properly
    content2 = Text()
    for i, line in enumerate(lines):
        if i > 0:
            content2.append("\n")
        content2.append(line)

    console.print()
    console.print(Panel(
        content2,
        title="[bold]Evaluation Complete[/bold]",
        border_style="green",
        box=box.HEAVY_EDGE,
        padding=(0, 1),
    ))
    console.print()
