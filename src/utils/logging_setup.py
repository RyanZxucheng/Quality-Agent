"""
终端输出美化模块
基于 rich 库提供彩色、结构化的终端输出，
同时保持文件日志的完整格式不变。
"""
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Console 单例 ──────────────────────────────────────────────────────────────

_console: Optional[Console] = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(highlight=True)
    return _console


# ── 日志抑制 ──────────────────────────────────────────────────────────────────

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
    """将第三方库的日志级别提升为 WARNING，减少终端噪音"""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ── 日志设置 ──────────────────────────────────────────────────────────────────

def setup_console_logging(verbose: bool = False, output_dir: Optional[str] = None) -> Console:
    """
    设置终端和文件日志

    - 终端：RichHandler，彩色输出，无时间戳/模块路径
    - 文件：标准 FileHandler，保留完整格式用于事后审计

    Args:
        verbose: True 时 root logger 设为 DEBUG 级别
        output_dir: 日志文件输出目录（None 时不写文件）

    Returns:
        Console 实例
    """
    console = get_console()
    root_logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO

    # 清除已有的 handlers（防止重复添加）
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # ── 终端 Handler ──────────────────────────────────────────────────────
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

    # ── 文件 Handler（保留完整格式） ──────────────────────────────────────
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

    # 抑制第三方库日志
    suppress_noisy_loggers()

    return console


# ── 结构化输出辅助函数 ───────────────────────────────────────────────────────

def print_header(title: str, subtitle: str = ""):
    """
    打印启动横幅

    ─────────────────────────────────────────────
    ╭───────────────────────────────────────────╮
    │  title                                    │
    │  subtitle                                 │
    ╰───────────────────────────────────────────╯
    """
    console = get_console()
    console.rule(style="dim")
    console.print(Panel.fit(
        f"[bold cyan]{title}[/bold cyan]"
        + (f"\n[dim]{subtitle}[/dim]" if subtitle else ""),
        border_style="cyan",
    ))


def print_summary_panel(total: int, retained: int, discarded: int,
                        retention_rate: float, average_score: float,
                        dimension_averages: Optional[dict] = None):
    """
    打印最终统计摘要面板

    ╭───────────────────────────────────────────╮
    │  Metric                    Value          │
    │  Total processed           150            │
    │  ...                                      │
    ╰───────────────────────────────────────────╯
    """
    console = get_console()
    console.rule(style="dim")

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
    )
    table.add_column("指标", style="white")
    table.add_column("数值", style="green", justify="right")

    table.add_row("总处理数", str(total))
    table.add_row("保留", str(retained))
    table.add_row("丢弃", str(discarded))
    table.add_row("保留率", f"{retention_rate:.1%}")
    table.add_row("平均分", f"{average_score:.1f}")

    if dimension_averages:
        console.rule(style="dim")
        dim_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
        )
        dim_table.add_column("维度", style="white")
        dim_table.add_column("平均分", style="green", justify="right")

        for name, avg in dimension_averages.items():
            dim_table.add_row(name, f"{avg:.1f}")

        console.print(Panel.fit(
            dim_table,
            title="[bold]维度平均分[/bold]",
            border_style="cyan",
        ))

    console.print(Panel.fit(
        table,
        title="[bold]评估摘要[/bold]",
        border_style="cyan",
    ))
    console.rule(style="dim")
