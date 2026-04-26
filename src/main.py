"""
Main entry point
"""
import argparse
import logging
import sys
from pathlib import Path

from src.processor import BatchProcessor
from src.config import AppConfig, set_config
from src.utils.logging_setup import setup_console_logging, print_header, print_summary_panel

logger = logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Medical QA Quality Assessment Agent"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Input file paths (JSON/CSV/JSONL)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/output",
        help="Output directory (default: data/output)"
    )
    parser.add_argument(
        "--total-threshold",
        type=int,
        default=70,
        help="Total score threshold (default: 70)"
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=int,
        default=35,
        help="Accuracy score threshold (default: 35)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Checkpoint save interval (default: 100)"
    )
    parser.add_argument(
        "--max-retained",
        type=int,
        default=None,
        help="Stop after retaining this many items (default: no limit)"
    )
    def _positive_int(value):
        iv = int(value)
        if iv < 1:
            raise argparse.ArgumentTypeError(f"batch-size must be >= 1, got {value}")
        return iv

    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Batch size, > 1 enables parallel processing (default: 1)"
    )

    # LLM model configuration
    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "openai", "vllm"],
        default="anthropic",
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--llm-model",
        default="claude-3-5-sonnet-20241022",
        help="LLM model name (default: claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--llm-base-url",
        default="",
        help="LLM API base URL (for vLLM / custom endpoints)"
    )
    parser.add_argument(
        "--llm-api-key",
        default="",
        help="LLM API key (default: read from environment)"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=1500,
        help="LLM max tokens (default: 1500)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug output (default: info-only)"
    )

    args = parser.parse_args()

    # Setup logging
    console = setup_console_logging(verbose=args.verbose, output_dir=args.output)

    # Validate input files
    for dataset_path in args.dataset:
        if not Path(dataset_path).exists():
            logger.error(f"Input file not found: {dataset_path}")
            sys.exit(1)

    # Configuration
    config = AppConfig(
        output_dir=args.output,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        max_retained=args.max_retained,
        batch_size=args.batch_size,
    )
    config.thresholds.total_min = args.total_threshold
    config.thresholds.accuracy_min = args.accuracy_threshold
    set_config(config)

    # ── Startup Banner ──────────────────────────────────────────────────────────
    batch_mode = "parallel" if args.batch_size > 1 else "sequential"
    batch_info = f"{args.batch_size} ({batch_mode})"
    thresholds_info = f"{args.total_threshold}/{args.accuracy_threshold}"
    llm_info = f"LLM: {config.llm_provider.value}/{config.llm_model}  |  Temperature: {config.llm_temperature}  |  Max Tokens: {config.llm_max_tokens}"
    if args.max_retained is not None:
        llm_info += f"  |  Max Retained: {args.max_retained}"

    print_header(
        title="Medical QA Quality Assessment",
        model=config.llm_model,
        input_files=", ".join(args.dataset),
        output_dir=args.output,
        thresholds=thresholds_info,
        batch_info=batch_info,
        llm_info=llm_info,
    )

    # Processing
    try:
        processor = BatchProcessor()
        summary = processor.process_file(
            input_paths=args.dataset,
            output_dir=args.output,
            checkpoint_interval=args.checkpoint_interval,
            max_retained=args.max_retained
        )

        # ── Summary Panel ──────────────────────────────────────────────────
        print_summary_panel(
            total=summary.total_processed,
            retained=summary.retained,
            discarded=summary.discarded,
            retention_rate=summary.retention_rate,
            average_score=summary.average_score,
            dimension_averages=summary.dimension_averages,
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
