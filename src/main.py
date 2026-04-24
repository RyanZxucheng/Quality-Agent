"""
主程序入口
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
    """主函数"""
    parser = argparse.ArgumentParser(
        description="医学 QA 数据质量评估 Agent"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="输入文件路径列表（支持 JSON/CSV/JSONL）"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/output",
        help="输出目录（默认: data/output）"
    )
    parser.add_argument(
        "--total-threshold",
        type=int,
        default=70,
        help="总分阈值（默认: 70）"
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=int,
        default=35,
        help="准确性阈值（默认: 35）"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="检查点保存间隔（默认: 100）"
    )
    parser.add_argument(
        "--max-retained",
        type=int,
        default=None,
        help="筛选出指定数量数据后停止（默认: 不限制）"
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
        help="每批次处理数量，大于 1 时启用并行处理（默认: 1）"
    )

    # LLM 模型配置
    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "openai", "vllm"],
        default="anthropic",
        help="LLM 提供商（默认: anthropic）"
    )
    parser.add_argument(
        "--llm-model",
        default="claude-3-5-sonnet-20241022",
        help="LLM 模型名称（默认: claude-3-5-sonnet-20241022）"
    )
    parser.add_argument(
        "--llm-base-url",
        default="",
        help="LLM API 地址（vLLM/自定义端点时使用）"
    )
    parser.add_argument(
        "--llm-api-key",
        default="",
        help="LLM API 密钥（默认从环境变量读取）"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="LLM 生成温度（默认: 0.1）"
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=1500,
        help="LLM 最大生成 token 数（默认: 1500）"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="启用详细/调试输出（默认: info-only）"
    )

    args = parser.parse_args()

    # 设置日志（确保输出目录存在）
    console = setup_console_logging(verbose=args.verbose, output_dir=args.output)

    # 验证输入文件
    for dataset_path in args.dataset:
        if not Path(dataset_path).exists():
            logger.error(f"输入文件不存在: {dataset_path}")
            sys.exit(1)

    # 配置
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

    # ── 启动横幅 ──────────────────────────────────────────────────────────────
    subtitle_lines = [
        f"输入文件: {', '.join(args.dataset)}",
        f"输出目录: {args.output}",
        f"LLM: {config.llm_provider.value}/{config.llm_model}",
        f"温度: {config.llm_temperature}, 最大token: {config.llm_max_tokens}",
        f"总分阈值: {args.total_threshold}",
        f"准确性阈值: {args.accuracy_threshold}",
    ]
    if args.max_retained is not None:
        subtitle_lines.append(f"最大保留数: {args.max_retained}")
    subtitle_lines.append(f"批次大小: {args.batch_size} ({'并行' if args.batch_size > 1 else '顺序'})")

    print_header(
        "医学 QA 数据质量评估 Agent",
        "\n".join(subtitle_lines),
    )

    # 处理
    try:
        processor = BatchProcessor()
        summary = processor.process_file(
            input_paths=args.dataset,
            output_dir=args.output,
            checkpoint_interval=args.checkpoint_interval,
            max_retained=args.max_retained
        )

        # ── 结束摘要面板 ──────────────────────────────────────────────────
        print_summary_panel(
            total=summary.total_processed,
            retained=summary.retained,
            discarded=summary.discarded,
            retention_rate=summary.retention_rate,
            average_score=summary.average_score,
            dimension_averages=summary.dimension_averages,
        )

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
