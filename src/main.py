"""
主程序入口
"""
import argparse
import logging
import sys
from pathlib import Path

from src.processor import BatchProcessor
from src.config import AppConfig, set_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("medical_qa_agent.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="医学 QA 数据质量评估 Agent"
    )
    parser.add_argument(
        "input",
        help="输入文件路径（JSON/CSV/JSONL）"
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

    args = parser.parse_args()

    # 验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    # 配置
    config = AppConfig(
        output_dir=args.output,
    )
    config.thresholds.total_min = args.total_threshold
    config.thresholds.accuracy_min = args.accuracy_threshold
    set_config(config)

    logger.info("=" * 60)
    logger.info("医学 QA 数据质量评估 Agent 启动")
    logger.info("=" * 60)
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"总分阈值: {args.total_threshold}")
    logger.info(f"准确性阈值: {args.accuracy_threshold}")

    # 处理
    try:
        processor = BatchProcessor()
        summary = processor.process_file(
            input_path=args.input,
            output_dir=args.output,
            checkpoint_interval=args.checkpoint_interval
        )

        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info("=" * 60)
        logger.info(f"总处理: {summary.total_processed}")
        logger.info(f"保留: {summary.retained}")
        logger.info(f"丢弃: {summary.discarded}")
        logger.info(f"保留率: {summary.retention_rate:.1%}")
        logger.info(f"平均分: {summary.average_score:.1f}")

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
