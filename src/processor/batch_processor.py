"""
批量处理器
处理 JSON/CSV/JSONL 文件，执行批量评估
流程: 工具收集证据 -> LLM评分 -> 代码决策
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from src.models import QAPair, EvaluationResult, EvaluationSummary, Conclusion, EvidencePackage
from src.evidence import EvidenceCollector
from src.scoring.llm_engine import LLMScoringEngine
from src.decision import DecisionEngine
from src.config import get_config
from src.utils.file_utils import safe_read_json, safe_read_jsonl, safe_read_csv, ensure_dir
from src.utils.logging_setup import get_console

logger = logging.getLogger(__name__)


def _make_progress(console):
    """创建统一风格的 Progress 实例"""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


class BatchProcessor:
    """
    批量处理器
    基于工具证据 + LLM 评分的质量评估流程
    支持顺序处理和并行批次处理
    """

    def __init__(
        self,
        evidence_collector: Optional[EvidenceCollector] = None,
        llm_scoring_engine: Optional[LLMScoringEngine] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.config = get_config()
        self.evidence_collector = evidence_collector or EvidenceCollector()
        self.llm_scoring_engine = llm_scoring_engine or LLMScoringEngine()
        self.decision_engine = decision_engine or DecisionEngine()
        # 检查点写入锁，防止后台线程重叠写入
        self._checkpoint_lock = threading.Lock()

    def process_file(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        max_retained: Optional[int] = None
    ) -> EvaluationSummary:
        """
        处理输入文件

        Args:
            input_paths: 输入文件路径列表（JSON/CSV/JSONL）
            output_dir: 输出目录
            checkpoint_interval: 检查点保存间隔
            max_retained: 最大保留数量，达到后提前停止

        Returns:
            EvaluationSummary: 评估摘要
        """
        output_dir = output_dir or self.config.output_dir
        ensure_dir(output_dir)

        # 加载 QA 对（支持多文件）
        qa_pairs = self._load_qa_pairs(input_paths)
        logger.info(f"已加载 {len(qa_pairs)} 条 QA 数据，来自 {len(input_paths)} 个文件")

        # 根据配置选择处理模式
        if self.config.is_parallel:
            logger.info(f"并行模式: batch_size={self.config.batch_size}")
            self.evidence_collector.warm_up()
            return self._process_parallel(
                qa_pairs, output_dir, checkpoint_interval, max_retained
            )
        else:
            logger.info("顺序模式: batch_size=1")
            return self._process_sequential(
                qa_pairs, output_dir, checkpoint_interval, max_retained
            )

    def _process_single(self, qa_pair: QAPair) -> Tuple[QAPair, EvaluationResult, Optional[Dict[str, Any]]]:
        """
        处理单个 QA 对

        Returns:
            (qa_pair, evaluation, output_dict) — output_dict 为 None 表示错误
        """
        try:
            evidence = self.evidence_collector.collect(qa_pair)
            score_result = self.llm_scoring_engine.score(qa_pair, evidence)
            evidence_pkg = evidence if isinstance(evidence, EvidencePackage) else None
            evaluation = self.decision_engine.decide(score_result, evidence_pkg)
            evaluation.evidence = evidence

            output = self._format_output(qa_pair, evidence, evaluation)
            return qa_pair, evaluation, output
        except Exception as e:
            logger.error(f"处理失败 {qa_pair.id}: {e}")
            evaluation = self._create_error_result(qa_pair, str(e))
            return qa_pair, evaluation, None

    def _process_sequential(
        self,
        qa_pairs: List[QAPair],
        output_dir: str,
        checkpoint_interval: int,
        max_retained: Optional[int],
    ) -> EvaluationSummary:
        """顺序处理模式"""
        results: List[EvaluationResult] = []
        retained_data: List[Dict[str, Any]] = []
        discarded_data: List[Dict[str, Any]] = []
        console = get_console()

        with _make_progress(console) as progress:
            task = progress.add_task("处理中", total=len(qa_pairs))

            for i, qa_pair in enumerate(qa_pairs):
                _, evaluation, output = self._process_single(qa_pair)
                results.append(evaluation)

                if output is not None:
                    if evaluation.conclusion == Conclusion.RETAIN:
                        retained_data.append(output)
                    else:
                        discarded_data.append(output)

                if max_retained is not None and len(retained_data) >= max_retained:
                    logger.info(f"达到最大保留数限制 ({max_retained})，提前停止")
                    break

                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(output_dir, results, retained_data, discarded_data)

                progress.update(task, advance=1)

        return self._save_results(output_dir, results, retained_data, discarded_data)

    def _process_parallel(
        self,
        qa_pairs: List[QAPair],
        output_dir: str,
        checkpoint_interval: int,
        max_retained: Optional[int],
    ) -> EvaluationSummary:
        """并行处理模式——流式批次提交

        batch_size = 每批同时在处理的数据条数（并发度）。
        始终保持最多 batch_size 个任务在运行：先提交第一批，
        后续每完成一个就补充一个，无批次间等待，也避免一次性
        提交全部任务造成的内存堆积。
        """
        results: List[EvaluationResult] = []
        retained_data: List[Dict[str, Any]] = []
        discarded_data: List[Dict[str, Any]] = []

        batch_size = self.config.batch_size
        total = len(qa_pairs)
        stopped = False

        console = get_console()
        progress = _make_progress(console)
        progress.start()
        task = progress.add_task("处理中", total=total)

        qa_iter = iter(enumerate(qa_pairs))
        pending_futures: Dict[Any, int] = {}

        def _submit_next() -> bool:
            """提交下一个任务，返回是否成功提交"""
            nonlocal stopped
            if stopped:
                return False
            try:
                idx, qa = next(qa_iter)
                future = executor.submit(self._process_single, qa)
                pending_futures[future] = idx
                return True
            except StopIteration:
                return False

        try:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # 先提交第一批任务，填满并发槽位
                for _ in range(batch_size):
                    if not _submit_next():
                        break

                while pending_futures:
                    future = next(as_completed(pending_futures))
                    idx = pending_futures.pop(future)

                    # 取结果
                    try:
                        qa_pair, evaluation, output = future.result()
                    except Exception as e:
                        qa_pair = qa_pairs[idx]
                        logger.error(f"Future 失败 {qa_pair.id}: {e}")
                        evaluation = self._create_error_result(qa_pair, str(e))
                        output = None

                    results.append(evaluation)
                    if output is not None:
                        if evaluation.conclusion == Conclusion.RETAIN:
                            retained_data.append(output)
                        else:
                            discarded_data.append(output)

                    progress.update(task, advance=1)

                    # 检查点
                    if len(results) % checkpoint_interval == 0:
                        self._save_checkpoint(output_dir, results, retained_data, discarded_data)

                    # 早停判断
                    if max_retained is not None and len(retained_data) >= max_retained:
                        logger.info(f"达到最大保留数限制 ({max_retained})，提前停止")
                        stopped = True
                        # 取消尚未开始的任务
                        for f in list(pending_futures.keys()):
                            if not f.done():
                                f.cancel()
                        pending_futures.clear()
                        break

                    # 补充新任务，维持并发度
                    _submit_next()
        finally:
            progress.stop()

        return self._save_results(output_dir, results, retained_data, discarded_data)

    def _load_qa_pairs(self, input_paths: List[str]) -> List[QAPair]:
        """加载 QA 对（支持多文件）"""
        all_items = []
        for input_path in input_paths:
            path = Path(input_path)
            suffix = path.suffix.lower()

            try:
                if suffix == ".json":
                    items = safe_read_json(input_path)
                elif suffix == ".csv":
                    items = safe_read_csv(input_path)
                elif suffix == ".jsonl":
                    items = safe_read_jsonl(input_path)
                else:
                    raise ValueError(f"不支持的文件格式: {suffix}")
            except FileNotFoundError:
                raise FileNotFoundError(f"输入文件不存在: {input_path}")

            all_items.extend(items)
            logger.info(f"已加载 {len(items)} 条数据从 {input_path}")

        qa_pairs = []
        for i, item in enumerate(all_items):
            try:
                qa_pairs.append(self._create_qa_pair(item, i))
            except ValueError as e:
                logger.warning(f"跳过无效 QA 条目（索引 {i}）: {e}")

        return qa_pairs


    def _create_qa_pair(self, item: Dict, index: int) -> QAPair:
        """创建 QAPair 对象"""
        # 自动识别 messages 格式（如 OpenAI 对话格式）
        if "messages" in item and isinstance(item["messages"], list):
            question = ""
            answer = ""
            for msg in item["messages"]:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user" and not question:
                        question = content
                    elif role == "assistant" and not answer:
                        answer = content
        else:
            # 支持多种字段名
            question = item.get("question") or item.get("Question") or item.get("q", "")
            answer = item.get("answer") or item.get("Answer") or item.get("a", "")

        qa_id = item.get("id") or item.get("ID") or f"qa_{index}"

        # 保留原始元数据（排除所有已识别的字段变体）
        excluded_keys = {"id", "ID", "question", "Question", "q", "answer", "Answer", "a", "messages"}
        metadata = {k: v for k, v in item.items() if k not in excluded_keys}

        # 保留原始输入数据的完整副本，用于输出时保持格式一致
        raw_data = dict(item)

        return QAPair(
            id=str(qa_id),
            question=str(question),
            answer=str(answer),
            metadata=metadata,
            raw_data=raw_data
        )

    def _format_output(
        self,
        qa_pair: QAPair,
        evidence: Any,
        evaluation: EvaluationResult
    ) -> Dict[str, Any]:
        """格式化输出——保持与输入格式一致，不新增字段"""
        return qa_pair.raw_data

    def _create_error_result(self, qa_pair: QAPair, error: str) -> EvaluationResult:
        """创建错误结果"""
        return EvaluationResult.create_error_result(qa_pair, error, Conclusion.DISCARD)

    def _save_checkpoint(
        self,
        output_dir: str,
        results: List[EvaluationResult],
        retained: List[Dict],
        discarded: List[Dict]
    ):
        """保存检查点（异步，后台线程写入，不阻塞处理）"""
        threading.Thread(
            target=self._save_checkpoint_sync,
            args=(output_dir, len(results), len(retained), len(discarded)),
            daemon=True,
        ).start()

    def _save_checkpoint_sync(
        self,
        output_dir: str,
        processed_count: int,
        retained_count: int,
        discarded_count: int,
    ):
        """检查点同步写入（在后台线程中执行）"""
        if not self._checkpoint_lock.acquire(blocking=False):
            logger.debug("检查点写入已在进行，跳过")
            return
        try:
            checkpoint_dir = Path(output_dir) / "checkpoint"
            checkpoint_dir.mkdir(exist_ok=True)
            with open(checkpoint_dir / "checkpoint.json", "w", encoding="utf-8") as f:
                json.dump({
                    "processed_count": processed_count,
                    "retained_count": retained_count,
                    "discarded_count": discarded_count,
                }, f, ensure_ascii=False, indent=2)
            logger.debug(f"检查点已保存: {processed_count} 条已处理")
        except Exception as e:
            logger.warning(f"检查点保存失败: {e}")
        finally:
            self._checkpoint_lock.release()

    def _save_results(
        self,
        output_dir: str,
        results: List[EvaluationResult],
        retained: List[Dict],
        discarded: List[Dict]
    ) -> EvaluationSummary:
        """保存最终结果"""
        output_path = Path(output_dir)

        # 保存保留的数据（只在有数据时保存）
        if retained:
            cleaned_dir = output_path / "cleaned_data"
            ensure_dir(str(cleaned_dir))
            with open(cleaned_dir / "retained_qa.json", "w", encoding="utf-8") as f:
                json.dump(retained, f, ensure_ascii=False, indent=2)

        # 保存丢弃的数据（只在有数据时保存）
        if discarded:
            rejected_dir = output_path / "rejected"
            ensure_dir(str(rejected_dir))
            with open(rejected_dir / "discarded_qa.json", "w", encoding="utf-8") as f:
                json.dump(discarded, f, ensure_ascii=False, indent=2)

        # 生成并保存报告
        from src.report import ReportGenerator
        report_gen = ReportGenerator()
        report = report_gen.generate(results)
        report_gen.save(report, str(output_path / "reports"))

        # 保存摘要
        summary = report.summary
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_processed": summary.total_processed,
                "retained": summary.retained,
                "discarded": summary.discarded,
                "retention_rate": summary.retention_rate,
                "average_score": summary.average_score,
                "dimension_averages": summary.dimension_averages
            }, f, ensure_ascii=False, indent=2)

        logger.info(
            f"结果已保存: 保留 {summary.retained} 条, "
            f"丢弃 {summary.discarded} 条 "
            f"(保留率 {summary.retention_rate:.1%})"
        )

        return summary
