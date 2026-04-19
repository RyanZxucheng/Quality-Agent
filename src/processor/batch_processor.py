"""
批量处理器
处理 JSON/CSV/JSONL 文件，执行批量评估
流程: 工具收集证据 -> LLM评分 -> 代码决策
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.models import QAPair, EvaluationResult, EvaluationSummary, Conclusion, EvidencePackage
from src.evidence import EvidenceCollector
from src.scoring.llm_engine import LLMScoringEngine
from src.decision import DecisionEngine
from src.config import get_config
from src.utils.file_utils import safe_read_json, safe_read_jsonl, safe_read_csv, ensure_dir, write_jsonl

logger = logging.getLogger(__name__)


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
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {len(input_paths)} file(s)")

        # 根据配置选择处理模式
        if self.config.is_parallel:
            logger.info(f"Parallel mode: batch_size={self.config.batch_size}, workers={self.config.max_workers}")
            self.evidence_collector.warm_up()
            return self._process_parallel(
                qa_pairs, output_dir, checkpoint_interval, max_retained
            )
        else:
            logger.info("Sequential mode: batch_size=1")
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
            logger.error(f"Failed to process {qa_pair.id}: {e}")
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

        for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Processing")):
            _, evaluation, output = self._process_single(qa_pair)
            results.append(evaluation)

            if output is not None:
                if evaluation.conclusion == Conclusion.RETAIN:
                    retained_data.append(output)
                else:
                    discarded_data.append(output)

            if max_retained is not None and len(retained_data) >= max_retained:
                logger.info(f"Reached max retained limit ({max_retained}), stopping early.")
                break

            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(output_dir, results, retained_data, discarded_data)

        return self._save_results(output_dir, results, retained_data, discarded_data)

    def _process_parallel(
        self,
        qa_pairs: List[QAPair],
        output_dir: str,
        checkpoint_interval: int,
        max_retained: Optional[int],
    ) -> EvaluationSummary:
        """并行批次处理模式"""
        results: List[EvaluationResult] = []
        retained_data: List[Dict[str, Any]] = []
        discarded_data: List[Dict[str, Any]] = []

        batch_size = self.config.batch_size
        max_workers = self.config.max_workers
        total = len(qa_pairs)
        processed = 0

        pbar = tqdm(total=total, desc="Processing")

        try:
            while processed < total:
                # 早停检查：允许最多超发 batch_size-1 条
                if max_retained is not None and len(retained_data) >= max_retained:
                    logger.info(f"Reached max retained limit ({max_retained}), stopping early.")
                    break

                # 切分当前批次
                batch = qa_pairs[processed : processed + batch_size]
                qa_to_index = {qa: i for i, qa in enumerate(batch)}
                batch_results: List[Tuple[QAPair, EvaluationResult, Optional[Dict[str, Any]]]] = []

                # 并行执行当前批次
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_qa = {
                        executor.submit(self._process_single, qa): qa for qa in batch
                    }
                    for future in as_completed(future_to_qa):
                        try:
                            batch_results.append(future.result())
                        except Exception as e:
                            qa = future_to_qa[future]
                            logger.error(f"Future failed for {qa.id}: {e}")
                            evaluation = self._create_error_result(qa, str(e))
                            batch_results.append((qa, evaluation, None))

                # 按原始顺序整理批次结果（保持输出稳定性）
                batch_results.sort(key=lambda x: qa_to_index[x[0]])

                # 在主线程更新结果（保证 checkpoint 同步安全）
                for qa_pair, evaluation, output in batch_results:
                    # 严格早停：逐条检查避免超发
                    if max_retained is not None and len(retained_data) >= max_retained:
                        logger.info(f"Reached max retained limit ({max_retained}), stopping early.")
                        break

                    results.append(evaluation)
                    if output is not None:
                        if evaluation.conclusion == Conclusion.RETAIN:
                            retained_data.append(output)
                        else:
                            discarded_data.append(output)
                    processed += 1
                    pbar.update(1)

                # 保存检查点（按已处理数量判断）
                if processed % checkpoint_interval == 0:
                    self._save_checkpoint(output_dir, results, retained_data, discarded_data)
        finally:
            pbar.close()

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
                    raise ValueError(f"Unsupported file format: {suffix}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Input file not found: {input_path}")

            all_items.extend(items)
            logger.info(f"Loaded {len(items)} items from {input_path}")

        qa_pairs = []
        for i, item in enumerate(all_items):
            try:
                qa_pairs.append(self._create_qa_pair(item, i))
            except ValueError as e:
                logger.warning(f"Skip invalid QA item at index {i}: {e}")

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

        return QAPair(
            id=str(qa_id),
            question=str(question),
            answer=str(answer),
            metadata=metadata
        )

    def _format_output(
        self,
        qa_pair: QAPair,
        evidence: Any,
        evaluation: EvaluationResult
    ) -> Dict[str, Any]:
        """格式化输出"""
        from src.models import EvidencePackage
        evidence_info = {}
        if isinstance(evidence, EvidencePackage):
            evidence_info = {
                "rounds_executed": evidence.rounds_executed,
                "evidence_insufficient": evidence.evidence_insufficient,
                "internal_chunks": (
                    len(evidence.internal_context.chunks)
                    if evidence.internal_context else 0
                ),
                "external_sources": len(evidence.external_evidence),
            }

        return {
            "id": qa_pair.id,
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "evidence": evidence_info,
            "metadata": qa_pair.metadata,
            "scores": {
                "total": evaluation.scores.total_score,
                "dimensions": {
                    d.name: d.score
                    for d in evaluation.scores.dimensions
                }
            },
            "conclusion": evaluation.conclusion.value,
            "conclusion_reason": evaluation.reason,
        }

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
        """保存检查点"""
        checkpoint_dir = Path(output_dir) / "checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)

        # 保存检查点
        with open(checkpoint_dir / "checkpoint.json", "w", encoding="utf-8") as f:
            json.dump({
                "processed_count": len(results),
                "retained_count": len(retained),
                "discarded_count": len(discarded)
            }, f, ensure_ascii=False, indent=2)

        logger.debug(f"Checkpoint saved: {len(results)} processed")

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
            write_jsonl(retained, str(cleaned_dir / "retained_qa.jsonl"))

        # 保存丢弃的数据（只在有数据时保存）
        if discarded:
            rejected_dir = output_path / "rejected"
            ensure_dir(str(rejected_dir))
            write_jsonl(discarded, str(rejected_dir / "discarded_qa.jsonl"))

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
            f"Results saved: {summary.retained} retained, "
            f"{summary.discarded} discarded "
            f"({summary.retention_rate:.1%} retention rate)"
        )

        return summary

