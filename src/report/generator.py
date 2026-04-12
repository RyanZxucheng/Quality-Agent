"""
报告生成器
生成评估报告并保存
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.models import (
    EvaluationResult,
    EvaluationReport,
    EvaluationSummary,
    Conclusion
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    报告生成器
    生成详细的评估报告
    """

    def generate(self, results: List[EvaluationResult]) -> EvaluationReport:
        """
        生成评估报告

        Args:
            results: 评估结果列表

        Returns:
            EvaluationReport: 完整报告
        """
        # 生成摘要
        summary = self._generate_summary(results)

        # 构建完整报告
        report = EvaluationReport(
            summary=summary,
            results=results,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        )

        return report

    def _generate_summary(self, results: List[EvaluationResult]) -> EvaluationSummary:
        """生成摘要"""
        total = len(results)
        retained = sum(1 for r in results if r.conclusion == Conclusion.RETAIN)
        discarded = total - retained

        retention_rate = retained / total if total > 0 else 0

        # 计算平均分
        scores = [r.scores.total_score for r in results]
        average_score = sum(scores) / len(scores) if scores else 0

        # 计算各维度平均分
        dimension_sums: Dict[str, float] = {}
        dimension_counts: Dict[str, int] = {}

        for r in results:
            for dim in r.scores.dimensions:
                if dim.name not in dimension_sums:
                    dimension_sums[dim.name] = 0
                    dimension_counts[dim.name] = 0
                dimension_sums[dim.name] += dim.score
                dimension_counts[dim.name] += 1

        dimension_averages = {
            name: dimension_sums[name] / dimension_counts[name]
            for name in dimension_sums
        }

        return EvaluationSummary(
            total_processed=total,
            retained=retained,
            discarded=discarded,
            retention_rate=retention_rate,
            average_score=average_score,
            dimension_averages=dimension_averages
        )

    def save(self, report: EvaluationReport, output_dir: str):
        """
        保存报告

        Args:
            report: 评估报告
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存完整报告（JSON）
        report_data = self._serialize_report(report)
        with open(output_path / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # 保存人类可读的摘要
        summary_text = self._generate_summary_text(report)
        with open(output_path / "summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)

        logger.info(f"Report saved to {output_dir}")

    def _serialize_report(self, report: EvaluationReport) -> Dict[str, Any]:
        """序列化报告为字典"""
        return {
            "evaluation_summary": {
                "total_processed": report.summary.total_processed,
                "retained": report.summary.retained,
                "discarded": report.summary.discarded,
                "retention_rate": f"{report.summary.retention_rate:.1%}",
                "average_score": round(report.summary.average_score, 2),
                "dimension_averages": {
                    k: round(v, 2)
                    for k, v in report.summary.dimension_averages.items()
                }
            },
            "detailed_results": [
                self._serialize_result(r)
                for r in report.results
            ],
            "metadata": report.metadata
        }

    def _serialize_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """序列化单个结果"""
        evidence = result.evidence
        evidence_data: Dict[str, Any] = {}

        if evidence is not None:
            from src.models import EvidencePackage
            if isinstance(evidence, EvidencePackage):
                evidence_data = self._serialize_evidence_package(evidence)

        return {
            "id": result.qa_pair.id,
            "question": result.qa_pair.question,
            "answer": result.qa_pair.answer,
            "evidence": evidence_data,
            "scores": {
                "total": result.scores.total_score,
                "dimensions": {
                    d.name: d.score
                    for d in result.scores.dimensions
                }
            },
            "issues": result.scores.issues,
            "conclusion": result.conclusion.value,
            "reason": result.reason,
        }

    def _serialize_evidence_package(self, pkg: "EvidencePackage") -> Dict[str, Any]:
        """序列化 EvidencePackage 为可读字典"""
        self_check_data = [
            {
                "confidence": round(r.confidence, 3),
                "next_action": r.next_action.value,
                "blocking_issues": r.blocking_issues,
                "missing_slots": r.missing_slots,
                "reasoning": r.reasoning,
            }
            for r in pkg.self_check_rounds
        ]

        internal_data = None
        if pkg.internal_context and pkg.internal_context.chunks:
            ic = pkg.internal_context
            internal_data = {
                "chunks_retrieved": len(ic.chunks),
                "top_chunks": [
                    {
                        "doc_id": c.doc_id,
                        "chunk_id": c.chunk_id,
                        "relevance_score": round(c.relevance_score, 3),
                        "content_preview": c.content[:150],
                    }
                    for c in ic.chunks[:3]
                ],
            }

        external_data = [
            {
                "tool": e.tool_name,
                "source": e.source,
                "url": e.url,
                "confidence": e.confidence,
                "snippet": e.snippet[:200],
                "query_used": e.query_used,
            }
            for e in pkg.external_evidence
        ]

        return {
            "rounds_executed": pkg.rounds_executed,
            "evidence_insufficient": pkg.evidence_insufficient,
            "self_check_rounds": self_check_data,
            "internal": internal_data,
            "external": external_data,
        }

    def _generate_summary_text(self, report: EvaluationReport) -> str:
        """生成文本摘要（含多轮证据统计）"""
        internal_used = 0
        external_used = 0
        insufficient_count = 0

        from src.models import EvidencePackage
        for r in report.results:
            if isinstance(r.evidence, EvidencePackage):
                pkg: EvidencePackage = r.evidence
                if pkg.internal_context and pkg.internal_context.chunks:
                    internal_used += 1
                if pkg.external_evidence:
                    external_used += 1
                if pkg.evidence_insufficient:
                    insufficient_count += 1

        lines = [
            "=" * 60,
            "医学 QA 数据质量评估报告",
            "=" * 60,
            "",
            f"生成时间: {report.metadata.get('generated_at', 'N/A')}",
            f"版本: {report.metadata.get('version', 'N/A')}",
            "",
            "-" * 60,
            "评估摘要",
            "-" * 60,
            f"总处理数: {report.summary.total_processed}",
            f"保留数: {report.summary.retained}",
            f"丢弃数: {report.summary.discarded}",
            f"保留率: {report.summary.retention_rate:.1%}",
            f"平均分: {report.summary.average_score:.1f}",
            "",
            "-" * 60,
            "多轮证据收集统计",
            "-" * 60,
            f"触发内部检索: {internal_used} 条",
            f"触发外部检索: {external_used} 条",
            f"证据不足（待人工复核）: {insufficient_count} 条",
            "",
            "-" * 60,
            "各维度平均分数",
            "-" * 60,
        ]

        for name, avg in report.summary.dimension_averages.items():
            lines.append(f"  {name}: {avg:.1f}")

        lines.extend([
            "",
            "=" * 60,
            "",
        ])

        return "\n".join(lines)

