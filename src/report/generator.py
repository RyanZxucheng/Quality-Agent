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
        self_check_data: List[Dict[str, Any]] = []

        if evidence is not None:
            from src.models import EvidencePackage
            if isinstance(evidence, EvidencePackage):
                # 提取自检轮次数据
                self_check_data = [
                    {
                        "confidence": round(r.confidence, 3),
                        "next_action": r.next_action.value,
                        "blocking_issues": r.blocking_issues,
                        "missing_slots": r.missing_slots,
                        "reasoning": r.reasoning,
                    }
                    for r in evidence.self_check_rounds
                ]

                # 构建 evidence 数据
                # 如果启用了 reranker 且有排序结果，只保留 ranked_results
                if evidence.ranked_results:
                    ranked_data = []
                    for rr in evidence.ranked_results:
                        item = {
                            "source": rr.source,
                            "content_preview": rr.content[:200],
                            "relevance_score": round(rr.relevance_score, 3),
                        }
                        # 根据来源添加原始数据
                        if rr.source == "internal" and rr.chunk:
                            item.update({
                                "doc_id": rr.chunk.doc_id,
                                "chunk_id": rr.chunk.chunk_id,
                                "original_relevance": round(rr.chunk.relevance_score, 3),
                            })
                        elif rr.source == "external" and rr.evidence:
                            item.update({
                                "tool": rr.evidence.tool_name,
                                "source_name": rr.evidence.source,
                                "url": rr.evidence.url,
                                "query_used": rr.evidence.query_used,
                            })
                        ranked_data.append(item)

                    evidence_data = {
                        "ranked_results": ranked_data,
                        "total_ranked": len(evidence.ranked_results),
                    }
                else:
                    # 未启用 reranker，保留原始的 internal 和 external
                    internal_data = None
                    if evidence.internal_context and evidence.internal_context.chunks:
                        ic = evidence.internal_context
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
                            "snippet": e.snippet,
                            "query_used": e.query_used,
                        }
                        for e in evidence.external_evidence
                    ]

                    evidence_data = {
                        "internal": internal_data,
                        "external": external_data,
                    }

        return {
            "id": result.qa_pair.id,
            "question": result.qa_pair.question,
            "answer": result.qa_pair.answer,
            "self_check_rounds": self_check_data,
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


    def _generate_summary_text(self, report: EvaluationReport) -> str:
        """生成文本摘要（含多轮证据统计）"""
        internal_used = 0
        external_used = 0

        from src.models import EvidencePackage
        for r in report.results:
            if isinstance(r.evidence, EvidencePackage):
                pkg: EvidencePackage = r.evidence
                if pkg.internal_context and pkg.internal_context.chunks:
                    internal_used += 1
                if pkg.external_evidence:
                    external_used += 1

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

