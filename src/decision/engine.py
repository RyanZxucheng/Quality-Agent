"""
决策引擎
根据评分结果决定保留或丢弃数据
"""
import logging
from typing import Any, Dict, List, Optional

from src.models import EvidencePackage, ScoreResult, Conclusion, EvaluationResult, Dimension
from src.config import get_config

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    决策引擎
    根据评分结果和阈值决定数据的保留或丢弃
    """

    def __init__(self):
        self.config = get_config()
        self.thresholds = self.config.thresholds

    def decide(
        self,
        score_result: ScoreResult,
        evidence_package: Optional[EvidencePackage] = None,
    ) -> EvaluationResult:
        """
        根据评分结果做出决策

        Args:
            score_result: 评分结果
            evidence_package: 证据包（可选，用于判断证据不足状态）

        Returns:
            EvaluationResult: 包含决策结论和原因
        """
        qa_pair = score_result.qa_pair
        total_score = score_result.total_score
        accuracy_score = score_result.get_dimension_score(Dimension.ACCURACY.value) or 0

        # 优先检查证据不足标志
        if evidence_package is not None and evidence_package.evidence_insufficient:
            logger.info(
                f"Decision for {qa_pair.id}: DISCARD (evidence insufficient after all rounds)"
            )
            return EvaluationResult(
                qa_pair=qa_pair,
                scores=score_result,
                conclusion=Conclusion.DISCARD,
                reason="证据不足，待人工复核",
            )

        # 正常决策逻辑
        conclusion, reason = self._make_decision(total_score, accuracy_score, score_result)

        logger.info(
            f"Decision for {qa_pair.id}: {conclusion.value} "
            f"(total={total_score}, accuracy={accuracy_score})"
        )

        return EvaluationResult(
            qa_pair=qa_pair,
            scores=score_result,
            conclusion=conclusion,
            reason=reason,
        )

    def _get_critical_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取严重问题列表"""
        return [
            i for i in issues
            if i.get("severity") == "high" or i.get("type") == "contraindicated_treatment"
        ]

    def _has_critical_issues(self, issues: List[Dict[str, Any]]) -> bool:
        """检查是否存在严重问题"""
        return len(self._get_critical_issues(issues)) > 0

    def _build_retain_reasons(self, total_score: int, accuracy_score: int) -> str:
        """构建保留原因"""
        reasons = []
        if total_score >= 80:
            reasons.append(f"质量优秀（总分 {total_score}）")
        else:
            reasons.append(f"质量合格（总分 {total_score}）")

        if accuracy_score >= 30:
            reasons.append("医学准确性良好")

        return "; ".join(reasons)

    def _make_decision(
        self,
        total_score: int,
        accuracy_score: int,
        score_result: ScoreResult
    ) -> tuple[Conclusion, str]:
        """
        核心决策逻辑

        Returns:
            tuple: (结论, 原因)
        """
        issues = score_result.issues

        # 硬性否决条件
        # 1. 总分不达标
        if total_score < self.thresholds.total_min:
            return (
                Conclusion.DISCARD,
                f"总分 {total_score} 低于阈值 {self.thresholds.total_min}"
            )

        # 2. 准确性不达标
        if accuracy_score < self.thresholds.accuracy_min:
            return (
                Conclusion.DISCARD,
                f"准确性分数 {accuracy_score} 低于阈值 {self.thresholds.accuracy_min}"
            )

        # 3. 检查是否有严重错误
        critical_issues = self._get_critical_issues(issues)
        if critical_issues:
            issue_types = ", ".join(set(i.get("type", "unknown") for i in critical_issues))
            return (
                Conclusion.DISCARD,
                f"存在严重问题: {issue_types}"
            )

        # 通过所有检查，保留数据
        return Conclusion.RETAIN, self._build_retain_reasons(total_score, accuracy_score)

    def decide_batch(
        self,
        score_results: List[ScoreResult],
        evidence_packages: Optional[List[Optional[EvidencePackage]]] = None,
    ) -> List[EvaluationResult]:
        """
        批量决策

        Args:
            score_results: 评分结果列表
            evidence_packages: 对应的证据包列表（可选）

        Returns:
            评估结果列表
        """
        if evidence_packages is None:
            evidence_packages = [None] * len(score_results)
        return [
            self.decide(sr, ep)
            for sr, ep in zip(score_results, evidence_packages)
        ]

    def get_statistics(self, results: List[EvaluationResult]) -> dict:
        """
        获取决策统计信息

        Args:
            results: 评估结果列表

        Returns:
            统计信息字典
        """
        total = len(results)
        retained = sum(1 for r in results if r.conclusion == Conclusion.RETAIN)
        discarded = total - retained

        retention_rate = retained / total if total > 0 else 0

        # 丢弃原因统计
        discard_reasons = {}
        for r in results:
            if r.conclusion == Conclusion.DISCARD:
                # 提取主要原因
                reason_key = r.reason.split("(")[0].strip()
                discard_reasons[reason_key] = discard_reasons.get(reason_key, 0) + 1

        return {
            "total": total,
            "retained": retained,
            "discarded": discarded,
            "retention_rate": retention_rate,
            "discard_reasons": discard_reasons
        }
