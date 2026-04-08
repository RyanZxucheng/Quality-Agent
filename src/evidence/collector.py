"""
证据收集器
并行调用多个工具收集与QA相关的证据
"""
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import QAPair
from src.tools import (
    EntityExtractorTool,
    TerminologyValidatorTool,
    WikipediaVerifierTool,
    GuidelineCheckerTool,
)

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """
    证据收集器
    并行调用多个工具，收集支持评分决策的证据
    """

    def __init__(self):
        # 初始化工具（这些可以复用）
        self.entity_extractor = EntityExtractorTool()
        self.terminology = TerminologyValidatorTool()
        self.wikipedia = WikipediaVerifierTool()
        self.guideline = GuidelineCheckerTool()

    def collect(self, qa_pair: QAPair) -> Dict[str, Any]:
        """
        收集所有相关证据

        Returns:
            {
                "entities": [...],
                "terminology_validation": {...},
                "wikipedia_verification": {...},
                "guideline_check": {...},
                "evidence_summary": "文本形式的证据摘要"
            }
        """
        logger.info(f"Collecting evidence for {qa_pair.id}")

        evidence = {
            "qa_id": qa_pair.id,
            "question": qa_pair.question,
            "answer": qa_pair.answer,
        }

        # 并行执行工具调用
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._collect_entities, qa_pair): "entities",
                executor.submit(self._collect_terminology, qa_pair): "terminology",
                executor.submit(self._collect_wikipedia, qa_pair): "wikipedia",
                executor.submit(self._collect_guideline, qa_pair): "guideline",
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    evidence[key] = future.result()
                except Exception as e:
                    logger.error(f"Failed to collect {key}: {e}")
                    evidence[key] = {"error": str(e)}

        # 生成文本形式的证据摘要（给LLM用）
        evidence["evidence_summary"] = self._generate_summary(evidence)

        return evidence

    def _collect_entities(self, qa_pair: QAPair) -> Dict[str, Any]:
        """收集实体识别证据"""
        result = self.entity_extractor.execute(qa_pair.question, qa_pair.answer)
        if result.success:
            return {
                "found": True,
                "entities": result.data.get("entities", []),
                "entities_by_type": result.data.get("entities_by_type", {}),
                "entity_count": result.data.get("entity_count", 0),
            }
        return {"found": False, "error": result.error}

    def _collect_terminology(self, qa_pair: QAPair) -> Dict[str, Any]:
        """收集术语验证证据"""
        result = self.terminology.execute(qa_pair.question, qa_pair.answer)
        if result.success:
            stats = result.data.get("statistics", {})
            return {
                "found": True,
                "standardization_rate": stats.get("standardization_rate", 0),
                "standardized_terms_count": stats.get("standardized", 0),
                "unstandardized_terms": result.data.get("unstandardized_terms", [])[:10],
                "in_icd10": stats.get("in_icd10", 0),
                "in_snomed": stats.get("in_snomed", 0),
            }
        return {"found": False, "error": result.error}

    def _collect_wikipedia(self, qa_pair: QAPair) -> Dict[str, Any]:
        """收集维基百科验证证据"""
        result = self.wikipedia.execute(qa_pair.question, qa_pair.answer)
        if result.success:
            return {
                "found": True,
                "average_confidence": result.data.get("average_confidence", 0),
                "entities_checked": result.data.get("entities_checked", 0),
                "entities_found": result.data.get("entities_found", 0),
                "verification_details": [
                    r for r in result.data.get("verification_results", [])
                    if r.get("found")
                ][:5],
            }
        return {"found": False, "error": result.error}

    def _collect_guideline(self, qa_pair: QAPair) -> Dict[str, Any]:
        """收集临床指南证据"""
        result = self.guideline.execute(qa_pair.question, qa_pair.answer)
        if result.success:
            return {
                "found": True,
                "conditions_identified": result.data.get("conditions_identified", 0),
                "compliance_rate": result.data.get("compliance_rate", 1.0),
                "issues": result.data.get("issues", []),
            }
        return {"found": False, "error": result.error}

    def _generate_summary(self, evidence: Dict[str, Any]) -> str:
        """生成证据的文本摘要，供LLM使用"""
        lines = []

        # 实体信息
        entities = evidence.get("entities", {})
        if entities.get("found"):
            lines.append(f"【识别到的医学实体】({entities.get('entity_count', 0)}个)")
            by_type = entities.get("entities_by_type", {})
            for etype, ents in by_type.items():
                lines.append(f"  - {etype}: {', '.join(ents[:5])}")

        # 术语验证
        term = evidence.get("terminology", {})
        if term.get("found"):
            lines.append(f"\n【术语标准化情况】")
            lines.append(f"  - 标准化率: {term.get('standardization_rate', 0):.1%}")
            lines.append(f"  - ICD-10匹配: {term.get('in_icd10', 0)}个")
            lines.append(f"  - SNOMED匹配: {term.get('in_snomed', 0)}个")
            if term.get("unstandardized_terms"):
                lines.append(f"  - 未标准化术语: {', '.join(term['unstandardized_terms'][:5])}")

        # 维基百科验证
        wiki = evidence.get("wikipedia", {})
        if wiki.get("found"):
            lines.append(f"\n【维基百科验证】")
            lines.append(f"  - 验证实体数: {wiki.get('entities_checked', 0)}")
            lines.append(f"  - 找到词条: {wiki.get('entities_found', 0)}")
            lines.append(f"  - 平均置信度: {wiki.get('average_confidence', 0):.2f}")

        # 指南检查
        guide = evidence.get("guideline", {})
        if guide.get("found"):
            lines.append(f"\n【临床指南符合性】")
            lines.append(f"  - 识别疾病: {guide.get('conditions_identified', 0)}种")
            lines.append(f"  - 合规率: {guide.get('compliance_rate', 1.0):.1%}")
            if guide.get("issues"):
                lines.append(f"  - 发现的问题:")
                for issue in guide["issues"][:3]:
                    lines.append(f"    * {issue.get('description', '')}")

        return "\n".join(lines) if lines else "未收集到有效证据"
