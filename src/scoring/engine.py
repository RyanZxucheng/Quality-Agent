"""
评分引擎
实现四个维度的质量评分
"""
import logging
import re
from typing import List, Dict, Any

from src.models import QAPair, ScoreResult, DimensionScore
from src.tools import (
    EntityExtractorTool,
    TerminologyValidatorTool,
    WikipediaVerifierTool,
    GuidelineCheckerTool,
)
from src.config import get_config

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    评分引擎
    对医学 QA 对进行多维度质量评分
    """

    def __init__(self):
        self.config = get_config()
        self.weights = self.config.weights

        # 初始化工具
        self.entity_extractor = EntityExtractorTool()
        self.terminology_validator = TerminologyValidatorTool()
        self.wikipedia_verifier = WikipediaVerifierTool()
        self.guideline_checker = GuidelineCheckerTool()

    def score(self, qa_pair: QAPair) -> ScoreResult:
        """
        对单个 QA 对进行评分

        Returns:
            ScoreResult: 包含各维度分数和总分的结果
        """
        logger.info(f"Scoring QA pair: {qa_pair.id}")

        # 执行各维度评分
        dimensions = []
        all_issues = []

        # 1. 完整性评分
        completeness_result = self._score_completeness(qa_pair)
        dimensions.append(completeness_result["dimension"])
        all_issues.extend(completeness_result.get("issues", []))

        # 2. 准确性评分
        accuracy_result = self._score_accuracy(qa_pair)
        dimensions.append(accuracy_result["dimension"])
        all_issues.extend(accuracy_result.get("issues", []))

        # 3. 专业性评分
        professionalism_result = self._score_professionalism(qa_pair)
        dimensions.append(professionalism_result["dimension"])
        all_issues.extend(professionalism_result.get("issues", []))

        # 4. 基础质量评分
        quality_result = self._score_basic_quality(qa_pair)
        dimensions.append(quality_result["dimension"])
        all_issues.extend(quality_result.get("issues", []))

        # 计算总分
        total_score = sum(d.score for d in dimensions)

        return ScoreResult(
            qa_pair=qa_pair,
            dimensions=dimensions,
            total_score=total_score,
            issues=all_issues
        )

    def _score_completeness(self, qa_pair: QAPair) -> Dict[str, Any]:
        """
        完整性评分 (0-30)
        检查：字段完整、实体覆盖、信息充分
        """
        score = 0
        details = {}
        issues = []

        # 1. 字段完整 (10分)
        field_score = 10
        if len(qa_pair.question.strip()) < 5:
            field_score -= 5
            issues.append({
                "dimension": "completeness",
                "type": "short_question",
                "description": "问题过短，可能信息不完整"
            })
        if len(qa_pair.answer.strip()) < 10:
            field_score -= 5
            issues.append({
                "dimension": "completeness",
                "type": "short_answer",
                "description": "回答过短，可能信息不充分"
            })
        score += max(0, field_score)
        details["field_complete"] = max(0, field_score)

        # 2. 实体覆盖 (10分)
        entity_result = self.entity_extractor.execute(qa_pair.question, qa_pair.answer)
        if entity_result.success:
            entity_count = entity_result.data.get("entity_count", 0)
            entities_by_type = entity_result.data.get("entities_by_type", {})

            if entity_count >= 3:
                entity_score = 10
            elif entity_count >= 1:
                entity_score = 5
            else:
                entity_score = 0
                issues.append({
                    "dimension": "completeness",
                    "type": "no_entities",
                    "description": "未识别到医学实体，可能缺乏医学内容"
                })

            score += entity_score
            details["entity_coverage"] = entity_score
            details["entity_count"] = entity_count
            details["entities_by_type"] = entities_by_type
        else:
            score += 5  # 实体提取失败，给一半分
            details["entity_coverage"] = 5
            details["entity_error"] = entity_result.error

        # 3. 信息充分 (10分)
        # 检查 answer 是否回应了 question
        sufficient_score = self._check_answer_sufficiency(qa_pair)
        score += sufficient_score
        details["information_sufficient"] = sufficient_score

        if sufficient_score < 5:
            issues.append({
                "dimension": "completeness",
                "type": "insufficient_answer",
                "description": "回答可能没有充分回应问题"
            })

        return {
            "dimension": DimensionScore(
                name="completeness",
                score=score,
                max_score=self.weights.completeness,
                details=details
            ),
            "issues": issues
        }

    def _score_accuracy(self, qa_pair: QAPair) -> Dict[str, Any]:
        """
        准确性评分 (0-35)
        检查：术语准确、指南符合、事实验证
        """
        score = 0
        details = {}
        issues = []

        # 1. 术语准确性 (15分)
        term_result = self.terminology_validator.execute(qa_pair.question, qa_pair.answer)
        if term_result.success:
            stats = term_result.data.get("statistics", {})
            standardization_rate = stats.get("standardization_rate", 0)

            # 根据标准化率给分
            term_score = int(15 * standardization_rate)
            score += term_score
            details["terminology_accuracy"] = term_score
            details["standardization_rate"] = standardization_rate
            details["unstandardized_terms"] = term_result.data.get("unstandardized_terms", [])

            if standardization_rate < 0.5:
                issues.append({
                    "dimension": "accuracy",
                    "type": "unstandardized_terms",
                    "description": f"较多术语未标准化（标准化率: {standardization_rate:.1%}）",
                    "terms": term_result.data.get("unstandardized_terms", [])[:5]
                })
        else:
            score += 7  # 验证失败给一半分
            details["terminology_accuracy"] = 7
            details["terminology_error"] = term_result.error

        # 2. 指南符合性 (15分)
        guideline_result = self.guideline_checker.execute(qa_pair.question, qa_pair.answer)
        if guideline_result.success:
            compliance_rate = guideline_result.data.get("compliance_rate", 1.0)
            guideline_issues = guideline_result.data.get("issues", [])

            guideline_score = int(15 * compliance_rate)
            score += guideline_score
            details["guideline_compliance"] = guideline_score
            details["compliance_rate"] = compliance_rate
            details["conditions_checked"] = guideline_result.data.get("conditions_identified", 0)

            # 添加指南相关 issues
            for issue in guideline_issues:
                issue["dimension"] = "accuracy"
                issues.append(issue)
        else:
            score += 7
            details["guideline_compliance"] = 7
            details["guideline_error"] = guideline_result.error

        # 3. 事实验证 (5分)
        wiki_result = self.wikipedia_verifier.execute(qa_pair.question, qa_pair.answer)
        if wiki_result.success:
            confidence = wiki_result.data.get("average_confidence", 0.5)
            wiki_score = int(5 * confidence)
            score += wiki_score
            details["fact_verification"] = wiki_score
            details["wiki_confidence"] = confidence
        else:
            score += 2
            details["fact_verification"] = 2

        return {
            "dimension": DimensionScore(
                name="accuracy",
                score=score,
                max_score=self.weights.accuracy,
                details=details
            ),
            "issues": issues
        }

    def _score_professionalism(self, qa_pair: QAPair) -> Dict[str, Any]:
        """
        专业性评分 (0-25)
        检查：术语规范、表达专业
        """
        score = 0
        details = {}
        issues = []

        answer_lower = qa_pair.answer.lower()

        # 1. 术语规范 (15分)
        # 检查是否使用了口语化表达
        informal_words = ["吃点", "喝点", "睡睡", "休息休息", "有点", "挺", "挺严重的"]
        informal_count = sum(1 for word in informal_words if word in answer_lower)

        term_standard_score = max(0, 15 - informal_count * 3)
        score += term_standard_score
        details["term_standardization"] = term_standard_score

        if informal_count > 0:
            issues.append({
                "dimension": "professionalism",
                "type": "informal_expression",
                "description": f"使用了 {informal_count} 处口语化表达"
            })

        # 2. 表达专业 (10分)
        # 检查是否有结构化表达（分点、逻辑词）
        professional_score = 10

        # 检查是否有明确的逻辑结构
        has_structure = any(marker in answer_lower for marker in [
            "首先", "其次", "此外", "最后", "第一", "第二", "总结",
            "建议", "注意事项", "包括", "例如"
        ])

        if not has_structure:
            professional_score -= 3
            issues.append({
                "dimension": "professionalism",
                "type": "lack_structure",
                "description": "回答缺乏清晰的逻辑结构"
            })

        # 检查是否有免责声明（医疗内容重要）
        has_disclaimer = any(phrase in answer_lower for phrase in [
            "仅供参考", "请咨询", "建议就医", "专业医生", "医疗机构"
        ])

        if not has_disclaimer:
            professional_score -= 2
            issues.append({
                "dimension": "professionalism",
                "type": "no_disclaimer",
                "description": "缺少医疗免责声明"
            })

        score += max(0, professional_score)
        details["expression_professional"] = max(0, professional_score)

        return {
            "dimension": DimensionScore(
                name="professionalism",
                score=score,
                max_score=self.weights.professionalism,
                details=details
            ),
            "issues": issues
        }

    def _score_basic_quality(self, qa_pair: QAPair) -> Dict[str, Any]:
        """
        基础质量评分 (0-10)
        检查：语法正确、可读性
        """
        score = 0
        details = {}
        issues = []

        answer = qa_pair.answer

        # 1. 语法正确 (5分)
        grammar_score = 5

        # 检查明显的语法错误
        # 重复标点
        if any(p * 2 in answer for p in [".", ",", "!", "?", "，", "。", "！", "？"]):
            grammar_score -= 1
            issues.append({
                "dimension": "basic_quality",
                "type": "repeated_punctuation",
                "description": "存在重复标点符号"
            })

        # 检查是否有乱码或特殊字符（简化检查）
        if re.search(r'[^\w\s.,!?;:，。！？；：\-\'"()（）]', answer):
            grammar_score -= 1

        # 检查句子长度是否合理（过长可能是复制粘贴）
        sentences = re.split(r'[.!?。！？]', answer)
        long_sentences = [s for s in sentences if len(s) > 200]
        if len(long_sentences) > 2:
            grammar_score -= 1
            issues.append({
                "dimension": "basic_quality",
                "type": "overlong_sentences",
                "description": "存在过长句子，可能影响可读性"
            })

        score += max(0, grammar_score)
        details["grammar_correct"] = max(0, grammar_score)

        # 2. 可读性 (5分)
        readability_score = 5

        # 检查段落结构
        paragraphs = [p.strip() for p in answer.split('\n') if p.strip()]
        if len(paragraphs) == 0:
            readability_score -= 2

        # 检查是否有明确的开始和结束
        if not any(answer.startswith(w) for w in ["是", "可以", "建议", "根据", "需要", "请"]):
            readability_score -= 1

        score += max(0, readability_score)
        details["readability"] = max(0, readability_score)

        return {
            "dimension": DimensionScore(
                name="basic_quality",
                score=score,
                max_score=self.weights.basic_quality,
                details=details
            ),
            "issues": issues
        }

    def _check_answer_sufficiency(self, qa_pair: QAPair) -> int:
        """
        检查回答是否充分回应问题
        返回 0-10 的分数
        """
        question = qa_pair.question.lower()
        answer = qa_pair.answer.lower()

        # 提取问题的关键词（简化：去掉停用词后的名词）
        stop_words = {"是什么", "怎么", "如何", "为什么", "吗", "呢", "的", "了", "在", "和"}
        q_clean = question
        for sw in stop_words:
            q_clean = q_clean.replace(sw, " ")

        q_keywords = [w for w in q_clean.split() if len(w) >= 2]

        # 检查关键词是否在回答中出现
        if not q_keywords:
            return 5  # 默认中等分数

        matched = sum(1 for kw in q_keywords if kw in answer)
        match_rate = matched / len(q_keywords)

        if match_rate >= 0.7:
            return 10
        elif match_rate >= 0.4:
            return 5
        else:
            return 2
