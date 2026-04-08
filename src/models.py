"""
数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Conclusion(Enum):
    """评估结论"""
    RETAIN = "RETAIN"
    DISCARD = "DISCARD"


class Dimension(Enum):
    """评分维度"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    PROFESSIONALISM = "professionalism"


class LLMProvider(Enum):
    """LLM 提供商"""
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    OPENAI = "openai"


@dataclass
class QAPair:
    """医学问答对"""
    id: str
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("ID cannot be empty")
        if not self.question or not self.answer:
            raise ValueError("Question and answer cannot be empty")


@dataclass
class MedicalEntity:
    """医学实体"""
    text: str
    label: str  # 如: DISEASE, DRUG, SYMPTOM
    start: int
    end: int
    normalized_term: Optional[str] = None  # 标准化术语


@dataclass
class ValidationResult:
    """验证结果"""
    tool_name: str
    is_valid: bool
    score: float  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class DimensionScore:
    """维度评分"""
    name: str
    score: int
    max_score: int
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0


@dataclass
class ScoreResult:
    """评分结果"""
    qa_pair: QAPair
    dimensions: List[DimensionScore]
    total_score: int
    issues: List[Dict[str, Any]] = field(default_factory=list)

    def get_dimension_score(self, name: str) -> Optional[int]:
        """获取指定维度的分数"""
        for dim in self.dimensions:
            if dim.name == name:
                return dim.score
        return None

    @classmethod
    def create_error_result(cls, qa_pair: QAPair, error: str, error_type: str = "processing_error") -> "ScoreResult":
        """创建错误评分结果"""
        return cls(
            qa_pair=qa_pair,
            dimensions=[
                DimensionScore(name=Dimension.COMPLETENESS.value, score=0, max_score=30),
                DimensionScore(name=Dimension.ACCURACY.value, score=0, max_score=45),
                DimensionScore(name=Dimension.PROFESSIONALISM.value, score=0, max_score=25),
            ],
            total_score=0,
            issues=[{"type": error_type, "error": error}]
        )


@dataclass
class EvaluationResult:
    """单条 QA 的评估结果"""
    qa_pair: QAPair
    scores: ScoreResult
    conclusion: Conclusion
    reason: str

    @classmethod
    def create_error_result(cls, qa_pair: QAPair, error: str, conclusion: Conclusion = Conclusion.DISCARD) -> "EvaluationResult":
        """创建错误评估结果"""
        return cls(
            qa_pair=qa_pair,
            scores=ScoreResult.create_error_result(qa_pair, error, "processing_error"),
            conclusion=conclusion,
            reason=f"处理错误: {error}"
        )


@dataclass
class EvaluationSummary:
    """评估摘要"""
    total_processed: int
    retained: int
    discarded: int
    retention_rate: float
    average_score: float
    dimension_averages: Dict[str, float]


@dataclass
class EvaluationReport:
    """完整评估报告"""
    summary: EvaluationSummary
    results: List[EvaluationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
