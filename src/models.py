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


class NextAction(Enum):
    """自检后的下一步动作"""
    PROCEED = "PROCEED"
    SEARCH = "SEARCH"


@dataclass
class QAPair:
    """医学问答对"""
    id: Optional[str] = None
    question: str = ""
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.question or not self.answer:
            raise ValueError("Question and answer cannot be empty")


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
    evidence: Dict[str, Any] = field(default_factory=dict)

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


# ─── 多轮证据收集相关模型 ─────────────────────────────────────────────────────

@dataclass
class SelfCheckResult:
    """自检结果（Round 0 输出）"""
    confidence: float           # 0-1，当前证据支撑评审的把握度
    blocking_issues: List[str]  # 阻碍评审的关键问题
    missing_slots: str          # 需要补充的信息描述（自然语言段落）
    next_action: NextAction     # 下一步动作（PROCEED 或 SEARCH）
    reasoning: str = ""         # 决策理由


@dataclass
class ChunkContext:
    """内部检索命中的文档片段"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int            # 在原文档中的位置序号，用于邻域扩展
    relevance_score: float      # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalContext:
    """内部检索结果（Round 1 输出）"""
    chunks: List[ChunkContext]

    def to_summary_text(self) -> str:
        """生成文本摘要供 LLM 使用"""
        if not self.chunks:
            return "内部知识库：未检索到相关内容"
        lines = ["【内部知识库检索结果】"]
        for i, chunk in enumerate(self.chunks[:5], 1):
            lines.append(f"  [{i}] 文档={chunk.doc_id} 相关度={chunk.relevance_score:.2f}")
            lines.append(f"      {chunk.content[:200]}")
        return "\n".join(lines)


@dataclass
class ExternalEvidence:
    """外部检索结果（Round 3 输出）"""
    tool_name: str
    source: str
    snippet: str
    url: str = ""
    query_used: str = ""


@dataclass
class RankedResult:
    """
    Reranker 输出的统一结果条目

    内部 chunk 和外部 evidence 经 Reranker 混合排序后统一为此格式。
    通过 source 字段区分来源，原始对象保留在 chunk / evidence 字段中。
    """
    source: str                                      # "internal" | "external"
    content: str                                     # passage 文本（供 Reranker 打分）
    relevance_score: float                           # Reranker 输出的相关度分数
    chunk: Optional["ChunkContext"] = None           # 内部检索原始对象
    evidence: Optional["ExternalEvidence"] = None   # 外部检索原始对象


@dataclass
class EvidencePackage:
    """聚合证据包（EvidenceCollector 的最终输出）"""
    qa_id: str
    self_check_rounds: List[SelfCheckResult]
    base_evidence: Dict[str, Any]           # 原有工具收集的证据
    internal_context: Optional[InternalContext]
    external_evidence: List[ExternalEvidence]
    evidence_insufficient: bool             # 经过全部轮次后仍证据不足
    evidence_summary: str                   # 供 LLM 评分使用的完整文本摘要
    rounds_executed: int                    # 实际执行的轮次数
    ranked_results: List[RankedResult] = field(default_factory=list)  # Reranker 输出（启用时）

    # ── 兼容旧版 Dict 接口，使现有 LLMScoringEngine / BatchProcessor 无需大改 ──
    def get(self, key: str, default: Any = None) -> Any:
        if key == "evidence_summary":
            return self.evidence_summary
        return self.base_evidence.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key == "evidence_summary":
            return self.evidence_summary
        return self.base_evidence[key]
