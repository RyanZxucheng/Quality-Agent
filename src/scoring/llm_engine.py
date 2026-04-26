"""
LLM 评分引擎
基于工具收集的证据，使用 LLM 进行质量评分
支持多种后端: Anthropic, vLLM, OpenAI
"""
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from src.config import get_config
from src.models import Dimension, DimensionScore, LLMProvider, QAPair, ScoreResult
from src.utils.enum_utils import str_to_enum
from src.utils.json_utils import parse_llm_json

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 获取回复"""
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude 客户端（底层使用 AsyncAnthropic）"""

    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

        if not api_key:
            raise ValueError("Anthropic API key not provided")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key)

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text


class OpenAICompatibleClient(LLMClient):
    """
    OpenAI 兼容客户端（底层使用 AsyncOpenAI）
    支持 vLLM, OpenAI, 或其他兼容 OpenAI API 的服务
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str = "",
        api_key: str = ""
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if base_url:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed"
            )
        else:
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            self.client = OpenAI(api_key=api_key)

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False
                }
            }
        )
        return response.choices[0].message.content


class LLMScoringEngine:
    """
    LLM 评分引擎
    基于工具证据对医学 QA 进行多维度评分
    支持多种 LLM 后端
    """

    SYSTEM_PROMPT = """You are a medical data quality assessment expert. Your task is to evaluate medical QA pairs based on multi-round collected objective evidence.

[Scoring Dimensions] (Total: 100)
1. Completeness (0-30): Whether the question is clear, the answer is sufficient, and the information is complete.
2. Accuracy (0-45): Whether medical knowledge is correct, terminology is accurate, and guidelines are followed (key: reference evidence).
3. Professionalism (0-25): Whether the expression is professional, uses standard terminology, and includes disclaimers.

[Evidence Credibility Hierarchy]
Evidence is classified into three levels, to be referenced in priority order:
- EXTERNAL (External authoritative sources): PubMed, WHO, NCCN, etc. Highest credibility.
- INTERNAL (Internal knowledge base): Institutional knowledge base retrieval results. High credibility.
- BASE (Base tools): Terminology standardization, entity recognition, Wikipedia, etc. Supplementary reference.

If sources conflict, prioritize higher credibility sources. If evidence is insufficient, note this in the scoring reason.

[Scoring Requirements]
- Must score based on provided evidence, especially for the accuracy dimension.
- Scoring reasons must cite the evidence source (BASE / INTERNAL / EXTERNAL).
- Output must be valid JSON format.

[Output Format]
{
    "completeness": {"score": integer 0-30, "reason": "scoring reason"},
    "accuracy": {"score": integer 0-45, "reason": "scoring reason (cite evidence source)"},
    "professionalism": {"score": integer 0-25, "reason": "scoring reason"}
}"""

    USER_PROMPT_TEMPLATE = """[QA Data to Evaluate]
Question: {question}
Answer: {answer}

[Multi-Round Collected Evidence]
{evidence_summary}

Please score based on the above evidence, prioritizing EXTERNAL > INTERNAL > BASE sources. Output JSON format:"""

    def __init__(
        self,
        provider: str = "",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        temperature: float = None,
        max_tokens: int = None,
    ):
        config = get_config()

        raw_provider = provider or config.llm_provider
        if isinstance(raw_provider, str):
            self.provider = str_to_enum(LLMProvider, raw_provider, LLMProvider.ANTHROPIC)
        else:
            self.provider = raw_provider

        self.model = model or config.llm_model
        self.base_url = base_url or config.llm_base_url
        self.api_key = api_key or config.llm_api_key
        self.temperature = temperature if temperature is not None else config.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else config.llm_max_tokens

        # 每个线程持有独立的异步客户端实例（AsyncAnthropic/AsyncOpenAI 不是线程安全的）
        self._client_local = threading.local()

    def _get_client(self) -> LLMClient:
        """按线程获取独立的 LLM 客户端实例"""
        if not hasattr(self._client_local, "client"):
            self._client_local.client = self._create_client()
        return self._client_local.client

    def _create_client(self) -> LLMClient:
        """根据配置创建 LLM 客户端"""
        if self.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(
                self.model, self.api_key, self.temperature, self.max_tokens
            )
        elif self.provider in [LLMProvider.VLLM, LLMProvider.OPENAI]:
            return OpenAICompatibleClient(
                self.model, self.temperature, self.max_tokens,
                self.base_url, self.api_key
            )
        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

    def score(self, qa_pair: QAPair, evidence: Dict[str, Any]) -> ScoreResult:
        """
        使用 LLM 基于证据进行评分

        Args:
            qa_pair: 待评估的 QA 对
            evidence: 证据收集器返回的证据（包含 evidence_summary）

        Returns:
            ScoreResult: 评分结果
        """
        logger.debug(f"LLM scoring for {qa_pair.id} using {self.provider}/{self.model}")

        # 构造 prompt
        prompt = self.USER_PROMPT_TEMPLATE.format(
            question=qa_pair.question,
            answer=qa_pair.answer,
            evidence_summary=evidence.get("evidence_summary", "无证据")
        )

        # 调用 LLM（按线程取独立客户端实例）
        try:
            client = self._get_client()
            content = client.chat_completion(self.SYSTEM_PROMPT, prompt)
            result = self._parse_json(content)

            # 构建 ScoreResult
            dimensions = [
                DimensionScore(
                    name=Dimension.COMPLETENESS.value,
                    score=result["completeness"]["score"],
                    max_score=30,
                    details={"reason": result["completeness"]["reason"]}
                ),
                DimensionScore(
                    name=Dimension.ACCURACY.value,
                    score=result["accuracy"]["score"],
                    max_score=45,
                    details={"reason": result["accuracy"]["reason"]}
                ),
                DimensionScore(
                    name=Dimension.PROFESSIONALISM.value,
                    score=result["professionalism"]["score"],
                    max_score=25,
                    details={"reason": result["professionalism"]["reason"]}
                ),
            ]

            total_score = sum(d.score for d in dimensions)

            return ScoreResult(
                qa_pair=qa_pair,
                dimensions=dimensions,
                total_score=total_score,
                issues=[]
            )

        except Exception as e:
            logger.error(f"LLM scoring failed for {qa_pair.id}: {e}")
            return self._create_error_result(qa_pair, str(e))

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON"""
        return parse_llm_json(text)

    def _create_error_result(self, qa_pair: QAPair, error: str) -> ScoreResult:
        """创建错误结果"""
        return ScoreResult.create_error_result(qa_pair, error, "llm_scoring_error")
