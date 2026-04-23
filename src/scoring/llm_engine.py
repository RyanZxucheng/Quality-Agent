"""
LLM 评分引擎
基于工具收集的证据，使用 LLM 进行质量评分
支持多种后端: Anthropic, vLLM, OpenAI
"""
import asyncio
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
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

        if not api_key:
            raise ValueError("Anthropic API key not provided")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=api_key)
        self._loop_cache = threading.local()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if not hasattr(self._loop_cache, "loop"):
            loop = asyncio.new_event_loop()
            self._loop_cache.loop = loop
        return self._loop_cache.loop

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        loop = self._get_loop()
        return loop.run_until_complete(
            self._achat(system_prompt, user_prompt)
        )

    async def _achat(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.messages.create(
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
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._loop_cache = threading.local()

        if base_url:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed"
            )
        else:
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            self.client = AsyncOpenAI(api_key=api_key)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if not hasattr(self._loop_cache, "loop"):
            loop = asyncio.new_event_loop()
            self._loop_cache.loop = loop
        return self._loop_cache.loop

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        loop = self._get_loop()
        return loop.run_until_complete(
            self._achat(system_prompt, user_prompt)
        )

    async def _achat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.client.chat.completions.create(
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

    SYSTEM_PROMPT = """你是一位医学数据质量评估专家。你的任务是基于多轮收集的客观证据，对医学问答对进行质量评分。

【评分维度】（总分100）
1. 完整性 (0-30分): 问题是否清晰、回答是否充分、信息是否完整
2. 准确性 (0-45分): 医学知识是否正确、术语是否准确、是否符合指南（重点参考证据）
3. 专业性 (0-25分): 表达是否专业、是否使用标准术语、是否有免责声明

【证据可信度层级】
证据分为三类，评分时按优先级从高到低参考：
- EXTERNAL（外部权威来源）：PubMed、WHO、NCCN 等，可信度最高
- INTERNAL（内部知识库）：机构知识库检索结果，可信度高
- BASE（基础工具）：术语标准化、实体识别、维基百科等，作为辅助参考

若不同来源结论冲突，优先采信高可信度来源；若证据明显不足，应在评分理由中注明。

【评分要求】
- 必须基于提供的证据进行评分，特别是准确性维度
- 评分理由需指明引用的证据来源（BASE / INTERNAL / EXTERNAL）
- 输出必须是有效的 JSON 格式

【输出格式】
{
    "completeness": {"score": 整数0-30, "reason": "评分理由"},
    "accuracy": {"score": 整数0-45, "reason": "评分理由（需引用证据来源）"},
    "professionalism": {"score": 整数0-25, "reason": "评分理由"}
}"""

    USER_PROMPT_TEMPLATE = """【待评估数据】
Question: {question}
Answer: {answer}

【多轮收集的客观证据】
{evidence_summary}

请基于以上证据进行评分，优先参考 EXTERNAL > INTERNAL > BASE 来源，输出JSON格式:"""

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
        logger.info(f"LLM scoring for {qa_pair.id} using {self.provider}/{self.model}")

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
