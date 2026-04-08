"""
LLM 评分引擎
基于工具收集的证据，使用 LLM 进行质量评分
支持多种后端: Anthropic, vLLM, OpenAI
"""
import logging
import os
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
    """Anthropic Claude 客户端"""

    def __init__(self, model: str, api_key: str = ""):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        self.client = Anthropic(api_key=self.api_key)

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text


class OpenAICompatibleClient(LLMClient):
    """
    OpenAI 兼容客户端
    支持 vLLM, OpenAI, 或其他兼容 OpenAI API 的服务
    """

    def __init__(
        self,
        model: str,
        base_url: str = "",
        api_key: str = ""
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        # 初始化客户端
        if base_url:
            # 自定义 base_url (如 vLLM)
            self.client = OpenAI(
                base_url=base_url,
                api_key=self.api_key or "not-needed"  # vLLM 可能不需要 key
            )
        else:
            # 默认 OpenAI
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            self.client = OpenAI(api_key=self.api_key)

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content


class LLMScoringEngine:
    """
    LLM 评分引擎
    基于工具证据对医学 QA 进行多维度评分
    支持多种 LLM 后端
    """

    SYSTEM_PROMPT = """你是一位医学数据质量评估专家。你的任务是基于工具检索到的客观证据，对医学问答对进行质量评分。

【评分维度】（总分100）
1. 完整性 (0-30分): 问题是否清晰、回答是否充分、信息是否完整
2. 准确性 (0-45分): 医学知识是否正确、术语是否准确、是否符合指南（重点参考证据）
3. 专业性 (0-25分): 表达是否专业、是否使用标准术语、是否有免责声明

【评分要求】
- 必须基于提供的证据进行评分，特别是准确性维度
- 给出具体分数和详细的评分理由
- 输出必须是有效的 JSON 格式

【输出格式】
{
    "completeness": {"score": 整数0-30, "reason": "评分理由"},
    "accuracy": {"score": 整数0-45, "reason": "评分理由（需引用证据）"},
    "professionalism": {"score": 整数0-25, "reason": "评分理由"}
}"""

    USER_PROMPT_TEMPLATE = """【待评估数据】
Question: {question}
Answer: {answer}

【工具检索到的客观证据】
{evidence_summary}

请基于以上证据进行评分，输出JSON格式:"""

    def __init__(
        self,
        provider: str = "",
        model: str = "",
        base_url: str = "",
        api_key: str = ""
    ):
        # 优先使用传入参数，否则使用配置
        config = get_config()

        raw_provider = provider or config.llm_provider
        if isinstance(raw_provider, str):
            self.provider = str_to_enum(LLMProvider, raw_provider, LLMProvider.ANTHROPIC)
        else:
            self.provider = raw_provider

        self.model = model or config.llm_model
        self.base_url = base_url or config.llm_base_url
        self.api_key = api_key or config.llm_api_key

        # 创建对应客户端
        self.client = self._create_client()

    def _create_client(self) -> LLMClient:
        """根据配置创建 LLM 客户端"""
        if self.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(self.model, self.api_key)
        elif self.provider in [LLMProvider.VLLM, LLMProvider.OPENAI]:
            return OpenAICompatibleClient(self.model, self.base_url, self.api_key)
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

        # 调用 LLM
        try:
            content = self.client.chat_completion(self.SYSTEM_PROMPT, prompt)
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
