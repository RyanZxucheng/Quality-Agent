"""
自检模块
调用 LLM 判断当前证据是否足够，并识别信息缺口
"""
import logging
import threading
from pathlib import Path
from typing import Optional

from src.config import get_config, SelfCheckConfig
from src.models import NextAction, SelfCheckResult
from src.utils.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = “””You are a medical QA quality self-check expert. Your task is to determine whether the currently available evidence is sufficient to make an accurate quality judgment on this medical QA pair — i.e., whether the QA's medical content is correct, complete, and professional — and to identify any key information gaps if needed.

[Core Workflow]
Base tool evidence → Round 0 self-check → If necessary, parallel search (internal KB + external sources) → Final scoring

[Your Judgment Goal]
The question is NOT “do I have all the latest medical knowledge about this topic?” Rather:
“Based on current evidence and my existing knowledge, am I already able to judge the quality of this QA?”

In other words:
- If you can already tell whether the answer is broadly correct or incorrect, you may PROCEED even if minor details remain uncertain
- You should only SEARCH when missing information would directly affect your judgment of the QA's correctness, completeness, or professionalism

[When to Choose PROCEED]
- This involves basic medical knowledge, common pharmacology, adverse reactions, contraindications, routine differential diagnosis, or fundamental treatment principles
- You can clearly determine the answer is broadly correct, incorrect, outdated, or incomplete
- Even if minor details are missing, they would not change your final quality judgment on this QA

[When to Choose SEARCH]
- Correctness clearly depends on the latest guidelines, consensus, evidence-based updates, or version timeliness
- Correctness depends on specific indications, population stratification, staging, typing, mutation status, dosage conditions, or treatment line
- You cannot confirm the treatment plan, recommendation status, applicable conditions, or contraindication boundaries mentioned in the answer
- You currently cannot determine whether the answer is “correct,” “partially correct but incomplete,” or “incorrect”

[System Reference Threshold]
The current system's reference confidence threshold for “can proceed to evaluation” is {min_confidence}.
- If your confidence is significantly below this threshold, you should generally choose SEARCH
- If your confidence meets or exceeds this threshold and no critical blocking issues exist, you may generally choose PROCEED
- This is a reference threshold to help align with system decisions, not a rigid formula; the core standard remains “is the evidence sufficient for quality judgment”

[Output Format]
Strictly output JSON with the following fields:
{
  “confidence”: float between 0 and 1, indicating your confidence in making an accurate quality judgment (not confidence in the answer itself, but in “I can judge this data's quality”),
  “blocking_issues”: [“specific issues blocking your judgment, empty list if none”],
  “missing_slots”: “a description of the single most critical piece of information needed, e.g. 'need to confirm whether osimertinib is still the current standard first-line recommendation for EGFR-mutant NSCLC'. Empty string if no information is missing.”,
  “next_action”: “PROCEED or SEARCH”,
  “reasoning”: “brief reason within 50 words”
}”””


class SelfChecker:
    """
    自检器
    在 Round 0 调用 LLM 判断证据充足性，输出 PROCEED 或 SEARCH
    """

    def __init__(self, config: Optional[SelfCheckConfig] = None):
        self.config = config or get_config().self_check
        self._system_prompt: Optional[str] = None
        self._llm_client = None
        self._warm_up_lock = threading.Lock()

    def warm_up(self) -> None:
        """预加载，避免并发时的懒加载竞争（线程安全）"""
        if self._system_prompt is not None and self._llm_client is not None:
            return
        with self._warm_up_lock:
            if self._system_prompt is not None and self._llm_client is not None:
                return
            _ = self._get_system_prompt()
            _ = self._get_llm_client()
        logger.debug("SelfChecker warmed up")

    def _get_system_prompt(self) -> str:
        """读取 prompt 模板（文件不存在时使用内置默认值）"""
        if self._system_prompt is not None:
            return self._system_prompt

        prompt_path = Path(self.config.prompt_path)
        if prompt_path.exists():
            try:
                prompt_template = prompt_path.read_text(encoding="utf-8")
                self._system_prompt = prompt_template.replace(
                    "{min_confidence}", f"{self.config.min_confidence:.2f}"
                )
                logger.debug(f"Loaded self-check prompt from {prompt_path}")
                return self._system_prompt
            except Exception as e:
                logger.warning(f"Failed to read prompt file {prompt_path}: {e}")

        self._system_prompt = _DEFAULT_SYSTEM_PROMPT.replace(
            "{min_confidence}", f"{self.config.min_confidence:.2f}"
        )
        return self._system_prompt

    def _get_llm_client(self):
        """懒加载 LLM 客户端（复用 llm_engine 的实现）"""
        if self._llm_client is None:
            from src.scoring.llm_engine import AnthropicClient, OpenAICompatibleClient
            cfg = get_config()
            from src.models import LLMProvider
            if cfg.llm_provider == LLMProvider.ANTHROPIC:
                self._llm_client = AnthropicClient(
                    cfg.llm_model, cfg.llm_api_key, cfg.llm_temperature, 512
                )
            else:
                self._llm_client = OpenAICompatibleClient(
                    cfg.llm_model, cfg.llm_temperature, 512,
                    cfg.llm_base_url, cfg.llm_api_key
                )
        return self._llm_client

    def check(self, question: str, answer: str) -> SelfCheckResult:
        """
        执行 Round 0 自检

        Args:
            question: 问题文本
            answer: 回答文本

        Returns:
            SelfCheckResult（next_action 为 PROCEED 或 SEARCH）
        """
        if not self.config.enabled:
            logger.debug("Self-check disabled, returning PROCEED")
            return SelfCheckResult(
                confidence=1.0,
                blocking_issues=[],
                missing_slots="",
                next_action=NextAction.PROCEED,
                reasoning="Self-check disabled",
            )

        user_prompt = self._build_user_prompt(question, answer)

        try:
            client = self._get_llm_client()
            raw = client.chat_completion(self._get_system_prompt(), user_prompt)
            return self._parse_result(raw)
        except Exception as e:
            logger.error(f"Self-check LLM call failed: {e}")
            return SelfCheckResult(
                confidence=0.5,
                blocking_issues=[f"Self-check LLM call failed: {e}"],
                missing_slots="",
                next_action=NextAction.PROCEED,
                reasoning="LLM call exception, conservatively proceeding to scoring",
            )

    def _build_user_prompt(self, question: str, answer: str) -> str:
        lines = [
            "[QA to Evaluate]",
            f"Question: {question}",
            f"Answer: {answer}",
            "",
            "[Available Actions]",
            "- PROCEED: Current evidence is sufficient, proceed to scoring directly",
            "- SEARCH: Insufficient evidence, system will automatically search internal KB and external sources",
            "",
            "Please determine if the above evidence is sufficient for quality review. Output JSON:",
        ]
        return "\n".join(lines)

    def _parse_result(self, raw_text: str) -> SelfCheckResult:
        """解析 LLM 输出，容错处理异常 JSON"""
        try:
            data = parse_llm_json(raw_text)
            next_action_str = str(data.get("next_action", "PROCEED")).upper()
            try:
                next_action = NextAction[next_action_str]
            except KeyError:
                logger.warning(f"Unknown next_action '{next_action_str}', defaulting to PROCEED")
                next_action = NextAction.PROCEED

            confidence = float(data.get("confidence", 0.5))

            return SelfCheckResult(
                confidence=confidence,
                blocking_issues=data.get("blocking_issues", []),
                missing_slots=str(data.get("missing_slots", "")),
                next_action=next_action,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"Failed to parse self-check result: {e}\nRaw: {raw_text[:300]}")
            return SelfCheckResult(
                confidence=0.5,
                blocking_issues=["Failed to parse self-check result"],
                missing_slots="",
                next_action=NextAction.PROCEED,
                reasoning="Parse error, conservatively proceeding to scoring",
            )
