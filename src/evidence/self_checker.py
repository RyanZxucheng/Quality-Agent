"""
自检模块
调用 LLM 判断当前证据是否足够，并识别信息缺口
"""
import logging
from pathlib import Path
from typing import Optional

from src.config import get_config, SelfCheckConfig
from src.models import NextAction, SelfCheckResult
from src.utils.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """你是医学 QA 质量自检专家。你的任务是判断当前已有证据是否足以对这条医学问答数据做出准确的质量判断，也就是判断这条 QA 的医学内容是否正确、完整、专业，并在必要时指出仍需补充的关键信息。

【核心流程】
基础工具证据 → Round 0 自检判断 → 若有必要则并行检索（内部知识库 + 外部来源）→ 最终评分

【你的判断目标】
你要回答的不是“我是否掌握了这道题所有最新最全的医学知识”，而是：
“基于当前证据和我的已有知识，我是否已经足以判断这条 QA 的质量好坏？”

换句话说：
- 如果你已经能判断回答大方向正确或错误，即使仍有少量细节拿不准，也可以 PROCEED
- 只有当这些缺失信息会直接影响你对 QA 正误、完整性或专业性的判断时，才应该 SEARCH

【何时选择 PROCEED】
- 这是基础医学常识、常见药理、不良反应、禁忌、常规鉴别、基础诊疗原则
- 你已经能够明确判断答案的大方向正确、错误、过时或不完整
- 即使缺少少量细节，也不会改变你对这条 QA 质量的最终判断

【何时选择 SEARCH】
- 是否正确明显依赖最新指南、共识、循证更新或版本时效性
- 是否正确取决于特定适应症、人群分层、分期、分型、突变状态、剂量条件或治疗线别
- 你无法确认回答提到的治疗方案、推荐地位、适用条件或禁忌边界
- 你目前无法判断回答究竟是“正确”“部分正确但不完整”还是“错误”

【系统参考阈值】
当前系统对“可以直接进入评审”的参考置信度阈值是 {min_confidence}。
- 如果你的把握明显低于这个阈值，通常应选择 SEARCH
- 如果你的把握达到或高于这个阈值，且不存在关键阻碍问题，通常可选择 PROCEED
- 这是帮助你与系统决策保持一致的参考阈值，不是死板公式；最终仍应以“是否足以做质量判断”为核心标准

【输出格式】
严格输出 JSON，包含以下字段：
{
  "confidence": 0到1之间的小数，表示你对当前能否做出准确质量判断的把握程度（不是对答案本身的把握，而是对"我能判断这条数据质量"的把握），
  "blocking_issues": ["阻碍你做出判断的具体问题，若没有则为空列表"],
  "missing_slots": "一段描述当前最需要补充的一个关键信息的文字，例如'需要确认奥希替尼是否仍属于EGFR突变NSCLC的当前标准一线推荐'。若不缺少任何信息则为空字符串",
  "next_action": "PROCEED 或 SEARCH",
  "reasoning": "不超过50字的简短理由"
}"""


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
                reasoning="自检已禁用",
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
                blocking_issues=[f"自检调用失败: {e}"],
                missing_slots="",
                next_action=NextAction.PROCEED,
                reasoning="LLM 调用异常，保守地直接评分",
            )

    def _build_user_prompt(self, question: str, answer: str) -> str:
        lines = [
            "【待评审 QA】",
            f"问题：{question}",
            f"回答：{answer}",
            "",
            "【可选动作】",
            "- PROCEED：当前证据已足够，直接进行评分",
            "- SEARCH：证据不足，系统将自动并行检索内部知识库与外部来源",
            "",
            "请判断上述证据是否足以完成质量评审，输出 JSON：",
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
                blocking_issues=["解析自检结果失败"],
                missing_slots="",
                next_action=NextAction.PROCEED,
                reasoning="解析异常，保守直接评分",
            )
