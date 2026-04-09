"""
配置管理
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from src.models import LLMProvider
from src.utils.enum_utils import str_to_enum


@dataclass
class ScoringWeights:
    """评分权重配置"""
    completeness: int = 30
    accuracy: int = 45
    professionalism: int = 25

    def validate(self) -> bool:
        total = self.completeness + self.accuracy + self.professionalism
        return total == 100


@dataclass
class Thresholds:
    """阈值配置"""
    total_min: int = 70
    accuracy_min: int = 35


@dataclass
class SourceWeights:
    """知识源权重"""
    terminology: float = 0.4
    guideline: float = 0.4
    wikipedia: float = 0.2


@dataclass
class AppConfig:
    """应用配置"""
    # 评分配置
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    source_weights: SourceWeights = field(default_factory=SourceWeights)

    # API 配置
    umls_api_key: str = field(default_factory=lambda: os.getenv("UMLS_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # LLM 模型配置（统一入口）
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500

    # NLP 模型配置
    spacy_model: str = "en_core_sci_sm"

    # 批处理配置
    batch_size: int = 10
    max_workers: int = 3

    # 输出配置
    output_dir: str = "data/output"
    input_dir: str = "data/input"

    def __post_init__(self):
        """初始化后处理，转换字符串为枚举"""
        # 处理 llm_provider 从字符串转换
        if isinstance(self.llm_provider, str):
            self.llm_provider = str_to_enum(LLMProvider, self.llm_provider, LLMProvider.ANTHROPIC)

        # 从环境变量填充API密钥（如果未提供）
        self._fill_api_keys_from_env()

        # 如果 llm_api_key 未设置，根据provider设置默认值
        if not self.llm_api_key:
            self._set_default_llm_api_key()

    def _fill_api_keys_from_env(self):
        """从环境变量填充API密钥"""
        if not self.anthropic_api_key and self.llm_provider == LLMProvider.ANTHROPIC:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.openai_api_key and self.llm_provider in [LLMProvider.VLLM, LLMProvider.OPENAI]:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    def _set_default_llm_api_key(self):
        """设置默认的LLM API密钥"""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            self.llm_api_key = self.anthropic_api_key
        elif self.llm_provider in [LLMProvider.VLLM, LLMProvider.OPENAI]:
            self.llm_api_key = self.openai_api_key

    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        if not self.weights.validate():
            errors.append("Scoring weights must sum to 100")

        # 验证必要的API密钥
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            errors.append("Anthropic API key is required for anthropic provider")
        elif self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            errors.append("OpenAI API key is required for openai provider")
        # vLLM可能不需要API密钥

        return errors


# 全局配置实例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig):
    """设置全局配置"""
    global _config
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid config: {', '.join(errors)}")
    _config = config
