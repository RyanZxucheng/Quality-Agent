"""
配置管理
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import logging

from src.models import LLMProvider
from src.utils.enum_utils import str_to_enum

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 文件，不存在或解析失败时返回空字典"""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Failed to load YAML config {path}: {e}")
        return {}


@dataclass
class SelfCheckConfig:
    """自检模块配置"""
    enabled: bool = True
    min_confidence: float = 0.7    # 作为 prompt 中“可直接评审”参考阈值
    min_rounds: int = 0            # 强制最少内部检索轮次（0 表示不强制）
    prompt_path: str = "config/prompts/self_check.md"

    @classmethod
    def from_yaml(cls, path: str = "config/self_check.yaml") -> "SelfCheckConfig":
        data = _load_yaml(path)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InternalSearchConfig:
    """内部检索配置"""
    enabled: bool = False           # 默认关闭，需建立索引后开启
    index_dir: str = "data/index"   # 索引目录（含 chunks.jsonl）
    bm25_top_k: int = 10
    vector_top_k: int = 10
    fusion_top_k: int = 3           # RRF 融合后保留的片段数

    @classmethod
    def from_yaml(cls, path: str = "config/internal_search.yaml") -> "InternalSearchConfig":
        data = _load_yaml(path)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LocalRerankConfig:
    """本地 CrossEncoder 模型配置"""
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class ApiRerankConfig:
    """API Reranker 配置（Jina / Cohere）"""
    provider: str = "jina"                              # jina | cohere
    model_name: str = "jina-reranker-v2-base-multilingual"
    api_key: str = ""                                   # 也可用 RERANK_API_KEY 环境变量


@dataclass
class RerankConfig:
    """Reranker 顶层配置"""
    enabled: bool = False
    backend: str = "local"                              # local | api
    local: LocalRerankConfig = field(default_factory=LocalRerankConfig)
    api: ApiRerankConfig = field(default_factory=ApiRerankConfig)
    top_n: int = 5

    @classmethod
    def from_yaml(cls, path: str = "config/rerank.yaml") -> "RerankConfig":
        data = _load_yaml(path)

        local_data = data.pop("local", {})
        api_data = data.pop("api", {})

        # 环境变量覆盖 api_key
        env_key = os.getenv("RERANK_API_KEY", "")
        if env_key:
            api_data["api_key"] = env_key

        obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if local_data:
            obj.local = LocalRerankConfig(
                **{k: v for k, v in local_data.items() if k in LocalRerankConfig.__dataclass_fields__}
            )
        if api_data:
            obj.api = ApiRerankConfig(
                **{k: v for k, v in api_data.items() if k in ApiRerankConfig.__dataclass_fields__}
            )
        return obj


@dataclass
class ExternalToolConfig:
    """单个外部工具配置"""
    name: str = ""
    enabled: bool = True
    query_template: str = "{missing_slot} medical evidence"
    api_key: str = ""
    endpoint: str = ""


@dataclass
class ExternalSearchConfig:
    """外部检索配置"""
    enabled: bool = False
    max_results_per_tool: int = 2
    timeout: int = 10
    tools: List[ExternalToolConfig] = field(default_factory=lambda: [
        ExternalToolConfig(name="pubmed", enabled=True,
                           query_template="{missing_slot} clinical evidence"),
    ])

    @classmethod
    def from_yaml(cls, path: str = "config/external_tools.yaml") -> "ExternalSearchConfig":
        data = _load_yaml(path)
        raw_tools = data.pop("tools", None)
        obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "tools"})
        if raw_tools:
            obj.tools = [
                ExternalToolConfig(**{k: v for k, v in t.items() if k in ExternalToolConfig.__dataclass_fields__})
                for t in raw_tools
            ]
        return obj


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
class AppConfig:
    """应用配置"""
    # 评分配置
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)

    # 多轮证据收集配置
    self_check: SelfCheckConfig = field(default_factory=SelfCheckConfig)
    internal_search: InternalSearchConfig = field(default_factory=InternalSearchConfig)
    external_search: ExternalSearchConfig = field(default_factory=ExternalSearchConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)

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

    # 批处理配置
    batch_size: int = 10
    max_workers: int = 3
    max_retained: Optional[int] = None

    # 输出配置
    output_dir: str = "data/output"
    input_dir: str = "data/input"

    def __post_init__(self):
        """初始化后处理，转换字符串为枚举"""
        if isinstance(self.llm_provider, str):
            self.llm_provider = str_to_enum(LLMProvider, self.llm_provider, LLMProvider.ANTHROPIC)

        self._fill_api_keys_from_env()

        if not self.llm_api_key:
            self._set_default_llm_api_key()

        # 从 YAML 文件加载子配置（文件不存在时使用默认值）
        self._load_sub_configs()

    def _load_sub_configs(self):
        """从 YAML 文件加载子模块配置"""
        self.self_check = SelfCheckConfig.from_yaml()
        self.internal_search = InternalSearchConfig.from_yaml()
        self.external_search = ExternalSearchConfig.from_yaml()
        self.rerank = RerankConfig.from_yaml()

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
