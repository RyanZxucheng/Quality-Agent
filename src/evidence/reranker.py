"""
Reranker 模块

支持三种后端，统一接口：
  - local   : 本地 CrossEncoder 模型（sentence-transformers，无需 API key 和显卡）
  - cohere  : Cohere Rerank API（需 api_key，pip install cohere）
  - jina    : Jina AI Reranker API（需 api_key，纯 HTTP，无额外依赖）

输入：query 字符串 + RankedResult 候选列表（内部 chunk 与外部 evidence 混合）
输出：按 relevance_score 降序排列的 RankedResult 列表（取 top_n）
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── 后端基类 ──────────────────────────────────────────────────────────────────

class BaseReranker(ABC):
    """Reranker 后端基类"""

    @abstractmethod
    def rerank(self, query: str, candidates: list, top_n: int) -> list:
        """
        对候选结果重排序

        Args:
            query: 检索查询（原始问题）
            candidates: RankedResult 候选列表
            top_n: 返回的最大结果数

        Returns:
            重排后按 relevance_score 降序排列的列表，长度 <= top_n
        """


# ── 本地 CrossEncoder ─────────────────────────────────────────────────────────

class LocalReranker(BaseReranker):
    """
    本地 CrossEncoder 模型 Reranker

    依赖：pip install sentence-transformers
    推荐模型（中英双语，适合医疗场景）：BAAI/bge-reranker-base
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded local reranker: {self.model_name} on {self.device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        return self._model

    def rerank(self, query: str, candidates: list, top_n: int) -> list:
        if not candidates:
            return candidates

        model = self._get_model()
        pairs = [(query, c.content) for c in candidates]

        try:
            scores = model.predict(pairs, batch_size=self.batch_size)
            for candidate, score in zip(candidates, scores):
                candidate.relevance_score = float(score)
            ranked = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
            return ranked[:top_n]
        except Exception as e:
            logger.error(f"Local reranker inference failed: {e}")
            return candidates[:top_n]


# ── Cohere Rerank API ─────────────────────────────────────────────────────────

class CohereReranker(BaseReranker):
    """
    Cohere Rerank API

    依赖：pip install cohere
    默认模型：rerank-v3.5
    """

    DEFAULT_MODEL = "rerank-v3.5"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import cohere
            self._client = cohere.Client(api_key=self.api_key)
            logger.info(f"Cohere reranker initialized (model={self.model})")
        except ImportError:
            raise ImportError(
                "cohere not installed. Install with: pip install cohere"
            )
        return self._client

    def rerank(self, query: str, candidates: list, top_n: int) -> list:
        if not candidates:
            return candidates

        client = self._get_client()
        docs = [c.content for c in candidates]

        try:
            response = client.rerank(
                query=query,
                documents=docs,
                model=self.model,
                top_n=min(top_n, len(candidates)),
            )
            reranked = []
            for item in response.results:
                candidate = candidates[item.index]
                candidate.relevance_score = float(item.relevance_score)
                reranked.append(candidate)
            return reranked
        except Exception as e:
            logger.error(f"Cohere reranker failed: {e}")
            return candidates[:top_n]


# ── Jina AI Reranker API ──────────────────────────────────────────────────────

class JinaReranker(BaseReranker):
    """
    Jina AI Reranker API（纯 HTTP，无额外依赖）

    默认模型：jina-reranker-v2-base-multilingual（支持中文）
    """

    ENDPOINT = "https://api.jina.ai/v1/rerank"
    DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 15):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout

    def rerank(self, query: str, candidates: list, top_n: int) -> list:
        if not candidates:
            return candidates

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": [c.content for c in candidates],
            "top_n": min(top_n, len(candidates)),
        }

        try:
            resp = requests.post(
                self.ENDPOINT, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()

            reranked = []
            for item in data.get("results", []):
                idx = item["index"]
                candidate = candidates[idx]
                candidate.relevance_score = float(item["relevance_score"])
                reranked.append(candidate)
            return reranked
        except Exception as e:
            logger.error(f"Jina reranker failed: {e}")
            return candidates[:top_n]


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def create_reranker(config) -> Optional[BaseReranker]:
    """
    根据 RerankConfig 创建对应的 Reranker 实例

    Returns:
        BaseReranker 实例，或 None（disabled / 配置错误时）
    """
    if not config.enabled:
        return None

    backend = config.backend.lower()

    if backend == "local":
        return LocalReranker(
            model_name=config.local.model_name,
            device=config.local.device,
            batch_size=config.local.batch_size,
        )

    if backend == "api":
        api_key = config.api.api_key or os.getenv("RERANK_API_KEY", "")
        if not api_key:
            logger.warning(
                f"API reranker ({config.api.provider}) requires api_key "
                "or RERANK_API_KEY env. Reranker disabled."
            )
            return None

        provider = config.api.provider.lower()
        if provider == "cohere":
            return CohereReranker(api_key=api_key, model=config.api.model_name)
        if provider == "jina":
            return JinaReranker(api_key=api_key, model=config.api.model_name)

        logger.warning(f"Unknown API reranker provider: '{config.api.provider}'. Reranker disabled.")
        return None

    logger.warning(f"Unknown reranker backend: '{config.backend}'. Reranker disabled.")
    return None
