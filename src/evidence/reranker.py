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
import threading
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class UnreliableCheckpointError(RuntimeError):
    """模型权重缺失关键打分头时抛出，避免使用不可靠分数。"""


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

    多线程安全：用 _infer_lock 串行化 GPU 推理，避免多个 QA 线程并发抢占同一模型
    导致显存 OOM 或推理抖动。如需更高吞吐，可改为跨 QA 微批队列。
    """

    SAFE_FALLBACK_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._engine = "cross_encoder"
        self._infer_lock = threading.Lock()

    def _prefer_hf_sequence_classifier(self) -> bool:
        """Qwen reranker 更适合走 transformers 原生 SequenceClassification 路径。"""
        name = self.model_name.lower()
        return "qwen" in name and "reranker" in name

    def _resolve_device(self, requested_device: str) -> str:
        """当 CUDA 不可用时自动回退 CPU，避免本地重排完全失效。"""
        device = (requested_device or "cpu").lower()
        if not device.startswith("cuda"):
            return requested_device

        try:
            import torch
            if torch.cuda.is_available():
                return requested_device
        except Exception:
            pass

        logger.warning(
            f"CUDA device '{requested_device}' is unavailable. "
            "Falling back to CPU for local reranker."
        )
        return "cpu"

    def _load_hf_sequence_classifier(self, device: str) -> None:
        """使用 transformers 加载 SequenceClassification 形式的 reranker。"""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )

        model_kwargs = {"trust_remote_code": True}
        if device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        missing_keys = set()
        try:
            loaded = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                output_loading_info=True,
                **model_kwargs,
            )
            if isinstance(loaded, tuple):
                model, loading_info = loaded
                missing_keys = set(loading_info.get("missing_keys") or [])
            else:
                model = loaded
        except TypeError:
            # 老版本 transformers 可能不支持 output_loading_info
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

        if any(key.endswith("score.weight") for key in missing_keys):
            raise UnreliableCheckpointError(
                "Model checkpoint is missing 'score.weight'. "
                "This usually means the model is not fully compatible with "
                "the current SequenceClassification loading path."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._engine = "hf_sequence_classifier"
        logger.debug(f"Loaded HF local reranker: {self.model_name} on {device}")

    def _predict_hf_scores(self, query: str, candidates: list) -> list:
        """用 HF SequenceClassification 模型对 (query, doc) 批量打分。"""
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HF reranker model is not initialized")

        docs = [c.content for c in candidates]
        device = self._model.device
        scores = []

        with torch.inference_mode():
            for i in range(0, len(docs), self.batch_size):
                batch_docs = docs[i:i + self.batch_size]
                encoded = self._tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                logits = outputs.logits

                if logits.dim() == 1:
                    batch_scores = logits
                elif logits.size(-1) == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    # 多分类头场景，取第一列作为相关性分数
                    batch_scores = logits[:, 0]

                scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores

    def _get_model(self):
        if self._model is not None:
            return self._model

        resolved_device = self._resolve_device(self.device)

        if self._prefer_hf_sequence_classifier():
            try:
                self._load_hf_sequence_classifier(resolved_device)
                return self._model
            except UnreliableCheckpointError as e:
                logger.warning(
                    f"Model '{self.model_name}' has unreliable scoring head: {e}. "
                    f"Switching to safe fallback model '{self.SAFE_FALLBACK_MODEL}'."
                )
                self.model_name = self.SAFE_FALLBACK_MODEL
                self._engine = "cross_encoder"
            except Exception as e:
                logger.warning(
                    "HF sequence-classifier reranker load failed for "
                    f"'{self.model_name}' on {resolved_device}: {e}. "
                    "Falling back to CrossEncoder path."
                )

                if resolved_device.startswith("cuda"):
                    try:
                        self._load_hf_sequence_classifier("cpu")
                        return self._model
                    except Exception as cpu_e:
                        logger.warning(
                            "HF reranker CPU fallback also failed for "
                            f"'{self.model_name}': {cpu_e}"
                        )

        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=resolved_device)
            self._engine = "cross_encoder"
            logger.debug(f"Loaded local reranker: {self.model_name} on {resolved_device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            if resolved_device.startswith("cuda"):
                logger.warning(
                    "Local CrossEncoder load failed on CUDA, retrying on CPU: "
                    f"{e}"
                )
                from sentence_transformers.cross_encoder import CrossEncoder
                self._model = CrossEncoder(self.model_name, device="cpu")
                self._engine = "cross_encoder"
            else:
                raise

        return self._model

    def rerank(self, query: str, candidates: list, top_n: int) -> list:
        if not candidates:
            return candidates

        with self._infer_lock:
            try:
                model = self._get_model()
                if self._engine == "hf_sequence_classifier":
                    scores = self._predict_hf_scores(query, candidates)
                else:
                    pairs = [(query, c.content) for c in candidates]
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
            logger.debug(f"Cohere reranker initialized (model={self.model})")
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
