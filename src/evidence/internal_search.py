"""
内部检索执行器
BM25 + TF-IDF 向量混合召回 → RRF 融合排序 → 邻域扩展
索引格式：data/index/chunks.jsonl，每行一个片段：
  {"chunk_id": "...", "doc_id": "...", "content": "...", "chunk_index": 0, "metadata": {}}
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import get_config, InternalSearchConfig
from src.models import ChunkContext, InternalContext

logger = logging.getLogger(__name__)


# ── 内部数据结构 ──────────────────────────────────────────────────────────────

class _Chunk:
    """内部片段表示"""
    __slots__ = ("chunk_id", "doc_id", "content", "chunk_index", "metadata")

    def __init__(self, chunk_id: str, doc_id: str, content: str,
                 chunk_index: int, metadata: dict):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.chunk_index = chunk_index
        self.metadata = metadata


# ── RRF 融合 ─────────────────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion 单条得分"""
    return 1.0 / (k + rank + 1)


def _rrf_merge(
    bm25_ranked: List[str],
    vector_ranked: List[str],
) -> List[Tuple[str, float]]:
    """
    将 BM25 排名列表和向量排名列表融合为统一排序

    Returns:
        List of (chunk_id, rrf_score)，降序排列
    """
    scores: Dict[str, float] = {}
    for rank, cid in enumerate(bm25_ranked):
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
    for rank, cid in enumerate(vector_ranked):
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── 主类 ─────────────────────────────────────────────────────────────────────

class InternalSearchExecutor:
    """
    内部知识库检索执行器

    初始化时懒加载索引（首次搜索时加载），无索引时优雅降级。
    """

    def __init__(self, config: Optional[InternalSearchConfig] = None):
        self.config = config or get_config().internal_search
        self._chunks: Optional[List[_Chunk]] = None
        self._bm25 = None           # rank_bm25.BM25Okapi 实例
        self._tfidf = None          # sklearn TfidfVectorizer
        self._tfidf_matrix = None   # 向量矩阵
        self._chunk_id_index: Dict[str, int] = {}  # chunk_id → list 下标

    # ── 索引加载 ──────────────────────────────────────────────────────────────

    def _ensure_index(self) -> bool:
        """确保索引已加载，返回是否成功"""
        if self._chunks is not None:
            return True

        chunks_file = Path(self.config.index_dir) / "chunks.jsonl"
        if not chunks_file.exists():
            logger.info(
                f"Internal search index not found at {chunks_file}. "
                "Skipping internal retrieval."
            )
            self._chunks = []
            return False

        try:
            self._load_chunks(chunks_file)
            self._build_bm25()
            self._build_tfidf()
            logger.info(f"Loaded {len(self._chunks)} chunks from {chunks_file}")
            return len(self._chunks) > 0
        except Exception as e:
            logger.error(f"Failed to load internal index: {e}")
            self._chunks = []
            return False

    def _load_chunks(self, path: Path):
        """从 JSONL 加载片段"""
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    chunks.append(_Chunk(
                        chunk_id=str(d.get("chunk_id", f"chunk_{i}")),
                        doc_id=str(d.get("doc_id", "unknown")),
                        content=str(d.get("content", "")),
                        chunk_index=int(d.get("chunk_index", i)),
                        metadata=d.get("metadata", {}),
                    ))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed chunk at line {i}: {e}")
        self._chunks = chunks
        self._chunk_id_index = {c.chunk_id: i for i, c in enumerate(chunks)}

    def _build_bm25(self):
        """构建 BM25 索引（使用 rank_bm25 库）"""
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [c.content.lower().split() for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 search disabled. "
                           "Install with: pip install rank-bm25")

    def _build_tfidf(self):
        """构建 TF-IDF 向量索引（sklearn）"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = [c.content for c in self._chunks]
            self._tfidf = TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._tfidf.fit_transform(corpus)
        except ImportError:
            logger.warning("scikit-learn not installed. Vector search disabled. "
                           "Install with: pip install scikit-learn")

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def search(
        self,
        question: str,
        answer: str,
        missing_slots: str,
    ) -> InternalContext:
        """
        执行混合检索

        Args:
            question: 原始问题
            answer: 原始回答
            missing_slots: 自检识别出的缺失信息描述（自然语言段落）

        Returns:
            InternalContext（无索引时返回空结果）
        """
        if not self.config.enabled or not self._ensure_index() or not self._chunks:
            return InternalContext(chunks=[])

        # 构造查询：问题 + 缺失槽位
        query = self._build_query(question, missing_slots)

        bm25_ranked = self._bm25_search(query)
        vector_ranked = self._vector_search(query)

        # RRF 融合
        merged = _rrf_merge(bm25_ranked, vector_ranked)

        # 取 TopK 进行精排（此处用 RRF 得分作为相关度代理）
        top_ids = [cid for cid, _ in merged[: self.config.rerank_top_n * 3]]
        reranked = self._simple_rerank(query, top_ids, merged)

        # 邻域扩展
        final_chunks = self._neighborhood_expand(reranked[: self.config.rerank_top_n])

        return InternalContext(chunks=final_chunks)

    def _build_query(self, question: str, missing_slots: str) -> str:
        """构造检索查询字符串"""
        parts = [question]
        if missing_slots:
            parts.append(missing_slots)
        return " ".join(parts)

    # ── BM25 检索 ─────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str) -> List[str]:
        """BM25 检索，返回 chunk_id 列表（按得分降序）"""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_k = self.config.bm25_top_k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i].chunk_id for i in ranked_indices[:top_k]]

    # ── 向量检索 ──────────────────────────────────────────────────────────────

    def _vector_search(self, query: str) -> List[str]:
        """TF-IDF 向量检索，返回 chunk_id 列表（按余弦相似度降序）"""
        if self._tfidf is None or self._tfidf_matrix is None or not self._chunks:
            return []
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self._tfidf.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
            top_k = self.config.vector_top_k
            ranked_indices = np.argsort(sims)[::-1][:top_k]
            return [self._chunks[i].chunk_id for i in ranked_indices]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    # ── 精排 ─────────────────────────────────────────────────────────────────

    def _simple_rerank(
        self,
        query: str,
        candidate_ids: List[str],
        rrf_scores: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """简单精排：使用 RRF 得分（生产环境可替换为 cross-encoder）"""
        score_map = dict(rrf_scores)
        return [(cid, score_map.get(cid, 0.0)) for cid in candidate_ids]

    # ── 邻域扩展 ──────────────────────────────────────────────────────────────

    def _neighborhood_expand(
        self, reranked: List[Tuple[str, float]]
    ) -> List[ChunkContext]:
        """
        对每个命中片段，扩展前后各 window 个相邻片段（同一 doc_id 内）
        去重并保持原有相关度排序
        """
        window = self.config.neighborhood_window
        seen_ids: set = set()
        result: List[ChunkContext] = []

        # 先加入命中片段本身，再追加邻居
        for cid, score in reranked:
            idx = self._chunk_id_index.get(cid)
            if idx is None or cid in seen_ids:
                continue
            center_chunk = self._chunks[idx]
            neighbors = self._get_neighbors(idx, center_chunk.doc_id, window)

            for neighbor_idx, is_center in neighbors:
                nc = self._chunks[neighbor_idx]
                if nc.chunk_id in seen_ids:
                    continue
                seen_ids.add(nc.chunk_id)
                result.append(ChunkContext(
                    chunk_id=nc.chunk_id,
                    doc_id=nc.doc_id,
                    content=nc.content,
                    chunk_index=nc.chunk_index,
                    relevance_score=score if is_center else score * 0.7,
                    metadata=nc.metadata,
                ))

        return result

    def _get_neighbors(
        self, center_idx: int, doc_id: str, window: int
    ) -> List[Tuple[int, bool]]:
        """获取同一文档内的相邻片段下标列表"""
        total = len(self._chunks)
        result = []
        for offset in range(-window, window + 1):
            ni = center_idx + offset
            if 0 <= ni < total and self._chunks[ni].doc_id == doc_id:
                result.append((ni, offset == 0))
        return result

