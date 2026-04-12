"""
证据收集器（并行检索 + Rerank 版本）

流程：
  Round 0：快速自检 → 判断是否需要检索
  Round 1：内部检索 + 外部检索【并行执行，按各自 enabled 配置】
  Rerank ：若 rerank.enabled，对混合候选集统一重排序；否则各自独立返回
  最终：汇总为 EvidencePackage
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.evidence.internal_search import InternalSearchExecutor
from src.evidence.reranker import BaseReranker, create_reranker
from src.evidence.self_checker import SelfChecker
from src.models import (
    EvidencePackage,
    ExternalEvidence,
    InternalContext,
    NextAction,
    QAPair,
    RankedResult,
    SelfCheckResult,
)
from src.tools.external_search import ExternalSearchRunner

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """
    证据收集器

    Round 0 先做自检，若需要检索则内部检索与外部检索并行执行，
    之后可选地用 Reranker 对混合结果统一重排，最终汇总为 EvidencePackage。
    """

    def __init__(self):
        # 按需调用的模块
        self.self_checker = SelfChecker()
        self.internal_searcher = InternalSearchExecutor()
        self.external_runner = ExternalSearchRunner()

        self.config = get_config()

        # 懒加载 reranker（首次调用时初始化，避免启动时加载大模型）
        self._reranker: Optional[BaseReranker] = None
        self._reranker_initialized = False

    def _get_reranker(self) -> Optional[BaseReranker]:
        """懒加载 Reranker 实例"""
        if not self._reranker_initialized:
            self._reranker = create_reranker(self.config.rerank)
            self._reranker_initialized = True
        return self._reranker

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def collect(self, qa_pair: QAPair) -> EvidencePackage:
        """
        执行证据收集

        Returns:
            EvidencePackage（兼容旧版 dict.get() 接口）
        """
        logger.info(f"[EvidenceCollector] Starting collection for {qa_pair.id}")

        self_check_rounds: List[SelfCheckResult] = []
        internal_context: Optional[InternalContext] = None
        external_evidence: List[ExternalEvidence] = []
        ranked_results: List[RankedResult] = []

        # ── Round 0：快速自检 ─────────────────────────────────────────────────
        round0 = self.self_checker.check(
            question=qa_pair.question,
            answer=qa_pair.answer,
        )
        self_check_rounds.append(round0)
        logger.info(
            f"[Round 0] confidence={round0.confidence:.2f} "
            f"action={round0.next_action.value} "
            f"missing='{round0.missing_slots[:80]}'"
        )

        if round0.next_action == NextAction.SEARCH:
            # ── Round 1：内部检索 + 外部检索【并行】─────────────────────────
            internal_context, external_evidence = self._run_parallel_search(
                qa_pair, round0.missing_slots
            )

            # ── Rerank：混合候选集统一重排序 ─────────────────────────────────
            ranked_results = self._run_rerank(
                question=qa_pair.question,
                internal_context=internal_context,
                external_evidence=external_evidence,
            )

        # ── 判断证据是否仍然不足 ──────────────────────────────────────────────
        evidence_insufficient = self._check_insufficient(
            round0, internal_context, external_evidence, ranked_results
        )

        # ── 汇总证据包 ────────────────────────────────────────────────────────
        evidence_summary = self._generate_evidence_summary(
            internal_context, external_evidence, ranked_results
        )

        package = EvidencePackage(
            qa_id=qa_pair.id,
            self_check_rounds=self_check_rounds,
            base_evidence={},  # 基础工具已移除，保留空字典兼容旧代码
            internal_context=internal_context,
            external_evidence=external_evidence,
            evidence_insufficient=evidence_insufficient,
            evidence_summary=evidence_summary,
            rounds_executed=len(self_check_rounds),
            ranked_results=ranked_results,
        )

        logger.info(
            f"[EvidenceCollector] Done for {qa_pair.id}: "
            f"rounds={package.rounds_executed} "
            f"internal_chunks={len(internal_context.chunks) if internal_context else 0} "
            f"external_items={len(external_evidence)} "
            f"ranked={len(ranked_results)} "
            f"insufficient={package.evidence_insufficient}"
        )
        return package

    # ── 并行检索 ──────────────────────────────────────────────────────────────

    def _run_parallel_search(
        self, qa_pair: QAPair, missing_slots: str
    ) -> tuple[Optional[InternalContext], List[ExternalEvidence]]:
        """
        内部检索与外部检索并行执行

        Returns:
            (internal_context, external_evidence) 元组
        """
        internal_context: Optional[InternalContext] = None
        external_evidence: List[ExternalEvidence] = []

        internal_enabled = self.config.internal_search.enabled
        external_enabled = self.config.external_search.enabled

        if not internal_enabled and not external_enabled:
            logger.debug("Both internal and external search disabled, skipping Round 1")
            return internal_context, external_evidence

        futures: Dict = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            if internal_enabled:
                futures["internal"] = executor.submit(
                    self._run_internal_search, qa_pair, missing_slots
                )
            if external_enabled:
                futures["external"] = executor.submit(
                    self._run_external_search, missing_slots
                )

            if "internal" in futures:
                try:
                    internal_context = futures["internal"].result()
                    logger.info(
                        f"[Round 1/internal] {len(internal_context.chunks)} chunks retrieved"
                    )
                except Exception as e:
                    logger.error(f"Internal search thread failed: {e}")
                    internal_context = InternalContext(chunks=[])

            if "external" in futures:
                try:
                    external_evidence = futures["external"].result()
                    logger.info(
                        f"[Round 1/external] {len(external_evidence)} items fetched"
                    )
                except Exception as e:
                    logger.error(f"External search thread failed: {e}")
                    external_evidence = []

        return internal_context, external_evidence

    def _run_internal_search(
        self, qa_pair: QAPair, missing_slots: str
    ) -> InternalContext:
        """执行内部检索"""
        try:
            return self.internal_searcher.search(
                question=qa_pair.question,
                answer=qa_pair.answer,
                missing_slots=missing_slots,
            )
        except Exception as e:
            logger.error(f"Internal search failed: {e}")
            return InternalContext(chunks=[])

    def _run_external_search(self, missing_slots: str) -> List[ExternalEvidence]:
        """执行外部检索"""
        try:
            return self.external_runner.fetch(missing_slots)
        except Exception as e:
            logger.error(f"External search failed: {e}")
            return []

    # ── Rerank ────────────────────────────────────────────────────────────────

    def _run_rerank(
        self,
        question: str,
        internal_context: Optional[InternalContext],
        external_evidence: List[ExternalEvidence],
    ) -> List[RankedResult]:
        """
        对内部 chunk 和外部 evidence 混合候选集进行 Rerank

        若 reranker 未启用，返回空列表（调用方使用原始 internal_context / external_evidence）。
        """
        reranker = self._get_reranker()
        if reranker is None:
            return []

        candidates: List[RankedResult] = []

        # 内部 chunk → RankedResult
        if internal_context:
            for chunk in internal_context.chunks:
                candidates.append(RankedResult(
                    source="internal",
                    content=chunk.content,
                    relevance_score=chunk.relevance_score,
                    chunk=chunk,
                ))

        # 外部 evidence → RankedResult
        for ev in external_evidence:
            candidates.append(RankedResult(
                source="external",
                content=ev.snippet,
                relevance_score=ev.confidence,
                evidence=ev,
            ))

        if not candidates:
            logger.debug("No candidates for reranking")
            return []

        top_n = self.config.rerank.top_n
        try:
            ranked = reranker.rerank(question, candidates, top_n)
            logger.info(
                f"[Rerank] {len(candidates)} candidates → top {len(ranked)} results "
                f"(internal={sum(1 for r in ranked if r.source == 'internal')}, "
                f"external={sum(1 for r in ranked if r.source == 'external')})"
            )
            return ranked
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return []

    # ── 证据充足性判断 ────────────────────────────────────────────────────────

    def _check_insufficient(
        self,
        round0: SelfCheckResult,
        internal_context: Optional[InternalContext],
        external_evidence: List[ExternalEvidence],
        ranked_results: List[RankedResult],
    ) -> bool:
        """判断最终证据是否仍然不足"""
        if round0.next_action != NextAction.SEARCH:
            return False

        # 有 rerank 结果视为充足
        if ranked_results:
            return False

        # 无 rerank 时，有任何内部或外部结果即视为充足
        has_internal = bool(internal_context and internal_context.chunks)
        has_external = bool(external_evidence)
        return not has_internal and not has_external

    # ── 摘要生成 ──────────────────────────────────────────────────────────────

    def _generate_evidence_summary(
        self,
        internal_context: Optional[InternalContext],
        external_evidence: List[ExternalEvidence],
        ranked_results: List[RankedResult],
    ) -> str:
        """生成全量证据的文本摘要（供 LLM 评分使用）"""
        parts: List[str] = []

        if ranked_results:
            # 优先使用 Rerank 后的统一排序结果
            parts.append("═══ 检索证据（Rerank 混合排序）═══")
            for i, result in enumerate(ranked_results, 1):
                source_tag = "内部知识库" if result.source == "internal" else "外部检索"
                if result.source == "internal" and result.chunk:
                    ref = f"文档={result.chunk.doc_id}"
                elif result.source == "external" and result.evidence:
                    ref = f"{result.evidence.tool_name} · {result.evidence.source}"
                else:
                    ref = result.source
                parts.append(
                    f"[{i}] [{source_tag}] {ref} (相关度={result.relevance_score:.3f})\n"
                    f"  {result.content[:300]}"
                )
        else:
            # Reranker 未启用时，分别展示内部和外部结果
            if internal_context and internal_context.chunks:
                parts.append("\n═══ INTERNAL 证据（内部知识库）═══")
                parts.append(internal_context.to_summary_text())

            if external_evidence:
                parts.append("\n═══ EXTERNAL 证据（外部检索）═══")
                for ev in external_evidence:
                    parts.append(
                        f"[{ev.tool_name}] {ev.source} (置信度={ev.confidence:.2f})\n"
                        f"  {ev.snippet[:300]}"
                    )

        return "\n".join(parts) if parts else "未收集到有效证据"
