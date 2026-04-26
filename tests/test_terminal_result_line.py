import os
import sys
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import (  # noqa: E402
    ChunkContext,
    Conclusion,
    DimensionScore,
    EvidencePackage,
    EvaluationResult,
    ExternalEvidence,
    InternalContext,
    NextAction,
    QAPair,
    ScoreResult,
    SelfCheckResult,
)
from src.processor.batch_processor import _format_result_line  # noqa: E402


class TestTerminalResultLine(unittest.TestCase):
    def _qa_pair(self, qa_id="qa_1"):
        return QAPair(id=qa_id, question="Question?", answer="Answer.")

    def _score_result(self, qa_pair, total_score=80):
        return ScoreResult(
            qa_pair=qa_pair,
            dimensions=[
                DimensionScore(name="completeness", score=25, max_score=30),
                DimensionScore(name="accuracy", score=35, max_score=45),
                DimensionScore(name="professionalism", score=20, max_score=25),
            ],
            total_score=total_score,
        )

    def _evaluation(self, qa_pair, total_score=80, conclusion=Conclusion.RETAIN, reason="OK"):
        return EvaluationResult(
            qa_pair=qa_pair,
            scores=self._score_result(qa_pair, total_score),
            conclusion=conclusion,
            reason=reason,
        )

    def _evidence_package(
        self,
        qa_pair,
        next_action,
        internal_chunks=0,
        external_items=0,
        evidence_insufficient=False,
    ):
        chunks = [
            ChunkContext(
                chunk_id=f"chunk_{i}",
                doc_id="doc",
                content="content",
                chunk_index=i,
                relevance_score=0.9,
            )
            for i in range(internal_chunks)
        ]
        external = [
            ExternalEvidence(
                tool_name="tool",
                source="source",
                snippet="snippet",
            )
            for _ in range(external_items)
        ]
        return EvidencePackage(
            qa_id=qa_pair.id,
            self_check_rounds=[
                SelfCheckResult(
                    confidence=0.5,
                    blocking_issues=[],
                    missing_slots="missing evidence",
                    next_action=next_action,
                )
            ],
            base_evidence={},
            internal_context=InternalContext(chunks=chunks),
            external_evidence=external,
            evidence_insufficient=evidence_insufficient,
            evidence_summary="",
            rounds_executed=1,
        )

    def _style_for_text(self, line, text):
        start = line.plain.index(text)
        end = start + len(text)
        for span in line.spans:
            if span.start <= start and span.end >= end:
                return span.style
        return None

    def test_search_result_line_marks_self_check_failed_before_found_evidence(self):
        qa_pair = self._qa_pair("qa_1")
        evidence = self._evidence_package(
            qa_pair,
            NextAction.SEARCH,
            internal_chunks=2,
            external_items=3,
        )

        line = _format_result_line(qa_pair, self._evaluation(qa_pair, 71), evidence)

        self.assertEqual(
            line.plain,
            "  qa_1: self-check failed → found 5 evidence items → score 71 → retained ✓",
        )
        self.assertEqual(self._style_for_text(line, "self-check failed"), "yellow")
        self.assertEqual(self._style_for_text(line, "found 5 evidence items"), "cyan")

    def test_search_result_line_marks_no_evidence_found_red(self):
        qa_pair = self._qa_pair("qa_2")
        evidence = self._evidence_package(
            qa_pair,
            NextAction.SEARCH,
            evidence_insufficient=True,
        )

        line = _format_result_line(
            qa_pair,
            self._evaluation(
                qa_pair,
                total_score=80,
                conclusion=Conclusion.DISCARD,
                reason="Evidence insufficient, manual review required",
            ),
            evidence,
        )

        self.assertEqual(
            line.plain,
            "  qa_2: self-check failed → no evidence found → score 80 → discarded "
            "(Evidence insufficient, manual review required) ✗",
        )
        self.assertEqual(self._style_for_text(line, "self-check failed"), "yellow")
        self.assertEqual(self._style_for_text(line, "no evidence found"), "red")

    def test_proceed_result_line_keeps_self_check_passed_green(self):
        qa_pair = self._qa_pair("qa_0")
        evidence = self._evidence_package(qa_pair, NextAction.PROCEED)

        line = _format_result_line(qa_pair, self._evaluation(qa_pair, 82), evidence)

        self.assertEqual(line.plain, "  qa_0: self-check passed → score 82 → retained ✓")
        self.assertEqual(self._style_for_text(line, "self-check passed"), "green")


if __name__ == "__main__":
    unittest.main()
