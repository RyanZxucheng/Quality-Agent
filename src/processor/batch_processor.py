"""
Batch processor
Process JSON/CSV/JSONL files through the evaluation pipeline:
Evidence Collection → LLM Scoring → Decision
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.group import Group
from rich.text import Text

from src.models import (
    QAPair,
    EvaluationResult,
    EvaluationSummary,
    Conclusion,
    EvidencePackage,
    NextAction,
)
from src.evidence import EvidenceCollector
from src.scoring.llm_engine import LLMScoringEngine
from src.decision import DecisionEngine
from src.config import get_config
from src.utils.file_utils import safe_read_json, safe_read_jsonl, safe_read_csv, ensure_dir
from src.utils.logging_setup import get_console

logger = logging.getLogger(__name__)

# Number of recent per-QA result lines to keep visible during processing
_MAX_RESULT_LINES = 6


def _make_progress(console):
    """Create a consistently styled Progress instance"""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def _format_result_line(
    qa_pair: QAPair,
    evaluation: EvaluationResult,
    evidence: Any,
) -> Text:
    """
    Format a single QA result line for live display.

    Returns a rich Text with colorized summary, e.g.:
      qa_0: self-check passed → score 82 → retained ✓
    """
    # Determine status icon and color
    is_retained = evaluation.conclusion == Conclusion.RETAIN
    icon = "✓" if is_retained else "✗"
    color = "green" if is_retained else "red"

    # Extract brief action description
    action_text = ""
    if evidence and isinstance(evidence, EvidencePackage):
        if evidence.self_check_rounds:
            round0 = evidence.self_check_rounds[0]
            if round0.next_action == NextAction.PROCEED:
                action_text = "self-check passed"
            elif round0.next_action == NextAction.SEARCH:
                num_internal = len(evidence.internal_context.chunks) if evidence.internal_context else 0
                num_external = len(evidence.external_evidence)
                if num_internal + num_external > 0:
                    total_sources = num_internal + num_external
                    action_text = f"searched {total_sources} source{'s' if total_sources > 1 else ''}"
                elif evidence.evidence_insufficient:
                    action_text = "evidence insufficient"
                else:
                    action_text = "self-check passed"
            else:
                action_text = round0.next_action.value.lower().replace("_", " ")

    score = evaluation.scores.total_score if evaluation.scores else 0

    # Build brief result label
    if is_retained:
        label = "retained"
    else:
        short_reason = ""
        if evaluation.reason:
            short_reason = evaluation.reason.split("(")[0].strip() if "(" in evaluation.reason else evaluation.reason
        label = f"discarded ({short_reason})" if short_reason else "discarded"

    line = Text()
    line.append(f"  {qa_pair.id}: ", style="bold")
    line.append(action_text, style="cyan")
    line.append(f" → score {score} → ", style="white")
    line.append(f"{label} ", style=f"bold {color}")
    line.append(icon, style=color)
    return line


def _format_error_line(qa_pair: QAPair, error: str) -> Text:
    """Format an error result line."""
    line = Text()
    line.append(f"  {qa_pair.id}: ", style="bold")
    line.append(f"failed ({error})", style="red")
    line.append(" → ", style="white")
    line.append("discarded ✗", style="bold red")
    return line


class BatchProcessor:
    """
    Batch processor
    Quality assessment pipeline using tool evidence + LLM scoring.
    Supports sequential and parallel batch processing.
    """

    def __init__(
        self,
        evidence_collector: Optional[EvidenceCollector] = None,
        llm_scoring_engine: Optional[LLMScoringEngine] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.config = get_config()
        self.evidence_collector = evidence_collector or EvidenceCollector()
        self.llm_scoring_engine = llm_scoring_engine or LLMScoringEngine()
        self.decision_engine = decision_engine or DecisionEngine()
        # Checkpoint write lock to prevent overlapping writes from background threads
        self._checkpoint_lock = threading.Lock()

    def process_file(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        max_retained: Optional[int] = None
    ) -> EvaluationSummary:
        """
        Process input files through the evaluation pipeline.

        Args:
            input_paths: input file paths (JSON/CSV/JSONL)
            output_dir: output directory
            checkpoint_interval: checkpoint save interval
            max_retained: stop early after retaining this many items

        Returns:
            EvaluationSummary
        """
        output_dir = output_dir or self.config.output_dir
        ensure_dir(output_dir)

        qa_pairs = self._load_qa_pairs(input_paths)
        logger.info(f"Loaded {len(qa_pairs)} QA items from {len(input_paths)} file(s)")

        if self.config.is_parallel:
            logger.info(f"Parallel mode: batch_size={self.config.batch_size}")
            self.evidence_collector.warm_up()
            return self._process_parallel(
                qa_pairs, output_dir, checkpoint_interval, max_retained
            )
        else:
            logger.info("Sequential mode: batch_size=1")
            return self._process_sequential(
                qa_pairs, output_dir, checkpoint_interval, max_retained
            )

    def _process_single(self, qa_pair: QAPair) -> Tuple[QAPair, EvaluationResult, Optional[Dict[str, Any]]]:
        """
        Process a single QA pair.

        Returns:
            (qa_pair, evaluation, output_dict) — output_dict is None on error
        """
        try:
            evidence = self.evidence_collector.collect(qa_pair)
            score_result = self.llm_scoring_engine.score(qa_pair, evidence)
            evidence_pkg = evidence if isinstance(evidence, EvidencePackage) else None
            evaluation = self.decision_engine.decide(score_result, evidence_pkg)
            evaluation.evidence = evidence

            output = self._format_output(qa_pair, evidence, evaluation)
            return qa_pair, evaluation, output
        except Exception as e:
            logger.error(f"Processing failed {qa_pair.id}: {e}")
            evaluation = self._create_error_result(qa_pair, str(e))
            return qa_pair, evaluation, None

    def _process_sequential(
        self,
        qa_pairs: List[QAPair],
        output_dir: str,
        checkpoint_interval: int,
        max_retained: Optional[int],
    ) -> EvaluationSummary:
        """Sequential processing mode with live display."""
        results: List[EvaluationResult] = []
        retained_data: List[Dict[str, Any]] = []
        discarded_data: List[Dict[str, Any]] = []
        result_lines: List[Text] = []
        console = get_console()

        progress = _make_progress(console)
        task = progress.add_task("Processing...", total=len(qa_pairs))

        with Live(
            Group(progress, Text("\n".join("" for _ in range(_MAX_RESULT_LINES)))),
            console=console,
            refresh_per_second=4,
            vertical_overflow="visible",
        ) as live:
            progress.start()
            for i, qa_pair in enumerate(qa_pairs):
                # Update progress bar description
                progress.update(task, description=f"[cyan]{qa_pair.id}[/]")

                _, evaluation, output = self._process_single(qa_pair)
                results.append(evaluation)

                if output is not None:
                    if evaluation.conclusion == Conclusion.RETAIN:
                        retained_data.append(output)
                    else:
                        discarded_data.append(output)

                if max_retained is not None and len(retained_data) >= max_retained:
                    logger.info(f"Reached max retained limit ({max_retained}), stopping early")
                    break

                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(output_dir, results, retained_data, discarded_data)

                # Build result line
                line = _format_result_line(qa_pair, evaluation, evaluation.evidence)
                result_lines.append(line)
                if len(result_lines) > _MAX_RESULT_LINES:
                    result_lines.pop(0)

                progress.update(task, advance=1)

                # Refresh the live display
                group_elements = [progress]
                group_elements.append(Text())  # spacing
                group_elements.extend(result_lines)
                live.update(Group(*group_elements))

            progress.stop()

        return self._save_results(output_dir, results, retained_data, discarded_data)

    def _process_parallel(
        self,
        qa_pairs: List[QAPair],
        output_dir: str,
        checkpoint_interval: int,
        max_retained: Optional[int],
    ) -> EvaluationSummary:
        """Parallel processing mode — streaming batch submission.

        batch_size = number of items concurrently in-flight.
        Always keeps at most batch_size tasks running: submits the first batch,
        then replenishes one-by-one as tasks complete.
        """
        results: List[EvaluationResult] = []
        retained_data: List[Dict[str, Any]] = []
        discarded_data: List[Dict[str, Any]] = []
        result_lines: List[Text] = []
        lock = threading.Lock()

        batch_size = self.config.batch_size
        total = len(qa_pairs)
        stopped = False

        console = get_console()
        progress = _make_progress(console)
        task = progress.add_task("Processing...", total=total)

        qa_iter = iter(enumerate(qa_pairs))
        pending_futures: Dict[Any, int] = {}

        def _submit_next() -> bool:
            """Submit the next task, returns whether a task was submitted."""
            nonlocal stopped
            if stopped:
                return False
            try:
                idx, qa = next(qa_iter)
                future = executor.submit(self._process_single, qa)
                pending_futures[future] = idx
                return True
            except StopIteration:
                return False

        def _on_result(qa_pair, evaluation, output):
            """Thread-safe result accumulation."""
            with lock:
                results.append(evaluation)
                if output is not None:
                    if evaluation.conclusion == Conclusion.RETAIN:
                        retained_data.append(output)
                    else:
                        discarded_data.append(output)

                line = _format_result_line(qa_pair, evaluation, evaluation.evidence)
                result_lines.append(line)
                if len(result_lines) > _MAX_RESULT_LINES:
                    result_lines.pop(0)

        def _build_group():
            with lock:
                group_elements = [progress]
                if result_lines:
                    group_elements.append(Text())
                    group_elements.extend(result_lines)
                return Group(*group_elements)

        with Live(
            Group(progress),
            console=console,
            refresh_per_second=4,
            vertical_overflow="visible",
        ) as live:
            progress.start()

            try:
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Submit initial batch
                    for _ in range(batch_size):
                        if not _submit_next():
                            break

                    while pending_futures:
                        future = next(as_completed(pending_futures))
                        idx = pending_futures.pop(future)

                        try:
                            qa_pair, evaluation, output = future.result()
                        except Exception as e:
                            qa_pair = qa_pairs[idx]
                            logger.error(f"Future failed {qa_pair.id}: {e}")
                            evaluation = self._create_error_result(qa_pair, str(e))
                            output = None
                            error_line = _format_error_line(qa_pair, str(e))
                            result_lines.append(error_line)
                            if len(result_lines) > _MAX_RESULT_LINES:
                                result_lines.pop(0)

                        _on_result(qa_pair, evaluation, output)
                        progress.update(task, advance=1)

                        # Checkpoint
                        if len(results) % checkpoint_interval == 0:
                            self._save_checkpoint(output_dir, results, retained_data, discarded_data)

                        # Early stop
                        if max_retained is not None and len(retained_data) >= max_retained:
                            logger.info(f"Reached max retained limit ({max_retained}), stopping early")
                            stopped = True
                            for f in list(pending_futures.keys()):
                                if not f.done():
                                    f.cancel()
                            pending_futures.clear()
                            break

                        # Replenish
                        _submit_next()

                        # Update display
                        live.update(_build_group())

            finally:
                progress.stop()

        return self._save_results(output_dir, results, retained_data, discarded_data)

    def _load_qa_pairs(self, input_paths: List[str]) -> List[QAPair]:
        """Load QA pairs from multiple files."""
        all_items = []
        for input_path in input_paths:
            path = Path(input_path)
            suffix = path.suffix.lower()

            try:
                if suffix == ".json":
                    items = safe_read_json(input_path)
                elif suffix == ".csv":
                    items = safe_read_csv(input_path)
                elif suffix == ".jsonl":
                    items = safe_read_jsonl(input_path)
                else:
                    raise ValueError(f"Unsupported file format: {suffix}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Input file not found: {input_path}")

            all_items.extend(items)
            logger.info(f"Loaded {len(items)} items from {input_path}")

        qa_pairs = []
        for i, item in enumerate(all_items):
            try:
                qa_pairs.append(self._create_qa_pair(item, i))
            except ValueError as e:
                logger.warning(f"Skipping invalid QA entry (index {i}): {e}")

        return qa_pairs


    def _create_qa_pair(self, item: Dict, index: int) -> QAPair:
        """Create a QAPair object from a raw dictionary."""
        if "messages" in item and isinstance(item["messages"], list):
            question = ""
            answer = ""
            for msg in item["messages"]:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user" and not question:
                        question = content
                    elif role == "assistant" and not answer:
                        answer = content
        else:
            question = item.get("question") or item.get("Question") or item.get("q", "")
            answer = item.get("answer") or item.get("Answer") or item.get("a", "")

        qa_id = item.get("id") or item.get("ID") or f"qa_{index}"

        excluded_keys = {"id", "ID", "question", "Question", "q", "answer", "Answer", "a", "messages"}
        metadata = {k: v for k, v in item.items() if k not in excluded_keys}

        raw_data = dict(item)

        return QAPair(
            id=str(qa_id),
            question=str(question),
            answer=str(answer),
            metadata=metadata,
            raw_data=raw_data
        )

    def _format_output(
        self,
        qa_pair: QAPair,
        evidence: Any,
        evaluation: EvaluationResult
    ) -> Dict[str, Any]:
        """Format output — preserves input format without adding extra fields."""
        return qa_pair.raw_data

    def _create_error_result(self, qa_pair: QAPair, error: str) -> EvaluationResult:
        """Create an error result."""
        return EvaluationResult.create_error_result(qa_pair, error, Conclusion.DISCARD)

    def _save_checkpoint(
        self,
        output_dir: str,
        results: List[EvaluationResult],
        retained: List[Dict],
        discarded: List[Dict]
    ):
        """Save checkpoint (async, background thread, non-blocking)."""
        threading.Thread(
            target=self._save_checkpoint_sync,
            args=(output_dir, len(results), len(retained), len(discarded)),
            daemon=True,
        ).start()

    def _save_checkpoint_sync(
        self,
        output_dir: str,
        processed_count: int,
        retained_count: int,
        discarded_count: int,
    ):
        """Synchronous checkpoint write (runs in background thread)."""
        if not self._checkpoint_lock.acquire(blocking=False):
            logger.debug("Checkpoint write already in progress, skipping")
            return
        try:
            checkpoint_dir = Path(output_dir) / "checkpoint"
            checkpoint_dir.mkdir(exist_ok=True)
            with open(checkpoint_dir / "checkpoint.json", "w", encoding="utf-8") as f:
                json.dump({
                    "processed_count": processed_count,
                    "retained_count": retained_count,
                    "discarded_count": discarded_count,
                }, f, ensure_ascii=False, indent=2)
            logger.debug(f"Checkpoint saved: {processed_count} items processed")
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
        finally:
            self._checkpoint_lock.release()

    def _save_results(
        self,
        output_dir: str,
        results: List[EvaluationResult],
        retained: List[Dict],
        discarded: List[Dict]
    ) -> EvaluationSummary:
        """Save final results."""
        output_path = Path(output_dir)

        if retained:
            cleaned_dir = output_path / "cleaned_data"
            ensure_dir(str(cleaned_dir))
            with open(cleaned_dir / "retained_qa.json", "w", encoding="utf-8") as f:
                json.dump(retained, f, ensure_ascii=False, indent=2)

        if discarded:
            rejected_dir = output_path / "rejected"
            ensure_dir(str(rejected_dir))
            with open(rejected_dir / "discarded_qa.json", "w", encoding="utf-8") as f:
                json.dump(discarded, f, ensure_ascii=False, indent=2)

        from src.report import ReportGenerator
        report_gen = ReportGenerator()
        report = report_gen.generate(results)
        report_gen.save(report, str(output_path / "reports"))

        summary = report.summary
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_processed": summary.total_processed,
                "retained": summary.retained,
                "discarded": summary.discarded,
                "retention_rate": summary.retention_rate,
                "average_score": summary.average_score,
                "dimension_averages": summary.dimension_averages
            }, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Results saved: {summary.retained} retained, "
            f"{summary.discarded} discarded "
            f"(retention rate: {summary.retention_rate:.1%})"
        )

        return summary
