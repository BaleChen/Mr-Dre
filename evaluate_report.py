"""
Entry point for evaluating generated reports.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

from dr_eval.checklist_eval import evaluate_checklist, aggregate_checklist_metrics
from dr_eval.citation_eval import evaluate_citations, aggregate_citation_metrics
from dr_eval.rubric_eval import evaluate_rubric, aggregate_rubric_metrics
from dr_eval.eval_prompts import (
    CHECKLIST_EVAL_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CONTENT_SUMMARIZATION_SYSTEM_PROMPT,
    RUBRIC_EVAL_SYSTEM_PROMPT,
    SUPPORTED_JUDGE_SYSTEM_PROMPT,
)
from engine.oai import GPT
from utils import load_jsonl, save_jsonl, load_dataset, append_jsonl, resolve_dataset_path

DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_SUMMARIZATION_MODEL = "gpt-4.1-nano"


def _load_existing_metrics(metrics_path: Path, ignore_existing: bool) -> Tuple[int, Set[int], List[Dict[str, Any]]]:
    """
    Load previously computed sample-level metrics along with their IDs.
    When ignore_existing is True, treat as if no prior metrics exist.
    """
    if ignore_existing:
        return 0, set(), []

    if metrics_path.exists():
        try:
            metrics = load_jsonl(str(metrics_path))
            ids = {metric.get("id") for metric in metrics if metric.get("id") is not None}
            return len(metrics), ids, metrics
        except Exception as e:
            logger.warning(f"Could not load existing metrics from {metrics_path}: {e}")
    return 0, set(), []


def _upsert_agg_entry(agg_metrics_file: Path, entry: Dict[str, Any]) -> None:
    """
    Write a dataset-level aggregate entry, replacing any earlier entry for the same eval_type
    so that re-running an evaluation does not leave stale duplicates behind.
    """
    entries = load_jsonl(str(agg_metrics_file)) if agg_metrics_file.exists() else []
    entries = [e for e in entries if e.get("eval_type") != entry["eval_type"]]
    entries.append(entry)
    save_jsonl(entries, str(agg_metrics_file))


def _postprocess_dr_tulu(reports: List[Dict[str, Any]]) -> None:
    """
    DR Tulu reports cite snippet IDs instead of URLs, so the citations are resolved
    through the snippet map before evaluation. Reports without a snippet map are left untouched.
    """
    if not any((r.get("full_response") or {}).get("snippet_map") for r in reports):
        return
    try:
        from engine.dr_tulu import parse_and_format_citations
    except ImportError:
        logger.warning(
            "Reports carry a DR Tulu snippet_map, but engine/dr_tulu.py is not available; "
            "citations are evaluated as they appear in the report text."
        )
        return

    for report in tqdm(reports, desc="DR Tulu pre-eval processing"):
        full_response = report["full_response"]
        if full_response.get("snippet_map"):
            full_response["report"] = parse_and_format_citations(
                full_response["report"], full_response["snippet_map"]
            )


def evaluate_reports(
    report_file: str,
    dataset_file: str,
    output_dir: str = None,
    eval_mode: str = "all",
    evaluator_model_name: str = DEFAULT_JUDGE_MODEL,
    claim_extraction_model_name: str = DEFAULT_JUDGE_MODEL,
    supported_judge_model_name: str = DEFAULT_JUDGE_MODEL,
    webpage_summarization_model_name: str = DEFAULT_SUMMARIZATION_MODEL,
    rubric_judge_model_name: str = DEFAULT_JUDGE_MODEL,
    ignore_existing: bool = False,
    num_questions: int = -1,
    turn: Optional[int] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate generated reports.

    Args:
        report_file: Path to generated report JSONL file
        dataset_file: Dataset name (e.g., "ResearcherBench") or path to a dataset JSONL file
        output_dir: Directory to save evaluation results (defaults to the report file's directory)
        eval_mode: Evaluation mode - "checklist", "citation", "rubric", or "all"
        evaluator_model_name: Model for checklist evaluation
        claim_extraction_model_name: Model for claim extraction
        supported_judge_model_name: Model for judging claim support
        webpage_summarization_model_name: Model for summarizing webpage content (optional)
        rubric_judge_model_name: Model for rubric evaluation
        ignore_existing: If True, re-evaluate all reports even when prior metrics exist
        num_questions: Number of questions to evaluate (default: -1 for all).
                       Uses the same seed (42) as call_dra.py to select the same subset.
        turn: Turn index of the reports. If None, it is parsed from the "turn{i}" suffix of the filename.
        **kwargs: Additional arguments passed to evaluation functions

    Returns:
        Dict mapping each evaluated mode to {"aggregate": Dict, "num_evaluated": int, "failed_ids": List}
    """
    logger.info(f"Evaluating {Path(report_file).name} ({eval_mode} mode)")

    # Load data
    dataset_file = resolve_dataset_path(dataset_file)
    reports = load_jsonl(report_file)
    dataset = load_dataset(dataset_file)
    logger.info(f"Loaded {len(reports)} reports, {len(dataset)} dataset items")

    unknown_ids = [r.get("id") for r in reports if r.get("id") not in dataset]
    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} report IDs are not in {dataset_file} (e.g., {unknown_ids[:5]}). "
            "Report IDs must match the dataset IDs, including their type (int vs. str)."
        )

    # Filter to sampled subset if num_questions is specified
    # Uses the same logic as call_dra.py to ensure the same subset is selected
    sampled_ids = None
    if num_questions > 0:
        # Load dataset as list and shuffle with same seed as call_dra.py
        dataset_list = load_jsonl(dataset_file)
        random.seed(42)
        random.shuffle(dataset_list)
        sampled_questions = dataset_list[:num_questions]
        sampled_ids = {q["id"] for q in sampled_questions}
        
        # Filter reports to only include sampled questions
        original_report_count = len(reports)
        reports = [r for r in reports if r.get("id") in sampled_ids]
        
        # Filter dataset dict to only include sampled questions
        dataset = {k: v for k, v in dataset.items() if k in sampled_ids}
        
        logger.info(f"Sampled {num_questions} questions (seed=42): {len(reports)}/{original_report_count} reports to evaluate")
        if len(reports) != num_questions:
            logger.warning(f"Warning: {len(reports)} reports != {num_questions} questions. This may cause evaluation issues.")

    report_path = Path(report_file)
    report_filename = report_path.stem

    # Output directory defaults to the same directory as the report file
    # unless user specifies a different one
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = report_path.parent

    # Extract turn from filename pattern *_turn{i}
    if turn is None:
        if "turn" in report_filename:
            turn_suffix = report_filename.split("turn")[-1]
            if turn_suffix.isdigit():
                turn = int(turn_suffix)
            else:
                raise ValueError(f"Invalid turn suffix: {turn_suffix}. Please check the report filename format.")
        else:
            raise ValueError(f"Report filename must contain 'turn{{i}}' pattern (e.g., 'gen_turn1.jsonl'), or pass the turn explicitly. Got: {report_filename}")

    agg_metrics_file = output_path / f"agg_metrics_turn{turn}.jsonl"
    logger.info(f"Output: {output_path} (turn {turn})")
    
    # Run evaluations based on evaluation mode
    if ignore_existing:
        logger.info("Re-evaluating all reports (ignore_existing=True)")

    # DR Tulu specific postprocessing
    _postprocess_dr_tulu(reports)

    # Each mode: (judge models, per-report evaluation function, aggregation function)
    modes = {
        "checklist": (
            lambda: {"evaluator_model": GPT(evaluator_model_name, system_prompt=CHECKLIST_EVAL_SYSTEM_PROMPT)},
            lambda report, models: evaluate_checklist(
                report, dataset[report.get("id")], models["evaluator_model"], **kwargs
            ),
            aggregate_checklist_metrics,
        ),
        "citation": (
            lambda: {
                "claim_extraction_model": GPT(claim_extraction_model_name, system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT),
                "supported_judge_model": GPT(supported_judge_model_name, system_prompt=SUPPORTED_JUDGE_SYSTEM_PROMPT),
                "webpage_summarization_model": GPT(webpage_summarization_model_name, system_prompt=CONTENT_SUMMARIZATION_SYSTEM_PROMPT),
            },
            lambda report, models: evaluate_citations(
                report,
                models["claim_extraction_model"],
                models["supported_judge_model"],
                webpage_summarization_model=models["webpage_summarization_model"],
                **kwargs
            ),
            aggregate_citation_metrics,
        ),
        "rubric": (
            lambda: {"rubric_judge_model": GPT(rubric_judge_model_name, system_prompt=RUBRIC_EVAL_SYSTEM_PROMPT)},
            lambda report, models: evaluate_rubric(
                report, dataset.get(report.get("id"), {}), models["rubric_judge_model"],
                rubric_dimensions=["presentation"], **kwargs
            ),
            aggregate_rubric_metrics,
        ),
    }

    summary: Dict[str, Dict[str, Any]] = {}
    for mode, (build_models, evaluate_one, aggregate) in modes.items():
        if eval_mode not in (mode, "all"):
            continue

        mode_output = output_path / f"{mode}_turn{turn}.jsonl"
        _, existing_ids, existing_records = _load_existing_metrics(mode_output, ignore_existing)
        if ignore_existing and mode_output.exists():
            # Start the per-report file afresh so that stale records are not mixed with new ones
            mode_output.unlink()
        # Filter existing records to sampled subset if num_questions is specified
        if sampled_ids is not None:
            existing_records = [r for r in existing_records if r.get("id") in sampled_ids]
            existing_ids = {r.get("id") for r in existing_records}
        if existing_records:
            logger.info(f"Resuming {mode}: {len(existing_records)} existing")

        combined_metrics = list(existing_records)
        failed_ids = []
        models = build_models()
        for report in tqdm(reports, desc=f"{mode.capitalize()} eval", unit="report"):
            question_id = report.get("id")
            if question_id in existing_ids:
                continue
            try:
                metric = evaluate_one(report, models)
                existing_ids.add(question_id)
                combined_metrics.append(metric)
                append_jsonl(metric, str(mode_output))
            except Exception as e:
                failed_ids.append(question_id)
                logger.error(f"{mode.capitalize()} eval failed for Q{question_id}: {e}")
                continue

        # Aggregate and save the latest dataset-level metrics across all evaluations
        agg_entry = {
            "turn": turn,
            "eval_type": mode,
            **aggregate(combined_metrics),
        }
        _upsert_agg_entry(agg_metrics_file, agg_entry)
        summary[mode] = {
            "aggregate": agg_entry,
            "num_evaluated": len(combined_metrics),
            "failed_ids": failed_ids,
        }
        if failed_ids:
            logger.warning(
                f"{mode.capitalize()}: {len(failed_ids)}/{len(reports)} reports failed and are excluded "
                f"from the aggregate; re-run the same command to retry them. Failed IDs: {failed_ids}"
            )
        logger.info(f"{mode.capitalize()} done → {mode_output.name}")
        
    logger.info(f"✓ All evaluations completed → {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated reports")
    parser.add_argument("--report_file", type=str, required=True,
                        help="Path to generated report JSONL file")
    parser.add_argument("--dataset_file", "--dataset", dest="dataset_file", type=str, required=True,
                        help="Dataset name (e.g., ResearcherBench) or path to a dataset JSONL file")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save evaluation results (defaults to same directory as report file)")
    parser.add_argument("--eval_mode", "--mode", dest="eval_mode", type=str, default="all",
                        choices=["checklist", "citation", "rubric", "all"],
                        help="Evaluation mode")
    parser.add_argument("--evaluator_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Model for checklist evaluation")
    parser.add_argument("--claim_extraction_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Model for claim extraction")
    parser.add_argument("--supported_judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Model for judging claim support")
    parser.add_argument("--webpage_summarization_model", type=str, default=DEFAULT_SUMMARIZATION_MODEL,
                        help="Model for summarizing webpage content (optional, e.g., gpt-4.1-nano)")
    parser.add_argument("--rubric_judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Model for rubric evaluation")
    parser.add_argument("--ignore_existing", action="store_true",
                        help="Re-evaluate all reports even if metrics already exist")
    parser.add_argument("--num_questions", type=int, default=-1,
                        help="Number of questions to evaluate (default: -1 for all). "
                             "Uses the same seed (42) as call_dra.py to select the same subset.")
    
    args = parser.parse_args()
    
    evaluate_reports(
        report_file=args.report_file,
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        eval_mode=args.eval_mode,
        evaluator_model_name=args.evaluator_model,
        claim_extraction_model_name=args.claim_extraction_model,
        supported_judge_model_name=args.supported_judge_model,
        webpage_summarization_model_name=args.webpage_summarization_model,
        rubric_judge_model_name=args.rubric_judge_model,
        ignore_existing=args.ignore_existing,
        num_questions=args.num_questions
    )


if __name__ == "__main__":
    main()

