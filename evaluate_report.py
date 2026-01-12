"""
Entry point for evaluating generated reports.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

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
from utils import load_jsonl, load_dataset, append_jsonl


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


def evaluate_reports(
    report_file: str,
    dataset_file: str,
    output_dir: str,
    eval_mode: str = "all",
    evaluator_model_name: str = None,
    claim_extraction_model_name: str = None,
    supported_judge_model_name: str = None,
    webpage_summarization_model_name: str = None,
    rubric_judge_model_name: str = None,
    ignore_existing: bool = False,
    num_questions: int = -1,
    **kwargs
):
    """
    Evaluate generated reports.

    Args:
        report_file: Path to generated report JSONL file
        dataset_file: Path to dataset JSONL file
        output_dir: Directory to save evaluation results
        eval_mode: Evaluation mode - "checklist", "citation", "rubric", or "all"
        evaluator_model_name: Model for checklist evaluation
        claim_extraction_model_name: Model for claim extraction
        supported_judge_model_name: Model for judging claim support
        webpage_summarization_model_name: Model for summarizing webpage content (optional)
        rubric_judge_model_name: Model for rubric evaluation
        ignore_existing: If True, re-evaluate all reports even when prior metrics exist
        num_questions: Number of questions to evaluate (default: -1 for all).
                       Uses the same seed (42) as call_dra.py to select the same subset.
        **kwargs: Additional arguments passed to evaluation functions
    """
    logger.info(f"Evaluating {Path(report_file).name} ({eval_mode} mode)")

    # Load data
    reports = load_jsonl(report_file)
    dataset = load_dataset(dataset_file)
    logger.info(f"Loaded {len(reports)} reports, {len(dataset)} dataset items")

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
    if "turn" in report_filename:
        turn_suffix = report_filename.split("turn")[-1]
        if turn_suffix.isdigit():
            turn = int(turn_suffix)
        else:
            raise ValueError(f"Invalid turn suffix: {turn_suffix}. Please check the report filename format.")
    else:
        raise ValueError(f"Report filename must contain 'turn{{i}}' pattern (e.g., 'gen_turn1.jsonl'). Got: {report_filename}")

    agg_metrics_file = output_path / f"agg_metrics_turn{turn}.jsonl"
    logger.info(f"Output: {output_path} (turn {turn})")
    
    # Run evaluations based on evaluation mode
    if ignore_existing:
        logger.info("Re-evaluating all reports (ignore_existing=True)")

    # DR Tulu specific postprocessing 
    if "dr-tulu" in report_file:
        from engine.dr_tulu import parse_and_format_citations
        
        for report in tqdm(reports, desc="DR Tulu pre-eval processing"):
            report_text = report["full_response"]["report"]
            snippet_map = report["full_response"]["snippet_map"]

            new_report_text = parse_and_format_citations(report_text, snippet_map)
            report["full_response"]["report"] = new_report_text

    if eval_mode in ["checklist", "all"]:
        checklist_output = output_path / f"checklist_turn{turn}.jsonl"
        _, existing_checklist_ids, existing_checklist_records = _load_existing_metrics(checklist_output, ignore_existing)
        # Filter existing records to sampled subset if num_questions is specified
        if sampled_ids is not None:
            existing_checklist_records = [r for r in existing_checklist_records if r.get("id") in sampled_ids]
            existing_checklist_ids = {r.get("id") for r in existing_checklist_records}
        if existing_checklist_records:
            logger.info(f"Resuming checklist: {len(existing_checklist_records)} existing")
        combined_checklist_metrics = list(existing_checklist_records)
        evaluator_model = GPT(evaluator_model_name, system_prompt=CHECKLIST_EVAL_SYSTEM_PROMPT)
        for report in tqdm(reports, desc="Checklist eval", unit="report"):
            question_id = report.get("id")
            assert question_id in dataset, f"Question ID {question_id} not found in dataset {dataset_file}"
            if question_id in existing_checklist_ids:
                continue
            try:
                metric = evaluate_checklist(
                    report,
                    dataset[question_id],
                    evaluator_model,
                    **kwargs
                )
                existing_checklist_ids.add(question_id)
                combined_checklist_metrics.append(metric)
                append_jsonl(metric, str(checklist_output))
            except Exception as e:
                logger.error(f"Checklist eval failed for Q{question_id}: {e}")
                continue
        
        # Aggregate and save the latest dataset-level metrics across all evaluations
        agg_metrics = aggregate_checklist_metrics(combined_checklist_metrics)
        agg_entry = {
            "turn": turn,
            "eval_type": "checklist",
            **agg_metrics,
        }
        append_jsonl(agg_entry, str(agg_metrics_file))
        logger.info(f"Checklist done → {checklist_output.name}")
    
    if eval_mode in ["citation", "all"]:
        citation_output = output_path / f"citation_turn{turn}.jsonl"
        _, existing_citation_ids, existing_citation_records = _load_existing_metrics(citation_output, ignore_existing)
        # Filter existing records to sampled subset if num_questions is specified
        if sampled_ids is not None:
            existing_citation_records = [r for r in existing_citation_records if r.get("id") in sampled_ids]
            existing_citation_ids = {r.get("id") for r in existing_citation_records}
        if existing_citation_records:
            logger.info(f"Resuming citation: {len(existing_citation_records)} existing")
        combined_citation_metrics = list(existing_citation_records)
        claim_extraction_model = GPT(claim_extraction_model_name, system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT)
        supported_judge_model = GPT(supported_judge_model_name, system_prompt=SUPPORTED_JUDGE_SYSTEM_PROMPT)
        webpage_summarization_model = GPT(webpage_summarization_model_name, system_prompt=CONTENT_SUMMARIZATION_SYSTEM_PROMPT)

        for report in tqdm(reports, desc="Citation eval", unit="report"):
            question_id = report.get("id")
            if question_id in existing_citation_ids:
                continue
            try:
                metric = evaluate_citations(
                    report,
                    claim_extraction_model,
                    supported_judge_model,
                    webpage_summarization_model=webpage_summarization_model,
                    **kwargs
                )
                existing_citation_ids.add(question_id)
                combined_citation_metrics.append(metric)
                append_jsonl(metric, str(citation_output))
            except Exception as e:
                logger.error(f"Citation eval failed for Q{question_id}: {e}")
                continue
        
        # Aggregate and save dataset-level metrics across all evaluations
        agg_metrics = aggregate_citation_metrics(combined_citation_metrics)
        agg_entry = {
            "turn": turn,
            "eval_type": "citation",
            **agg_metrics,
        }
        append_jsonl(agg_entry, str(agg_metrics_file))
        logger.info(f"Citation done → {citation_output.name}")
    
    if eval_mode in ["rubric", "all"]:
        rubric_output = output_path / f"rubric_turn{turn}.jsonl"
        _, existing_rubric_ids, existing_rubric_records = _load_existing_metrics(rubric_output, ignore_existing)
        # Filter existing records to sampled subset if num_questions is specified
        if sampled_ids is not None:
            existing_rubric_records = [r for r in existing_rubric_records if r.get("id") in sampled_ids]
            existing_rubric_ids = {r.get("id") for r in existing_rubric_records}
        if existing_rubric_records:
            logger.info(f"Resuming rubric: {len(existing_rubric_records)} existing")
        combined_rubric_metrics = list(existing_rubric_records)
        rubric_judge_model = GPT(rubric_judge_model_name, system_prompt=RUBRIC_EVAL_SYSTEM_PROMPT)
        for report in tqdm(reports, desc="Rubric eval", unit="report"):
            question_id = report.get("id")
            dataset_item = dataset.get(question_id, {})
            if question_id in existing_rubric_ids:
                continue
            
            try:
                metric = evaluate_rubric(
                    report,
                    dataset_item,
                    rubric_judge_model,
                    rubric_dimensions=["organization"],
                    **kwargs
                )
                existing_rubric_ids.add(question_id)
                combined_rubric_metrics.append(metric)
                append_jsonl(metric, str(rubric_output))
            except Exception as e:
                logger.error(f"Rubric eval failed for Q{question_id}: {e}")
                continue
        
        agg_metrics = aggregate_rubric_metrics(combined_rubric_metrics)
        agg_entry = {
            "turn": turn,
            "eval_type": "rubric",
            **agg_metrics,
        }
        append_jsonl(agg_entry, str(agg_metrics_file))
        logger.info(f"Rubric done → {rubric_output.name}")
        
    logger.info(f"✓ All evaluations completed → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated reports")
    parser.add_argument("--report_file", type=str, required=True,
                        help="Path to generated report JSONL file")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to dataset JSONL file")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save evaluation results (defaults to same directory as report file)")
    parser.add_argument("--eval_mode", "--mode", dest="eval_mode", type=str, default="all",
                        choices=["checklist", "citation", "rubric", "all"],
                        help="Evaluation mode")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4.1-mini",
                        help="Model for checklist evaluation")
    parser.add_argument("--claim_extraction_model", type=str, default="gpt-4.1-mini",
                        help="Model for claim extraction")
    parser.add_argument("--supported_judge_model", type=str, default="gpt-4.1-mini",
                        help="Model for judging claim support")
    parser.add_argument("--webpage_summarization_model", type=str, default="gpt-4.1-nano",
                        help="Model for summarizing webpage content (optional, e.g., gpt-4.1-nano)")
    parser.add_argument("--rubric_judge_model", type=str, default="gpt-4.1-mini",
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

