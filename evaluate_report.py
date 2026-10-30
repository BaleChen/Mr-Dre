"""
Evaluate generated reports in the output/<dataset>/<turn type>/<model>/gen_turn{i}.jsonl
folder layout of the paper's runs.

The newer `mrdre evaluate` command takes any reports file and output directory.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing mrdre modules: the disk caches read MCP_CACHE_DIR when they are imported
load_dotenv()

from mrdre.evaluate import EVAL_MODES, evaluate_reports, load_reports, summarize  # noqa: E402
from mrdre.utils import append_jsonl, load_dataset, load_jsonl, sample_questions, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


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
    parser.add_argument("--webpage_summarization_model", type=str, default="gpt-5.6-luna",
                        help="Model for summarizing webpage content (optional, e.g., gpt-5.6-luna)")
    parser.add_argument("--rubric_judge_model", type=str, default="gpt-4.1-mini",
                        help="Model for rubric evaluation")
    parser.add_argument("--ignore_existing", action="store_true",
                        help="Re-evaluate all reports even if metrics already exist")
    parser.add_argument("--num_questions", type=int, default=-1,
                        help="Number of questions to evaluate (default: -1 for all). "
                             "Uses the same seed (42) as generation to select the same subset.")
    args = parser.parse_args()
    setup_logging()

    # Extract turn from filename pattern *_turn{i}
    report_path = Path(args.report_file)
    turn_suffix = report_path.stem.split("turn")[-1]
    if "turn" not in report_path.stem or not turn_suffix.isdigit():
        raise ValueError(f"Report filename must contain 'turn{{i}}' pattern (e.g., 'gen_turn1.jsonl'). Got: {report_path.stem}")
    turn = int(turn_suffix)
    output_dir = Path(args.output_dir) if args.output_dir else report_path.parent

    dataset = load_dataset(args.dataset_file)
    sampled = sample_questions(load_jsonl(args.dataset_file), args.num_questions)
    reports = load_reports(args.report_file, dataset, {item["id"] for item in sampled})
    logger.info(f"Evaluating {report_path.name}: {len(reports)} reports ({args.eval_mode} mode), output: {output_dir}")

    records = evaluate_reports(
        reports,
        dataset,
        output_dir,
        modes=EVAL_MODES if args.eval_mode == "all" else (args.eval_mode,),
        file_suffix=f"_turn{turn}",
        evaluator_model_name=args.evaluator_model,
        claim_extraction_model_name=args.claim_extraction_model,
        supported_judge_model_name=args.supported_judge_model,
        webpage_summarization_model_name=args.webpage_summarization_model,
        rubric_judge_model_name=args.rubric_judge_model,
        ignore_existing=args.ignore_existing,
    )

    # Save the latest dataset-level metrics across all evaluations
    for eval_type, agg_metrics in summarize(records).items():
        append_jsonl({"turn": turn, "eval_type": eval_type, **agg_metrics}, str(output_dir / f"agg_metrics_turn{turn}.jsonl"))
    logger.info(f"✓ All evaluations completed → {output_dir}")


if __name__ == "__main__":
    main()
