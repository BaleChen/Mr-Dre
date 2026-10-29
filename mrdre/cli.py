"""
The mrdre command: simulate feedback, evaluate reports, and generate reports
with the supported engines.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv

# Load .env from the working directory before importing mrdre modules:
# the disk caches read MCP_CACHE_DIR when they are imported
load_dotenv(find_dotenv(usecwd=True))

from mrdre.evaluate import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    EVAL_MODES,
    check_output_dir,
    evaluate_reports,
    load_evaluation,
    load_reports,
    summarize,
)
from mrdre.feedback import FEEDBACK_TYPES  # noqa: E402
from mrdre.revision import attach_feedback, compute_revision_metrics  # noqa: E402
from mrdre.utils import load_jsonl, resolve_dataset, sample_questions, save_json, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_HELP = (
    "Benchmark name in data/ (researcherbench, researchrubrics, rigorousbench) "
    "or path to a dataset JSONL file with id, question and checklist"
)
LIMIT_HELP = (
    "Use a fixed random subset of N questions (seed 42); generation, feedback and "
    "evaluation select the same subset for the same N (default: -1 for all)"
)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="mrdre", description=__doc__.strip())
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_evaluate_parser(subparsers)

    args = parser.parse_args(argv)
    setup_logging()
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as e:
        # Problems with the inputs: show the message without a traceback
        raise SystemExit(f"mrdre {args.command}: {e}")


# --------------------------------------------------------------------------- #
# mrdre evaluate                                                              #
# --------------------------------------------------------------------------- #

def _add_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Score a reports file; with --previous-eval and --feedback, also compute revision metrics",
        description=(
            "Score each report for comprehensiveness (checklist), factuality (citation) and "
            "presentation (rubric). With --previous-eval and --feedback, also compute the revision "
            "metrics against the previous turn, from the scores of its earlier evaluation."
        ),
    )
    parser.add_argument("--dataset", required=True, help=DATASET_HELP)
    parser.add_argument("--reports", required=True,
                        help="Reports JSONL file: one record per question with 'id' and 'report' "
                             "(or 'full_response.report', as written by generation)")
    parser.add_argument("--out", required=True,
                        help="Output directory for per-report scores and summary.json")
    parser.add_argument("--mode", default="all", choices=[*EVAL_MODES, "all"],
                        help="checklist (comprehensiveness), citation (factuality), "
                             "rubric (presentation), or all (default)")
    parser.add_argument("--previous-eval", metavar="DIR",
                        help="Output directory of `mrdre evaluate` for the previous turn; enables the "
                             "revision metrics, computed from its scores without re-scoring")
    parser.add_argument("--feedback",
                        help="Feedback file that the reports answered (required with --previous-eval)")
    parser.add_argument("--feedback-type", choices=FEEDBACK_TYPES,
                        help="Feedback type, for feedback files without a feedback_type field")
    parser.add_argument("--limit", type=int, default=-1, help=LIMIT_HELP)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL}, as in the paper); "
                             "webpage summaries always use gpt-4.1-nano")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-evaluate reports that already have scores in --out")
    parser.set_defaults(run=_run_evaluate)


def _run_evaluate(args: argparse.Namespace) -> None:
    # Validate all inputs before any judge call
    if args.feedback and not args.previous_eval:
        raise SystemExit("mrdre evaluate: --feedback requires --previous-eval")
    if args.previous_eval and not args.feedback:
        raise SystemExit("mrdre evaluate: --previous-eval requires --feedback")
    out = Path(args.out)
    if args.previous_eval and Path(args.previous_eval).resolve() == out.resolve():
        raise SystemExit("mrdre evaluate: --previous-eval is the previous turn's output directory, not --out")

    dataset_items = load_jsonl(str(resolve_dataset(args.dataset)))
    dataset = {item["id"]: item for item in dataset_items}
    question_ids = {item["id"] for item in sample_questions(dataset_items, args.limit)}
    reports = load_reports(args.reports, dataset, question_ids)
    modes = EVAL_MODES if args.mode == "all" else (args.mode,)
    if args.previous_eval:
        previous_reports, previous_records = load_evaluation(
            Path(args.previous_eval), dataset, {report["id"] for report in reports}, modes, args.judge_model
        )
        feedback_records = load_jsonl(args.feedback)
        feedback_type = _feedback_type(feedback_records, args.feedback_type)

    judge_models = {
        "evaluator_model_name": args.judge_model,
        "claim_extraction_model_name": args.judge_model,
        "supported_judge_model_name": args.judge_model,
        "rubric_judge_model_name": args.judge_model,
    }
    check_output_dir(out, reports, args.overwrite)

    logger.info(f"Evaluating {len(reports)} reports from {args.reports}")
    records = evaluate_reports(reports, dataset, out, modes, ignore_existing=args.overwrite, **judge_models)
    summary: Dict[str, Any] = summarize(records)

    if args.previous_eval:
        logger.info(f"Comparing with the scores of the previous turn in {args.previous_eval}")
        matched = attach_feedback(reports, feedback_records)
        logger.info(f"Matched {matched}/{len(reports)} reports to {feedback_type} feedback")
        summary["revision"] = compute_revision_metrics(
            previous_reports,
            reports,
            previous_records,
            records,
            feedback_type,
            judge_model_name=args.judge_model,
            judge_output_file=out / "incorporation_judge.jsonl",
        )

    save_json(summary, str(out / "summary.json"))
    logger.info(f"✓ Summary → {out / 'summary.json'}")
    print(json.dumps(summary, indent=2))


def _feedback_type(feedback_records: List[Dict[str, Any]], override: Optional[str]) -> str:
    """Return the feedback type, from --feedback-type or from the records' feedback_type field."""
    if override:
        return override
    types = {item.get("feedback_type") for item in feedback_records}
    if len(types) == 1 and types <= set(FEEDBACK_TYPES):
        return types.pop()
    raise SystemExit(
        f"mrdre evaluate: cannot tell the feedback type from the feedback file (found {sorted(map(str, types))}); "
        "pass --feedback-type"
    )


if __name__ == "__main__":
    main()
