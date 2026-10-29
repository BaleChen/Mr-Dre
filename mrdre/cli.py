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
    changed_reports,
    check_output_dir,
    evaluate_reports,
    load_evaluation,
    load_reports,
    summarize,
)
from mrdre.feedback import FEEDBACK_TYPES, simulate_feedback  # noqa: E402
from mrdre.generate import generate_reports, load_engine  # noqa: E402
from mrdre.revision import attach_feedback, compute_revision_metrics  # noqa: E402
from mrdre.utils import (  # noqa: E402
    append_jsonl,
    load_jsonl,
    load_partial_jsonl,
    resolve_dataset,
    sample_questions,
    save_json,
    save_jsonl,
    setup_logging,
)

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
    _add_generate_parser(subparsers)
    _add_feedback_parser(subparsers)
    _add_evaluate_parser(subparsers)

    args = parser.parse_args(argv)
    setup_logging()
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as e:
        # Problems with the inputs: show the message without a traceback
        raise SystemExit(f"mrdre {args.command}: {e}")


def _load_dataset(name_or_path: str, limit: int):
    """Return the dataset indexed by question ID and the selected dataset items."""
    dataset_items = load_jsonl(str(resolve_dataset(name_or_path)))
    dataset = {item["id"]: item for item in dataset_items}
    return dataset, sample_questions(dataset_items, limit)


# --------------------------------------------------------------------------- #
# mrdre generate                                                              #
# --------------------------------------------------------------------------- #

def _add_generate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Generate a turn of reports with a supported engine or your own function",
        description=(
            "Write one report per question. Without --previous, the engine receives the question "
            "(the first turn). With --previous and --feedback, it receives the conversation so far "
            "followed by the feedback, and revises its report."
        ),
    )
    parser.add_argument("--dataset", required=True, help=DATASET_HELP)
    parser.add_argument("--engine",
                        help="A built-in engine (o3-deep-research, o4-mini-deep-research, sonar-deep-research; "
                             "dr-tulu or odr-* with your own DR Tulu or LangGraph server), or module:function "
                             "for a function in "
                             "your code, where module is importable from the working directory or is a path "
                             "to a .py file. The function takes the message list and returns the report as a "
                             "string or as a dict with a 'report' key; up to --batch-size calls run at once, "
                             "in threads")
    parser.add_argument("--out", required=True, help="Output reports JSONL file")
    parser.add_argument("--previous", nargs="+", metavar="REPORTS",
                        help="Reports of all earlier turns, oldest first")
    parser.add_argument("--feedback", help="Feedback on the latest --previous reports, from `mrdre feedback`")
    parser.add_argument("--use-reviser", action="store_true",
                        help="Instead of an engine, revise the latest reports with the reviser agent "
                             "(a model served with vLLM at localhost:8005, with web search)")
    parser.add_argument("--limit", type=int, default=-1, help=LIMIT_HELP)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Requests sent to the engine at a time (default: 10)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate reports of questions that already have a report in --out")
    parser.set_defaults(run=_run_generate)


def _run_generate(args: argparse.Namespace) -> None:
    if bool(args.previous) != bool(args.feedback):
        raise SystemExit("mrdre generate: a revision turn needs both --previous and --feedback")
    if bool(args.engine) == args.use_reviser:
        raise SystemExit("mrdre generate: pass either --engine or --use-reviser")
    if args.use_reviser and not args.previous:
        raise SystemExit("mrdre generate: --use-reviser revises earlier reports and needs --previous and --feedback")
    if args.batch_size < 1:
        raise SystemExit("mrdre generate: --batch-size must be at least 1")
    out = Path(args.out)
    inputs = [*(args.previous or []), *([args.feedback] if args.feedback else [])]
    if any(out.resolve() == Path(path).resolve() for path in inputs):
        raise SystemExit("mrdre generate: --out must not be one of the input files")

    dataset, questions = _load_dataset(args.dataset, args.limit)
    question_ids = {question["id"] for question in questions}
    previous_turns = [load_reports(path, dataset, question_ids) for path in args.previous or []]
    feedback_records = load_jsonl(args.feedback) if args.feedback else None

    if args.use_reviser:
        from mrdre.engine.reviser_agent import ReviserAgent
        agent = ReviserAgent()
    else:
        agent = load_engine(args.engine)
    missing = generate_reports(
        agent,
        questions,
        out,
        previous_turns,
        feedback_records,
        use_reviser=args.use_reviser,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    if missing:
        raise SystemExit(
            f"mrdre generate: {len(missing)} of {len(questions)} questions have no report (e.g. Q{missing[0]}) "
            "because their requests failed or timed out; run the same command again to retry them"
        )


# --------------------------------------------------------------------------- #
# mrdre feedback                                                              #
# --------------------------------------------------------------------------- #

def _add_feedback_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "feedback",
        help="Simulate user feedback on a turn of reports",
        description=(
            "Write one simulated user feedback per report: content feedback on checklist items "
            "the report missed, format feedback on its writing and presentation, or the fixed "
            "reflection request. The feedback only goes into the output file; the next turn is "
            "generated separately."
        ),
    )
    parser.add_argument("--dataset", required=True, help=DATASET_HELP)
    parser.add_argument("--reports", required=True, help="Reports JSONL file to give feedback on")
    parser.add_argument("--feedback-type", required=True, choices=FEEDBACK_TYPES, help="Feedback type")
    parser.add_argument("--out", required=True, help="Output feedback JSONL file")
    parser.add_argument("--eval",
                        help="Checklist scores of the reports, required for content feedback: an "
                             "`mrdre evaluate` output directory or a checklist JSONL file")
    parser.add_argument("--num-items", type=int, default=1,
                        help="Missed checklist items per content feedback (default: 1)")
    parser.add_argument("--refine", action="store_true",
                        help="Rewrite each content or format feedback into a localized edit plan (uses gpt-4.1)")
    parser.add_argument("--limit", type=int, default=-1, help=LIMIT_HELP)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for choosing checklist items and format examples, combined with "
                             "each question ID so a report's feedback does not depend on the other reports "
                             "(default: 42)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate feedback for reports that already have feedback in --out")
    parser.set_defaults(run=_run_feedback)


def _run_feedback(args: argparse.Namespace) -> None:
    if args.feedback_type == "content" and not args.eval:
        raise SystemExit("mrdre feedback: content feedback needs the checklist scores of the reports (--eval)")
    if args.refine and args.feedback_type == "reflection":
        raise SystemExit("mrdre feedback: --refine applies to content and format feedback")
    if args.num_items < 1:
        raise SystemExit("mrdre feedback: --num-items must be at least 1")

    dataset, questions = _load_dataset(args.dataset, args.limit)
    reports = load_reports(args.reports, dataset, {question["id"] for question in questions})

    checklist_records = None
    if args.feedback_type == "content":
        eval_path = Path(args.eval)
        if eval_path.is_dir():
            changed = changed_reports(eval_path, reports)
            if changed:
                raise ValueError(f"{eval_path} holds scores of different reports (e.g. Q{', Q'.join(changed[:5])})")
            eval_path = eval_path / "checklist.jsonl"
        checklist_records = load_jsonl(str(eval_path))

    # Resume: skip reports that already have feedback in the output file
    out = Path(args.out)
    existing = load_partial_jsonl(out)
    report_ids = {str(report["id"]) for report in reports}
    if args.overwrite and existing:
        existing = [item for item in existing if str(item["id"]) not in report_ids]
        save_jsonl(existing, str(out))
    # A file holds one kind of feedback, since evaluation reads its type from the records
    kind = _feedback_kind(args.feedback_type, args.refine, args.num_items)
    other = next((item for item in existing if _feedback_kind(
        item.get("feedback_type", args.feedback_type), "original_feedback" in item, len(item.get("item_id") or [])
    ) != kind), None)
    if other is not None:
        raise ValueError(f"{out} holds other feedback than {kind} (e.g. Q{other['id']}); use a new --out or --overwrite")
    done_ids = {str(item["id"]) for item in existing}
    pending = [report for report in reports if str(report["id"]) not in done_ids]
    if not pending:
        logger.info(f"All {len(reports)} reports already have feedback in {out}")
        return
    if len(pending) < len(reports):
        logger.info(f"Resuming: {len(reports) - len(pending)} reports already have feedback in {out}")

    logger.info(f"Simulating {args.feedback_type} feedback on {len(pending)} reports from {args.reports}")
    written = 0
    for item in simulate_feedback(
        pending, args.feedback_type, checklist_records, args.num_items, args.refine, args.seed
    ):
        append_jsonl(item, str(out))
        written += 1
    skipped = f" ({len(pending) - written} without feedback)" if written < len(pending) else ""
    logger.info(f"✓ {written} feedback records → {out}{skipped}")


def _feedback_kind(feedback_type: str, refined: bool, num_items: int) -> str:
    """Describe a kind of feedback, e.g. "refined content feedback on 2 checklist items"."""
    items = f" on {num_items} checklist item{'s' if num_items > 1 else ''}" if feedback_type == "content" else ""
    return f"{'refined ' if refined else ''}{feedback_type} feedback{items}"


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

    dataset, questions = _load_dataset(args.dataset, args.limit)
    question_ids = {question["id"] for question in questions}
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
