"""
Single entry point for evaluating any Deep Research Agent (DRA) with Mr Dre.

Two subcommands:

    evaluate   Evaluate the reports of one or more turns (Comprehensiveness, Factuality,
               Presentation) and, when several turns are given, the multi-turn revision
               metrics between consecutive turns. Writes summary.json and two CSV tables.

    feedback   Simulate the user feedback for the next turn (content, format, or reflection)
               from the reports of the current turn. Use this when your DRA runs outside
               this repository: feed the exported conversations to your agent, then
               evaluate its revised reports.

Reports can be given in Mr Dre's native format (as written by call_dra.py) or as a flat
JSONL with one {"id": ..., "report": "..."} object per line. See docs/evaluate_your_dra.md.

Examples:
    python mrdre.py evaluate --dataset ResearcherBench --reports my_turn1.jsonl --output_dir results/my-dra
    python mrdre.py feedback --dataset ResearcherBench --output_dir results/my-dra --turn 1 --type content_feedback
    python mrdre.py evaluate --dataset ResearcherBench --reports my_turn1.jsonl my_turn2.jsonl --output_dir results/my-dra
    python mrdre.py evaluate --dataset ResearcherBench --run_dir output/ResearcherBench/content_feedback_k=1/o4-mini-deep-research
"""

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mrdre")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from evaluate_change import COLUMNS as CHANGE_COLUMNS, compute_change_metrics
from evaluate_report import DEFAULT_JUDGE_MODEL, DEFAULT_SUMMARIZATION_MODEL, evaluate_reports
from feedback import REFLECTION_FEEDBACK, generate_feedback_batch, generate_format_feedback_batch
from utils import load_dataset, load_jsonl, resolve_dataset_path, save_jsonl

GEN_FILE_PATTERN = re.compile(r"gen_turn(\d+)\.jsonl$")
FEEDBACK_TYPES = ["content_feedback", "format_feedback", "reflection"]

# Headline metric of each evaluation mode: (aggregate key, column name)
TURN_COLUMNS = {
    "checklist": [("avg_coverage_score", "Checklist Coverage")],
    "citation": [("avg_faithfulness_score", "Citation Faithfulness"), ("avg_groundedness_score", "Claim Groundedness")],
    "rubric": [("avg_presentation", "Presentation")],
}


# --------------------------------------------------------------------------------------
# Input normalization
# --------------------------------------------------------------------------------------

def _match_id(raw_id: Any, dataset: Dict[Any, Dict[str, Any]]) -> Any:
    """Map a report ID onto the dataset's ID, tolerating int/str mismatches (e.g., "12" vs. 12)."""
    if raw_id in dataset:
        return raw_id
    for key in dataset:
        if str(key) == str(raw_id):
            return key
    raise ValueError(f"Report ID {raw_id!r} is not in the dataset.")


def _signal_from_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the feedback signal of a flat record, if it carries one."""
    if record.get("signal"):
        return record["signal"]
    if record.get("feedback"):
        return {
            "feedback_text": record["feedback"],
            "targeted_item": record.get("targeted_item", []),
            "targeted_id": record.get("targeted_id", []),
        }
    return None


def normalize_reports(
    records: List[Dict[str, Any]],
    dataset: Dict[Any, Dict[str, Any]],
    feedback_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert report records into Mr Dre's native format:
        {"id", "question", "question_metadata", "signal", "full_response": {"report", ...}}

    Native records are kept as they are. Flat records ({"id", "report", ...}) are wrapped, and
    any extra fields are kept under full_response. For turns after the first, the feedback
    signal is taken from the record ("signal" or "feedback") or, failing that, from the
    feedback file exported by `mrdre.py feedback`.
    """
    normalized = []
    for record in records:
        qid = _match_id(record.get("id"), dataset)
        item = dataset[qid]

        if isinstance(record.get("full_response"), dict) and "report" in record["full_response"]:
            normalized.append({**record, "id": qid})
            continue
        if not isinstance(record.get("report"), str):
            raise ValueError(
                f"Q{qid}: each record needs either a 'report' string or a 'full_response' dict with a 'report' key."
            )

        signal = _signal_from_record(record)
        if signal is None and feedback_by_id is not None and str(qid) in feedback_by_id:
            fb = feedback_by_id[str(qid)]
            signal = {
                "feedback_text": fb["feedback"],
                "targeted_item": fb.get("targeted_item", []),
                "targeted_id": fb.get("targeted_id", []),
            }

        extra = {k: v for k, v in record.items()
                 if k not in {"id", "question", "report", "signal", "feedback", "targeted_item", "targeted_id"}}
        normalized.append({
            "id": qid,
            "question": record.get("question", item["question"]),
            "question_metadata": item.get("metadata", {}),
            "signal": signal,
            "full_response": {"report": record["report"], **extra},
        })
    return normalized


def _load_feedback_file(output_dir: Path, turn: int) -> Optional[Dict[str, Dict[str, Any]]]:
    path = output_dir / f"feedback_turn{turn}.jsonl"
    if not path.exists():
        return None
    return {str(fb["id"]): fb for fb in load_jsonl(str(path))}


def prepare_turns(
    dataset: Dict[Any, Dict[str, Any]],
    output_dir: Path,
    report_files: Optional[List[str]] = None,
    run_dir: Optional[str] = None,
) -> Dict[int, Path]:
    """
    Collect the report file of each turn and write it, in native format, to
    {output_dir}/gen_turn{i}.jsonl, where all downstream metrics are stored.

    Returns:
        Mapping from turn index to its normalized generation file
    """
    if run_dir:
        sources = {}
        for path in sorted(Path(run_dir).glob("gen_turn*.jsonl")):
            match = GEN_FILE_PATTERN.search(path.name)
            if match:
                sources[int(match.group(1))] = path
        if not sources:
            raise FileNotFoundError(f"No gen_turn{{i}}.jsonl files found in {run_dir}")
    else:
        sources = {i + 1: Path(f) for i, f in enumerate(report_files)}

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_files = {}
    for turn, source in sorted(sources.items()):
        target = output_dir / f"gen_turn{turn}.jsonl"
        records = load_jsonl(str(source))
        normalized = normalize_reports(records, dataset, _load_feedback_file(output_dir, turn))
        if normalized != records or source.resolve() != target.resolve():
            save_jsonl(normalized, str(target))
        gen_files[turn] = target

        missing = len(dataset) - len(normalized)
        logger.info(f"Turn {turn}: {len(normalized)} reports from {source}"
                    + (f" ({missing} dataset questions have no report)" if missing > 0 else ""))
        if turn > 1 and any(r.get("signal") is None for r in normalized):
            logger.warning(f"Turn {turn}: some reports carry no feedback signal; incorporation rate may be undefined.")
    return gen_files


# --------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------

def detect_feedback_type(gen_file: Path) -> Optional[str]:
    """Infer the feedback type of a revision turn from the signals stored with its reports."""
    signals = [r.get("signal") or {} for r in load_jsonl(str(gen_file))]
    texts = [s.get("feedback_text") for s in signals if s.get("feedback_text")]
    if any(s.get("targeted_id") for s in signals):
        return "content_feedback"
    if texts and all(t == REFLECTION_FEEDBACK for t in texts):
        return "reflection"
    if texts:
        return "format_feedback"
    return None


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else ("" if value is None else str(value))


def _print_table(title: str, header: List[str], rows: List[List[Any]]) -> None:
    widths = [max(len(str(h)), *(len(_fmt(r[i])) for r in rows)) for i, h in enumerate(header)]
    print(f"\n{title}")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(_fmt(v).ljust(w) for v, w in zip(row, widths)))


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([[_fmt(v) for v in row] for row in rows])


def run_evaluate(args: argparse.Namespace) -> None:
    dataset_file = resolve_dataset_path(args.dataset)
    dataset = load_dataset(dataset_file)
    output_dir = Path(args.output_dir or args.run_dir)
    gen_files = prepare_turns(dataset, output_dir, args.reports, args.run_dir)

    # 1. Single-report metrics for every turn (resumable: finished reports are skipped)
    turn_rows, summary = [], {"dataset": dataset_file, "turns": {}, "revisions": {}}
    for turn, gen_file in gen_files.items():
        results = evaluate_reports(
            report_file=str(gen_file),
            dataset_file=dataset_file,
            output_dir=str(output_dir),
            eval_mode=args.eval_mode,
            evaluator_model_name=args.judge_model,
            claim_extraction_model_name=args.judge_model,
            supported_judge_model_name=args.judge_model,
            webpage_summarization_model_name=args.summarization_model,
            rubric_judge_model_name=args.judge_model,
            ignore_existing=args.ignore_existing,
            turn=turn,
        )
        row = {"Turn": turn, "Reports": len(load_jsonl(str(gen_file)))}
        failed = set()
        for mode, columns in TURN_COLUMNS.items():
            for key, name in columns:
                row[name] = results[mode]["aggregate"].get(key) if mode in results else None
            if mode in results:
                failed.update(results[mode]["failed_ids"])
        row["Failed"] = len(failed)
        turn_rows.append(row)
        summary["turns"][turn] = {**row, "failed_ids": sorted(failed, key=str)}

    turn_header = list(turn_rows[0].keys())
    _write_csv(output_dir / "summary_turns.csv", turn_header, [list(r.values()) for r in turn_rows])
    _print_table("Per-turn metrics", turn_header, [list(r.values()) for r in turn_rows])

    # 2. Multi-turn revision metrics between consecutive turns
    turns = sorted(gen_files)
    revision_rows = []
    if len(turns) > 1 and args.eval_mode != "all":
        logger.warning("Revision metrics need all three evaluation modes; use --eval_mode all to compute them.")
    elif len(turns) > 1:
        for prev, curr in zip(turns, turns[1:]):
            feedback_type = args.feedback_type if args.feedback_type != "auto" else detect_feedback_type(gen_files[curr])
            metrics = compute_change_metrics(
                str(gen_files[prev]),
                str(gen_files[curr]),
                formal_change=feedback_type == "format_feedback",
                judge_model_name=args.judge_model,
                output_file=str(output_dir / f"incorporation_turn{curr}.jsonl"),
            )
            # Incorporation is only defined for explicit content or format feedback
            if feedback_type not in ("content_feedback", "format_feedback"):
                metrics["Incorporation Rate"] = None
            row = {"Revision": f"{prev}->{curr}", "Feedback": feedback_type or "unknown", **metrics}
            revision_rows.append(row)
            summary["revisions"][f"{prev}->{curr}"] = row

        revision_header = ["Revision", "Feedback"] + CHANGE_COLUMNS
        _write_csv(output_dir / "summary_revisions.csv", revision_header, [list(r.values()) for r in revision_rows])
        _print_table("Revision metrics", revision_header, [list(r.values()) for r in revision_rows])

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary written to {output_dir / 'summary.json'}")
    if any(r["Failed"] for r in turn_rows):
        print("Some reports failed to evaluate and are excluded from the averages; re-run the same command to retry them.")


# --------------------------------------------------------------------------------------
# feedback
# --------------------------------------------------------------------------------------

def _conversation(records_by_turn: Dict[int, Dict[Any, Dict]], qid: Any, upto_turn: int) -> Optional[List[Dict[str, str]]]:
    """Rebuild the conversation (question, reports, and feedback) of a question up to a turn."""
    messages = None
    for turn in range(1, upto_turn + 1):
        record = records_by_turn.get(turn, {}).get(qid)
        if record is None:
            return None
        if turn == 1:
            messages = [{"role": "user", "content": record["question"]}]
        else:
            feedback_text = (record.get("signal") or {}).get("feedback_text")
            if not feedback_text:
                return None
            messages.append({"role": "user", "content": feedback_text})
        messages.append({"role": "assistant", "content": record["full_response"]["report"]})
    return messages


def run_feedback(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dataset_file = resolve_dataset_path(args.dataset)
    dataset = load_dataset(dataset_file)
    output_dir = Path(args.output_dir or args.run_dir)
    gen_files = prepare_turns(dataset, output_dir, args.reports, args.run_dir)
    if args.turn not in gen_files:
        raise ValueError(f"Turn {args.turn} not found; available turns: {sorted(gen_files)}")

    reports = load_jsonl(str(gen_files[args.turn]))
    qids = [str(r["id"]) for r in reports]
    questions_by_id = {str(r["id"]): r["question"] for r in reports}

    if args.type == "content_feedback":
        # Content feedback targets checklist items the current report fails, so the
        # checklist evaluation of this turn is needed first (resumed if already done).
        evaluate_reports(
            report_file=str(gen_files[args.turn]),
            dataset_file=dataset_file,
            output_dir=str(output_dir),
            eval_mode="checklist",
            evaluator_model_name=args.judge_model,
            turn=args.turn,
        )
        eval_map = {str(m["id"]): m for m in load_jsonl(str(output_dir / f"checklist_turn{args.turn}.jsonl"))}
        feedbacks = generate_feedback_batch(qids, eval_map, questions_by_id, num_points=args.num_items)
    elif args.type == "format_feedback":
        reports_by_id = {str(r["id"]): r["full_response"]["report"] for r in reports}
        feedbacks = generate_format_feedback_batch(qids, reports_by_id, questions_by_id)
    else:
        feedbacks = [{"id": r["id"], "feedback": REFLECTION_FEEDBACK, "item": [], "item_id": []} for r in reports]

    next_turn = args.turn + 1
    records_by_turn = {t: {r["id"]: r for r in load_jsonl(str(f))} for t, f in gen_files.items() if t <= args.turn}
    records = []
    for fb in feedbacks:
        qid = _match_id(fb["id"], dataset)
        messages = _conversation(records_by_turn, qid, args.turn)
        if messages is None:
            logger.warning(f"Q{qid}: conversation history is incomplete, skipping")
            continue
        records.append({
            "id": qid,
            "turn": next_turn,
            "type": args.type,
            "feedback": fb["feedback"],
            "targeted_item": fb.get("item", []),
            "targeted_id": fb.get("item_id", []),
            "messages": messages + [{"role": "user", "content": fb["feedback"]}],
        })

    output_file = output_dir / f"feedback_turn{next_turn}.jsonl"
    save_jsonl(records, str(output_file))
    skipped = len(reports) - len(records)
    print(f"\nWrote {len(records)} {args.type} prompts for turn {next_turn} → {output_file}")
    if skipped:
        print(f"{skipped} questions received no feedback (e.g., no failed checklist item to target).")
    print("Run your DRA on each record's 'messages' and save {\"id\", \"report\"} per line; "
          "then pass that file as the next turn to `mrdre.py evaluate`.")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (ResearcherBench, RigorousBench, ResearchRubrics) or path to a dataset JSONL file")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reports", nargs="+",
                        help="Report JSONL files, one per turn, in turn order (turn 1 first)")
    source.add_argument("--run_dir",
                        help="Directory containing gen_turn{i}.jsonl files (e.g., written by call_dra.py). "
                             "If neither --reports nor --run_dir is given, the files already in --output_dir are used.")
    parser.add_argument("--output_dir",
                        help="Directory for normalized reports, metrics, and summaries (defaults to --run_dir)")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL,
                        help=f"LLM judge for all evaluation steps (default: {DEFAULT_JUDGE_MODEL})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate any Deep Research Agent with Mr Dre.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate reports and multi-turn revisions")
    _add_input_args(evaluate)
    evaluate.add_argument("--eval_mode", default="all", choices=["all", "checklist", "citation", "rubric"],
                          help="Evaluation dimensions to run (default: all)")
    evaluate.add_argument("--summarization_model", default=DEFAULT_SUMMARIZATION_MODEL,
                          help=f"Model that summarizes cited webpages (default: {DEFAULT_SUMMARIZATION_MODEL})")
    evaluate.add_argument("--feedback_type", default="auto", choices=["auto"] + FEEDBACK_TYPES,
                          help="Feedback type of the revision turns (default: inferred from the stored feedback)")
    evaluate.add_argument("--ignore_existing", action="store_true",
                          help="Re-evaluate all reports instead of resuming from existing metrics")

    feedback = subparsers.add_parser("feedback", help="Simulate user feedback for the next turn")
    _add_input_args(feedback)
    feedback.add_argument("--turn", type=int, required=True, help="Turn whose reports receive feedback")
    feedback.add_argument("--type", required=True, choices=FEEDBACK_TYPES, help="Feedback type")
    feedback.add_argument("--num_items", type=int, default=1,
                          help="Checklist items targeted per question (content_feedback only; default: 1)")
    feedback.add_argument("--seed", type=int, default=42, help="Random seed for selecting feedback targets")

    args = parser.parse_args()
    if args.output_dir is None and args.run_dir is None:
        parser.error("give --output_dir (and --reports), or --run_dir")
    if args.reports is None and args.run_dir is None:
        args.run_dir = args.output_dir
    if args.command == "evaluate":
        run_evaluate(args)
    else:
        run_feedback(args)


if __name__ == "__main__":
    sys.exit(main())
