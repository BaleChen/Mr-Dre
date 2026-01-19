"""
Compare two generation runs and emit a single CSV-friendly summary row with:
Break Rate, Incorporation Rate, Checklist Coverage Delta,
Citation Faithfulness Delta, Claim Groundedness Delta, Presentation Delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from dotenv import load_dotenv
from tqdm import tqdm

from dr_eval.eval_prompts import PAIRWISE_JUDGE_SYSTEM_PROMPT, PAIRWISE_JUDGE_USER_PROMPT
from engine.oai import GPT
from utils import load_jsonl

load_dotenv()

logger = logging.getLogger(__name__)

GEN_FILE_PATTERN = re.compile(r"gen_turn(\d+)\.jsonl$", re.IGNORECASE)
CHECKLIST_FLIP_THRESHOLD = 1e-6
COLUMNS = [
    "Break Rate",
    "Incorporation Rate",
    "Checklist Coverage Delta",
    "Citation Faithfulness Delta",
    "Claim Groundedness Delta",
    "Presentation Delta",
]
GPT_MODEL = "gpt-4.1-mini"


def parse_run(gen_file: str) -> Tuple[Path, Path, int]:
    path = Path(gen_file).expanduser().resolve()
    turn = int(GEN_FILE_PATTERN.search(path.name).group(1))
    parts = list(path.parts)
    out_idx = parts.index("output")
    run_dir = Path(*parts[: out_idx + 4])
    return path, run_dir, turn


def load_metrics(run_dir: Path, turn: int, filename: str) -> List[Dict]:
    path = run_dir / filename.format(turn=turn)
    return load_jsonl(str(path)) if path.is_file() else []


def align_by_id(items_a: Sequence[Dict], items_b: Sequence[Dict]) -> List[Tuple[Dict, Dict]]:
    """Return pairs of entries that share the same `id` value."""
    indexed = {item["id"]: item for item in items_a if "id" in item}
    return [
        (indexed[item["id"]], item)
        for item in items_b
        if item.get("id") in indexed
    ]


def _is_point_successful(point: Dict[str, Any]) -> bool:
    weight = point["weight"]
    covered = point["covered"]
    if weight < 0:
        return covered == 0.0
    return covered == 1.0


def collect_targeted_items(gen_reports: Sequence[Dict[str, Any]]) -> Dict[str, Set[int]]:
    targeted: Dict[str, Set[int]] = {}
    for entry in gen_reports:
        qid = str(entry["id"])
        signal = entry.get("signal") or {}
        raw_ids = None
        for key in ("targeted_id", "targeted_ids", "targeted_items", "targeted_item_ids"):
            if key in signal:
                raw_ids = signal.get(key)
                break
        if not raw_ids:
            continue
        targeted[qid] = set(raw_ids)
    return targeted


def checklist_stats(
    check_v1: Sequence[Dict],
    check_v2: Sequence[Dict],
    targeted_items: Optional[Mapping[str, Set[int]]],
) -> Tuple[float, float, float]:
    """
    Compute checklist statistics between two versions.
    
    Returns:
        Tuple of (fix_rate, break_rate, coverage_delta)
    """
    pairs = align_by_id(check_v1, check_v2)
    if not pairs:
        return 0.0, 0.0, 0.0

    targeted_lookup = {str(k): set(v) for k, v in (targeted_items or {}).items()}

    fix_rates = []
    break_rates = []
    coverage_deltas = []

    for m1, m2 in pairs:
        coverage_deltas.append(m2.get("coverage_score", 0.0) - m1.get("coverage_score", 0.0))

        details1 = {d.get("id"): d for d in m1.get("checklist_details", []) if d.get("id") is not None}
        details2 = {d.get("id"): d for d in m2.get("checklist_details", []) if d.get("id") is not None}
        shared_ids = set(details1) & set(details2)
        question_targeted = targeted_lookup.get(str(m1.get("id")), set())
        targeted_ids = question_targeted & shared_ids

        # Calculate fix rate for targeted items
        s_fix_total = 0
        s_fix_count = 0
        for cid in targeted_ids:
            point_v2 = details2[cid]
            if _is_point_successful(point_v2):
                s_fix_count += 1
            s_fix_total += 1

        if s_fix_total > 0:
            fix_rates.append(s_fix_count / s_fix_total)

        # Calculate break rate for non-targeted items
        s_broken = 0
        s_break_denom = 0
        for cid in shared_ids - targeted_ids:
            p1, p2 = details1[cid], details2[cid]
            v1, v2 = p1["covered"], p2["covered"]
            w = p1.get("weight", 1.0)

            directed_delta = (v2 - v1) if w >= 0 else (v1 - v2)

            if (w >= 0 and v1 > CHECKLIST_FLIP_THRESHOLD) or \
               (w < 0 and v1 < 1.0 - CHECKLIST_FLIP_THRESHOLD):
                s_break_denom += 1

            if directed_delta < -CHECKLIST_FLIP_THRESHOLD:
                s_broken += 1

        if s_break_denom > 0:
            break_rates.append(s_broken / s_break_denom)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return _avg(fix_rates), _avg(break_rates), _avg(coverage_deltas)


def citation_stats(cite_v1: List[Dict], cite_v2: List[Dict]) -> Tuple[float, float]:
    pairs = align_by_id(cite_v1, cite_v2)
    if not pairs:
        return 0.0, 0.0
    faith = []
    ground = []
    for m1, m2 in pairs:
        faith.append(m2.get("faithfulness_score", 0.0) - m1.get("faithfulness_score", 0.0))
        ground.append(m2.get("groundedness_score", 0.0) - m1.get("groundedness_score", 0.0))
    return (
        sum(faith) / len(faith) if faith else 0.0,
        sum(ground) / len(ground) if ground else 0.0,
    )


def rubric_stats(rub_v1: List[Dict], rub_v2: List[Dict]) -> float:
    pairs = align_by_id(rub_v1, rub_v2)
    deltas = [m2["organization"] - m1["organization"] for m1, m2 in pairs if "organization" in m1 and "organization" in m2]
    return sum(deltas) / len(deltas) if deltas else 0.0


def judge_incorporation_rate(gen_reports_v1: List[Dict], gen_reports_v2: List[Dict]) -> Tuple[float, List[Dict]]:
    pairs = align_by_id(gen_reports_v1, gen_reports_v2)
    if not pairs:
        return 0.0, []
    judge_model = GPT(GPT_MODEL, system_prompt=PAIRWISE_JUDGE_SYSTEM_PROMPT)
    scores: List[Optional[float]] = [None] * len(pairs)
    detailed_results: List[Dict] = []

    def score_pair(pair: Tuple[Dict, Dict]) -> Tuple[float, Dict]:
        d1, d2 = pair
        prompt = PAIRWISE_JUDGE_USER_PROMPT.format(
            question=d1["question"], 
            report=d1["full_response"]["report"],
            revised_report=d2["full_response"]["report"],
            feedback=d2["signal"],
        )
        response = judge_model(prompt)
        response_dict = json.loads(response)
        score = response_dict["score"]
        
        # Collect detailed information
        result = {
            "id": d1["id"],
            "score": score,
            "question": d1["question"],
            "signal": d2["signal"],
            "report1": d1["full_response"]["report"],
            "report2": d2["full_response"]["report"]
        }
        return score, result

    max_workers = min(len(pairs), 10) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_pair, pair): idx for idx, pair in enumerate(pairs)}
        with tqdm(
            total=len(pairs),
            desc="Judging incorporation",
            leave=False,
            disable=len(pairs) < 5,
        ) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    score, result = future.result()
                    scores[idx] = score
                    detailed_results.append(result)
                except Exception as exc:
                    scores[idx] = None
                    logger.error(f"Incorporation judge failed for pair {idx}: {exc}")
                finally:
                    pbar.update(1)

    valid_scores = [score for score in scores if score is not None]
    average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    return average_score, detailed_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two generation runs.")
    parser.add_argument("gen_file_v1", type=str, help="Generation file for version 1")
    parser.add_argument("gen_file_v2", type=str, help="Generation file for version 2")
    parser.add_argument("--formal-change", default=False, action="store_true",
                        help="Whether to compute the eval metrics for formal feedback.")
    parser.add_argument("--output-file", type=str, default="incorporation_results.jsonl",
                        help="Output file to save detailed scores")
    args = parser.parse_args()

    gen_v1, run_v1, turn_v1 = parse_run(args.gen_file_v1)
    gen_v2, run_v2, turn_v2 = parse_run(args.gen_file_v2)

    gen_reports_v1 = load_jsonl(str(gen_v1))
    gen_reports_v2 = load_jsonl(str(gen_v2))
    targeted_items = collect_targeted_items(gen_reports_v2)

    # Load metrics
    check_v1 = load_metrics(run_v1, turn_v1, "checklist_turn{turn}.jsonl")
    check_v2 = load_metrics(run_v2, turn_v2, "checklist_turn{turn}.jsonl")
    cite_v1 = load_metrics(run_v1, turn_v1, "citation_turn{turn}.jsonl")
    cite_v2 = load_metrics(run_v2, turn_v2, "citation_turn{turn}.jsonl")
    rub_v1 = load_metrics(run_v1, turn_v1, "rubric_turn{turn}.jsonl")
    rub_v2 = load_metrics(run_v2, turn_v2, "rubric_turn{turn}.jsonl")

    fix_rate, break_rate, coverage_delta = checklist_stats(check_v1, check_v2, targeted_items)
    faith_delta, ground_delta = citation_stats(cite_v1, cite_v2)
    presentation_delta = rubric_stats(rub_v1, rub_v2)

    # Compute incorporation rate: fix_rate for checklist feedback, judge for formal feedback
    # (they don't happen together, so we merge them into one metric)
    if args.formal_change:
        incorporation_rate, detailed_results = judge_incorporation_rate(gen_reports_v1, gen_reports_v2)

        if detailed_results:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in detailed_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"Detailed results saved to: {output_path}")
    else:
        incorporation_rate = fix_rate

    values = [
        break_rate,
        incorporation_rate,
        coverage_delta,
        faith_delta,
        ground_delta,
        presentation_delta,
    ]

    writer = csv.writer(sys.stdout)
    writer.writerow(COLUMNS)
    writer.writerow([f"{val:.4f}" for val in values])


if __name__ == "__main__":
    main()
