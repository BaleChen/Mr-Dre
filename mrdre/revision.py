"""
Revision metrics between two turns of reports: Break Rate, Incorporation Rate,
Checklist Coverage Delta, Citation Faithfulness Delta, Claim Groundedness Delta
and Presentation Delta.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from tqdm import tqdm

from mrdre.dr_eval.eval_prompts import PAIRWISE_JUDGE_SYSTEM_PROMPT, PAIRWISE_JUDGE_USER_PROMPT
from mrdre.engine.oai import GPT
from mrdre.utils import save_jsonl

logger = logging.getLogger(__name__)

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
    deltas = [m2["presentation"] - m1["presentation"] for m1, m2 in pairs if "presentation" in m1 and "presentation" in m2]
    return sum(deltas) / len(deltas) if deltas else 0.0


def judge_incorporation_rate(
    gen_reports_v1: List[Dict],
    gen_reports_v2: List[Dict],
    judge_model_name: str = GPT_MODEL,
) -> Tuple[float, List[Dict]]:
    pairs = align_by_id(gen_reports_v1, gen_reports_v2)
    if not pairs:
        return 0.0, []
    judge_model = GPT(judge_model_name, system_prompt=PAIRWISE_JUDGE_SYSTEM_PROMPT)
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

def attach_feedback(reports: List[Dict[str, Any]], feedback_records: List[Dict[str, Any]]) -> int:
    """
    Store each report's feedback as its "signal", in the form generation writes.

    The feedback file is the only source: a "signal" already in a report is replaced,
    and removed if the feedback file has no record for that report.

    Args:
        reports: Revised reports, modified in place
        feedback_records: Feedback records with "id", "feedback", "item" and "item_id"

    Returns:
        Number of reports that received feedback
    """
    feedback_by_id = {str(item["id"]): item for item in feedback_records}
    matched = 0
    for report in reports:
        item = feedback_by_id.get(str(report["id"]))
        if item is None:
            report.pop("signal", None)
            continue
        report["signal"] = {
            "feedback_text": item.get("feedback", ""),
            "targeted_item": item.get("item", []),
            "targeted_id": item.get("item_id", []),
        }
        matched += 1
    return matched


def compute_revision_metrics(
    gen_reports_v1: List[Dict[str, Any]],
    gen_reports_v2: List[Dict[str, Any]],
    metrics_v1: Mapping[str, List[Dict[str, Any]]],
    metrics_v2: Mapping[str, List[Dict[str, Any]]],
    feedback_type: str,
    judge_model_name: str = GPT_MODEL,
    judge_output_file: Optional[Path] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute the revision metrics between two turns.

    Args:
        gen_reports_v1: Reports before revision
        gen_reports_v2: Revised reports, each with the "signal" it answered
        metrics_v1: Per-report evaluation records of v1, by mode ("checklist", "citation", "rubric")
        metrics_v2: Per-report evaluation records of v2, by mode
        feedback_type: "content" (Incorporation Rate is the fix rate of the targeted checklist
            items), "format" (pairwise LLM judge) or "reflection" (no Incorporation Rate)
        judge_model_name: Model for the pairwise judge
        judge_output_file: Where to save the pairwise judge details (format feedback only)

    Returns:
        Mapping from metric name (see COLUMNS) to value. Metrics whose evaluation mode
        was not run are omitted; Incorporation Rate is None for reflection.
    """
    if feedback_type not in ("content", "format", "reflection"):
        raise ValueError(f"Unknown feedback type: {feedback_type!r}")

    results: Dict[str, Optional[float]] = {}
    fix_rate = None
    if "checklist" in metrics_v1 and "checklist" in metrics_v2:
        targeted_items = collect_targeted_items(gen_reports_v2)
        fix_rate, break_rate, coverage_delta = checklist_stats(
            metrics_v1["checklist"], metrics_v2["checklist"], targeted_items
        )
        results["Break Rate"] = break_rate
        results["Checklist Coverage Delta"] = coverage_delta

    # Incorporation Rate: fix rate for content feedback, pairwise judge for format feedback
    # (they don't happen together, so we merge them into one metric)
    if feedback_type == "format":
        answered = [report for report in gen_reports_v2 if report.get("signal")]
        incorporation_rate, detailed_results = judge_incorporation_rate(
            gen_reports_v1, answered, judge_model_name
        )
        results["Incorporation Rate"] = incorporation_rate
        if detailed_results and judge_output_file is not None:
            judge_output_file.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(detailed_results, str(judge_output_file))
            logger.info(f"Detailed results saved to: {judge_output_file}")
    elif feedback_type == "content":
        if fix_rate is not None:
            results["Incorporation Rate"] = fix_rate
    else:
        results["Incorporation Rate"] = None

    if "citation" in metrics_v1 and "citation" in metrics_v2:
        faith_delta, ground_delta = citation_stats(metrics_v1["citation"], metrics_v2["citation"])
        results["Citation Faithfulness Delta"] = faith_delta
        results["Claim Groundedness Delta"] = ground_delta
    if "rubric" in metrics_v1 and "rubric" in metrics_v2:
        results["Presentation Delta"] = rubric_stats(metrics_v1["rubric"], metrics_v2["rubric"])

    return {column: results[column] for column in COLUMNS if column in results}
