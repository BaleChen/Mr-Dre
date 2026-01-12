"""
Checklist-based evaluation module.
Evaluates reports against a checklist of required points.
"""

import json
import logging
import re
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dr_eval.eval_prompts import (
    get_checklist_eval_prompts,
)
from utils import model_name

logger = logging.getLogger(__name__)


def safe_load_json(response: Any, idx: int, question_id: int):
    """
    Safely parse JSON response with fallback for invalid escape sequences.
    
    Args:
        response: JSON string to parse
        idx: Index of the checklist item (for logging)
        question_id: Question ID (for logging)
    
    Returns:
        Parsed JSON object
    """
    valid_escapes = r'["\\/bfnrtu]'
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(f"JSON decode failed (idx={idx}, Q{question_id}), attempting fix")
        fixed = re.sub(rf'\\(?!{valid_escapes})', r'\\\\', response)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed after fix (idx={idx}, Q{question_id}): {e}")
            raise


def evaluate_checklist(
    report_obj: Dict[str, Any],
    dataset_item: Dict[str, Any],
    evaluator_model: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a report against a checklist.
    
    Args:
        report_obj: Generated report object with structure:
            {
                "id": int,
                "question": str,
                "question_metadata": Dict,
                "signal": Dict,
                "full_response": Dict,
                "report_length": int
            }
        dataset_item: Dataset item with checklist:
            {
                "id": int,
                "question": str,
                "checklist": List[Dict],
                "metadata": Dict
            }
        evaluator_model: Model ID to use for evaluation
        **kwargs: Additional arguments (e.g., API keys, model config)
    
    Returns:
        Dict with structure:
        {
            "id": int,
            "checklist_evaluator_model": str,
            "coverage_score": float,
            "total_points": int,
            "covered_points": int,
            "checklist_details": List[Dict]
        }
    """
    question = dataset_item["question"]
    report = report_obj["full_response"]["report"]
    checklist = dataset_item["checklist"]
    
    prompts = get_checklist_eval_prompts(question, report, checklist)

    # Multi-threaded GPT calls
    call_gpt = lambda prompt: evaluator_model(prompt)
    responses = []
    with ThreadPoolExecutor(max_workers=min(len(prompts), 20)) as executor:
        future_to_prompt = {executor.submit(call_gpt, prompt): idx
                           for idx, prompt in enumerate(prompts)}

        # Collect results as they complete, maintaining order
        results = [None] * len(prompts)
        with tqdm(total=len(prompts), desc="Checklist items", disable=len(prompts) < 5, leave=False) as pbar:
            for future in as_completed(future_to_prompt):
                idx = future_to_prompt[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"
                    logger.error(f"Checklist item {idx} failed: {e}")
                pbar.update(1)

        responses = results

    # Parse responses and get coverage score
    coverage_score = 0.0
    covered_points = 0
    total_weights = sum(item["weight"] for item in checklist if item["weight"] > 0)
    checklist_details = []
    for idx, response in enumerate(responses):
        response_dict = safe_load_json(response, idx, report_obj["id"])
        score, justification = response_dict["score"], response_dict["justification"]

        coverage_score += score * checklist[idx]["weight"]

        if (checklist[idx]["weight"] >= 0 and score == 1.0) or (checklist[idx]["weight"] < 0 and score == 0.0):
            covered_points += 1
        checklist_details.append({
            "id": checklist[idx]["id"],
            "item": checklist[idx]["item"],
            "weight": checklist[idx]["weight"],
            "covered": score,
            "justification": justification
        })
    coverage_score /= total_weights

    # Placeholder return structure
    return {
        "id": report_obj["id"],
        "checklist_evaluator_model": model_name(evaluator_model),
        "coverage_score": coverage_score,
        "total_points": len(checklist),
        "covered_points": covered_points,
        "checklist_details": checklist_details
    }


def aggregate_checklist_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate checklist metrics across multiple samples.
    
    Args:
        metrics: List of sample-level checklist metric objects
    
    Returns:
        Dict with aggregated metrics (e.g., average coverage_score)
    """
    if not metrics:
        return {}

    avg_coverage_score = sum(metric["coverage_score"] for metric in metrics) / len(metrics)
    avg_covered_points = sum(metric["covered_points"] for metric in metrics) / len(metrics)
    avg_total_points = sum(metric["total_points"] for metric in metrics) / len(metrics)
    avg_covered_percentage = sum(metric["covered_points"] / metric["total_points"] for metric in metrics) / len(metrics) if avg_total_points > 0 else 0.0
    
    return {
        "avg_coverage_score": avg_coverage_score,
        "avg_covered_points": avg_covered_points,
        "avg_total_points": avg_total_points,
        "avg_covered_percentage": avg_covered_percentage,
        "total_samples": len(metrics),
    }

