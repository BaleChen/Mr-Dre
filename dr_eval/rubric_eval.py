"""
Rubric-based evaluation module.
Evaluates reports against rubrics (currently only presentation rubric).
"""

import json
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dr_eval.eval_prompts import (
    get_rubric_eval_prompts,
    get_rubric_checklist_items,
)
from utils import model_name

logger = logging.getLogger(__name__)


def evaluate_rubric(
    report_obj: Dict[str, Any],
    dataset_item: Dict[str, Any],
    judge_model: Any,
    rubric_dimensions: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a report against quality rubrics.
    
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
        dataset_item: Dataset item (not used for rubrics, but kept for API consistency)
        judge_model: Model to use for rubric evaluation
        rubric_dimensions: List of rubric dimensions to evaluate (default: ["organization"])
        **kwargs: Additional arguments
    
    Returns:
        Dict with structure:
        {
            "id": int,
            "judge_model": str,
            "rubric_dims": List[str],
            "organization": float,
            "organization_details": List[Dict]
        }
    """
    if rubric_dimensions is None:
        rubric_dimensions = ["organization"]
    
    question = report_obj.get("question", "")
    report = report_obj.get("full_response", {}).get("report", "")
    
    result = {
        "id": report_obj.get("id"),
        "judge_model": model_name(judge_model),
        "rubric_dims": rubric_dimensions,
    }
    
    # Evaluate each rubric dimension
    for dim in rubric_dimensions:
        checklist = get_rubric_checklist_items(dim)
        prompts = get_rubric_eval_prompts(question, report, dim)
        
        # Multi-threaded GPT calls
        call_gpt = lambda prompt: judge_model(prompt)
        responses = []
        with ThreadPoolExecutor(max_workers=min(len(prompts), 10)) as executor:
            future_to_prompt = {executor.submit(call_gpt, prompt): idx
                               for idx, prompt in enumerate(prompts)}

            # Collect results as they complete, maintaining order
            results = [None] * len(prompts)
            with tqdm(total=len(prompts), desc=f"Rubric: {dim}", disable=len(prompts) < 5, leave=False) as pbar:
                for future in as_completed(future_to_prompt):
                    idx = future_to_prompt[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"Error: {str(e)}"
                        logger.error(f"Rubric {dim} item {idx} failed: {e}")
                    pbar.update(1)

            responses = results
        
        # Parse responses and calculate dimension score (equal weights)
        scores = []
        dim_details = []
        
        for idx, response in enumerate(responses):
            try:
                response_dict = json.loads(response)
                score = response_dict["score"]
                justification = response_dict["justification"]
            except (json.JSONDecodeError, KeyError) as e:
                score = 0.0
                justification = f"Error parsing response: {str(e)}"
                logger.error(f"Rubric {dim} item {idx} parse error: {e}")

            scores.append(score)
            dim_details.append({
                "id": idx,
                "item": checklist[idx],
                "covered": score,
                "justification": justification
            })
        
        # delete all the -1 entries in the scores
        scores = [score for score in scores if score != -1]

        dim_score = sum(scores) / len(scores) if scores else 0.0
        
        result[dim] = dim_score
        result[f"{dim}_details"] = dim_details
    
    return result


def aggregate_rubric_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate rubric metrics across multiple samples.
    
    Args:
        metrics: List of sample-level rubric metric objects
    
    Returns:
        Dict with aggregated metrics (e.g., average scores per dimension)
    """
    if not metrics:
        return {}
    
    # Get all rubric dimensions from the first metric
    rubric_dims = metrics[0].get("rubric_dims", [])
    
    aggregated = {
        "total_samples": len(metrics),
    }
    
    # Aggregate scores for each dimension
    for dim in rubric_dims:
        dim_scores = [metric.get(dim, 0.0) for metric in metrics if dim in metric]
        if dim_scores:
            aggregated[f"avg_{dim}"] = sum(dim_scores) / len(dim_scores)
        else:
            aggregated[f"avg_{dim}"] = 0.0
    
    return aggregated

