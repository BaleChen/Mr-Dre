import json
import os
import argparse
import logging
import random
import time
from typing import Dict, Any
import sys
from tqdm import tqdm
from dotenv import load_dotenv

from utils import load_jsonl, save_jsonl, append_jsonl, get_file_path
from engine import DRA
from engine.reviser_agent import ReviserAgent
from feedback import generate_feedback_batch, generate_formal_feedback_batch
from feedback_refine import refine_feedback, load_feedback, load_gen_files

load_dotenv()

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "1800"))
POLL_INTERVAL_SECONDS   = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
BATCH_TIMEOUT_MINUTES   = int(os.getenv("BATCH_TIMEOUT_MINUTES", "300"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    force=True,               
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx (used by OpenAI SDK)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_feedbacks(
    base_output_dir: str,
    model_name: str,
    current_round_type: str,
    current_round: int,
    needed_qids,
    questions_by_id: Dict[str, str],
    num_items: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve feedback for a set of question IDs. Supports both checklist-based and formal feedback.

    Returns a mapping from question ID (as string) to a feedback dict:
        {qid_str -> {"feedback_text": ..., "targeted_item": [...], "targeted_id": [...]}}

    Parameters:
    - base_output_dir: Base directory for all outputs of the given dataset
    - model_name: Name of the DRA model
    - current_round_type: Type of the current round ("checklist_feedback" or "formal_feedback")
    - current_round: Index of the current round (starting from 1)
    - needed_qids: Collection of question IDs that need feedback
    - questions_by_id: Mapping from question ID to original question text
    - num_items: Number of checklist evaluation points (only for checklist_feedback)

    Returns:
    Mapping from qid string to feedback dict for all IDs with available feedback
    """
    
    feedbacks: Dict[str, Dict[str, Any]] = {}

    prev_round = current_round - 1
    if prev_round == 1:
        prev_round_type = "init"

    else:
        prev_round_type = current_round_type
        # Add k-value suffix to prev_round_type if needed
        if num_items is not None and prev_round != 1:
            # Check if prev_round_type is a refined type
            if prev_round_type.startswith('refined_'):
                # Extract original type
                original_type = prev_round_type[len('refined_'):]
                # Only add k-value suffix if original type needs it
                if original_type == 'checklist_feedback':
                    prev_round_type = f"{prev_round_type}_k={num_items}"
            elif prev_round_type == 'checklist_feedback':
                # For non-refined specific types, add k-value suffix if needed
                prev_round_type = f"{prev_round_type}_k={num_items}"
    
    # -------------------- Handle different feedback types --------------------
    if current_round_type == "checklist_feedback":
        # -------------------- Load existing checklist feedback file --------------------
        feedback_file = os.path.join(
            base_output_dir,
            prev_round_type,
            model_name,
            f"checklist_feedback_k={num_items}_turn{prev_round}.jsonl",
        )

        if os.path.exists(feedback_file):
            try:
                feedback_data = load_jsonl(feedback_file)
                for item in feedback_data:
                    qid_str = str(item["id"])
                    feedbacks[qid_str] = {
                        "feedback_text": item.get("feedback", ""),
                        "targeted_item": item.get("item", []),
                        "targeted_id": item.get("item_id", []),
                    }
            except Exception as e:
                logger.warning(f"Failed to load feedback file {feedback_file}: {e}")
        else:
            logger.info(f"Feedback file not found for round {prev_round}: {feedback_file}")

        # -------------------- Compute missing qids --------------------
        needed_qids = [str(q) for q in needed_qids]
        needed_set = set(needed_qids)
        missing_qids = sorted(list(needed_set - set(feedbacks.keys())))

        if not missing_qids:
            # All required feedback already exists.
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        # -------------------- Load checklist eval from previous round --------------------
        eval_filename = f"checklist_turn{prev_round}.jsonl"
        eval_file = os.path.join(base_output_dir, prev_round_type, model_name, eval_filename)
        if not os.path.exists(eval_file):
            logger.warning(f"Eval file not found: {eval_file} — cannot generate feedback")
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        logger.info(f"Generating feedback for {len(missing_qids)} questions from {os.path.basename(eval_file)}")

        eval_map: Dict[str, Dict[str, Any]] = {}
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = item.get("id")
                if qid is None:
                    continue
                eval_map[str(qid)] = item

        # -------------------- Generate missing feedback via feedback.py --------------------
        new_feedbacks = generate_feedback_batch(
            qids=missing_qids,
            eval_map=eval_map,
            questions_by_id=questions_by_id,
            num_points=num_items,
        )

        # Update in-memory dictionary
        for item in new_feedbacks:
            qid_str = str(item["id"])
            feedbacks[qid_str] = {
                "feedback_text": item.get("feedback", ""),
                "targeted_item": item.get("item", []),
                "targeted_id": item.get("item_id", []),
            }

        # Append the complete new feedback items back to the feedback file (JSONL)
        for item in new_feedbacks:
            append_jsonl(item, feedback_file)

        logger.info(f"Saved {len(new_feedbacks)} feedback items → {os.path.basename(feedback_file)}")

        # Return only the subset needed for this round
        return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}
        
    elif current_round_type == "formal_feedback":
        # -------------------- Load existing formal feedback file --------------------
        feedback_file = os.path.join(
            base_output_dir,
            prev_round_type,
            model_name,
            f"formal_feedback_turn{prev_round}.jsonl",
        )

        if os.path.exists(feedback_file):
            try:
                feedback_data = load_jsonl(feedback_file)
                for item in feedback_data:
                    qid_str = str(item["id"])
                    feedbacks[qid_str] = {
                        "feedback_text": item.get("feedback", ""),
                        "targeted_item": item.get("item", []),
                        "targeted_id": item.get("item_id", []),
                    }
            except Exception as e:
                logger.warning(f"Failed to load feedback file {feedback_file}: {e}")
        else:
            logger.info(f"Feedback file not found for round {prev_round}: {feedback_file}")

        # -------------------- Compute missing qids --------------------
        needed_qids = [str(q) for q in needed_qids]
        needed_set = set(needed_qids)
        missing_qids = sorted(list(needed_set - set(feedbacks.keys())))

        if not missing_qids:
            # All required feedback already exists.
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        # -------------------- Load reports from previous round --------------------
        # For formal feedback, we need the actual reports from the previous round
        report_file = get_file_path(base_output_dir, model_name, prev_round, prev_round_type)
        if not os.path.exists(report_file):
            logger.warning(f"Report file not found: {report_file} — cannot generate formal feedback")
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        logger.info(f"Generating formal feedback for {len(missing_qids)} questions from {os.path.basename(report_file)}")

        reports_by_id: Dict[str, str] = {}
        try:
            report_data = load_file(report_file)
            for item in report_data:
                qid = item.get("id")
                if qid is None:
                    continue
                # Get the report content from full_response.report (reference: line 374-416)
                full_response = item.get("full_response")
                report = full_response['report'] if full_response else None
                if report:
                    reports_by_id[str(qid)] = report
        except Exception as e:
            logger.warning(f"Failed to load report file {report_file}: {e}")
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        # -------------------- Generate missing formal feedback --------------------
        new_feedbacks = generate_formal_feedback_batch(
            qids=missing_qids,
            reports_by_id=reports_by_id,
            questions_by_id=questions_by_id,
        )

        # Update in-memory dictionary
        for item in new_feedbacks:
            qid_str = str(item["id"])
            feedbacks[qid_str] = {
                "feedback_text": item.get("feedback", ""),
                "targeted_item": item.get("item", []),
                "targeted_id": item.get("item_id", []),
            }

        # Append the complete new feedback items back to the feedback file (JSONL)
        for item in new_feedbacks:
            append_jsonl(item, feedback_file)

        logger.info(f"Saved {len(new_feedbacks)} formal feedback items → {os.path.basename(feedback_file)}")

        # Return only the subset needed for this round
        return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}
        

    elif current_round_type.startswith("refined_"):
        # -------------------- Extract original feedback type --------------------
        original_type = current_round_type[len("refined_"):]
        
        # -------------------- Load existing refined feedback file --------------------
        # Determine if we need to add k-value suffix
        if original_type == "checklist_feedback":
            feedback_file = os.path.join(
                base_output_dir,
                prev_round_type,
                model_name,
                f"refined_{original_type}_k={num_items}_turn{prev_round}.jsonl",
            )
        elif original_type == "formal_feedback":
            feedback_file = os.path.join(
                base_output_dir,
                prev_round_type,
                model_name,
                f"refined_{original_type}_turn{prev_round}.jsonl",
            )
        else:
            logger.warning(f"Unsupported original feedback type for refinement: {original_type}")
            return {}

        if os.path.exists(feedback_file):
            try:
                feedback_data = load_jsonl(feedback_file)
                for item in feedback_data:
                    qid_str = str(item["id"])
                    feedbacks[qid_str] = {
                        "feedback_text": item.get("feedback", ""),
                        "targeted_item": item.get("item", []),
                        "targeted_id": item.get("item_id", []),
                    }
            except Exception as e:
                logger.warning(f"Failed to load refined feedback file {feedback_file}: {e}")
        else:
            logger.info(f"Refined feedback file not found for round {prev_round}: {feedback_file}")

        # -------------------- Compute missing qids --------------------
        needed_qids = [str(q) for q in needed_qids]
        needed_set = set(needed_qids)
        missing_qids = sorted(list(needed_set - set(feedbacks.keys())))

        if not missing_qids:
            # All required feedback already exists.
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        # -------------------- Load reports from previous round --------------------
        # For refined feedback, we need the actual reports from the previous round
        report_file = get_file_path(base_output_dir, model_name, prev_round, prev_round_type)
        if not os.path.exists(report_file):
            logger.warning(f"Report file not found: {report_file} — cannot generate refined feedback")
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        logger.info(f"Generating refined feedback for {len(missing_qids)} questions from {os.path.basename(report_file)}")

        # Load original feedback based on type
        # Determine if we need to add k-value suffix
        if original_type == "checklist_feedback":
            original_feedback_file = os.path.join(
                base_output_dir,
                prev_round_type,
                model_name,
                f"{original_type}_k={num_items}_turn{prev_round}.jsonl",
            )
        elif original_type == "formal_feedback":
            original_feedback_file = os.path.join(
                base_output_dir,
                prev_round_type,
                model_name,
                f"{original_type}_turn{prev_round}.jsonl",
            )
        else:
            logger.warning(f"Unsupported original feedback type for refinement: {original_type}")
            return {}

        if not os.path.exists(original_feedback_file):
            logger.warning(f"Original feedback file not found: {original_feedback_file} — cannot generate refined feedback")
            return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}

        original_feedbacks = load_feedback(original_feedback_file)
        feedback_map = {str(f["id"]): f for f in original_feedbacks}

        # Load gen files for reports
        gen_dir = os.path.join(base_output_dir, prev_round_type, model_name)
        gen_map = load_gen_files(gen_dir)

        # -------------------- Generate missing refined feedback --------------------
        new_feedbacks = []
        for qid in tqdm(missing_qids, desc="Generating refined feedback", unit="qid"):
            if qid in feedback_map and qid in gen_map:
                original_feedback = feedback_map[qid]
                gen_entry = gen_map[qid]
                try:
                    refined_entry = refine_feedback(original_feedback, gen_entry)
                    new_feedbacks.append(refined_entry)
                except Exception as e:
                    logger.error(f"Failed to refine feedback for Q{qid}: {e}")

        # Update in-memory dictionary
        for item in new_feedbacks:
            qid_str = str(item["id"])
            feedbacks[qid_str] = {
                "feedback_text": item.get("feedback", ""),
                "targeted_item": item.get("item", []),
                "targeted_id": item.get("item_id", []),
            }

        # Append the complete new feedback items back to the feedback file (JSONL)
        for item in new_feedbacks:
            append_jsonl(item, feedback_file)

        logger.info(f"Saved {len(new_feedbacks)} refined feedback items → {os.path.basename(feedback_file)}")

        # Return only the subset needed for this round
        return {qid: feedbacks[qid] for qid in needed_qids if qid in feedbacks}
    else:
        logger.warning(f"Unsupported feedback type: {current_round_type}")
        return {}


def construct_conversations(
    base_output_dir: str,
    model_name: str,
    current_round: int,
    current_round_type: str,
    questions,
    num_items: int = 1,
):
    """
    Construct conversations for all questions in the current round.

    Builds conversation history for each question ID by aggregating:
    - Original user question
    - Assistant's reports from previous rounds
    - User signals/feedback from previous rounds
    - Current round's signal (reflection, checklist_feedback, or formal_feedback)

    Parameters:
    - base_output_dir: Base directory for all outputs of the given dataset
    - model_name: Name of the DRA model
    - current_round: Index of the current round (starting from 1)
    - current_round_type: Type of the current round ("init", "reflection", "checklist_feedback", "formal_feedback")
    - questions: List of question objects from the dataset
    - num_items: Number of checklist items (only for checklist_feedback)

    Returns:
    - conversations: Mapping from question ID to list of messages
    - signals: Mapping from question ID to the signal used in this round
    """
    
    conversations: Dict[str, Any] = {}
    signals: Dict[str, Dict[str, Any]] = {}

    # 1. Initialize: add the original question for each question
    for q in questions:
        qid = str(q["id"])
        conversations[qid] = [
            {
                "role": "user",
                "content": q["question"],
            }
        ]

    # 2. If current_round > 1, load history files round by round and append
    if current_round > 1:
        for rnd in range(1, current_round):
            prev_file = get_file_path(
                base_output_dir,
                model_name,
                rnd,
                current_round_type,
                num_items,
            )

            if not os.path.exists(prev_file):
                logger.warning(f"Previous round file not found: {prev_file}")
                return {}, {}

            try:
                history_data = load_jsonl(prev_file)
            except Exception as e:
                logger.warning(f"Failed to load previous round file {prev_file}: {e}")
                return {}, {}

            # Create a mapping from id -> item for faster lookup
            history_id2item: Dict[str, Any] = {}
            for item in history_data:
                qid_item = str(item.get("id"))
                if qid_item is not None:
                    history_id2item[qid_item] = item

            to_delete = []
            # Append previous round content to each question's conversation
            for qid in list(conversations.keys()):
                conv = conversations[qid]
                qdata = history_id2item.get(qid)
                if not qdata:
                    logger.warning(f"Q{qid} not found in round {rnd} file, skipping")
                    to_delete.append(qid)
                    continue

                signal_obj = qdata.get("signal") or {}
                signal_prev = signal_obj.get("feedback_text")
                full_response = qdata.get("full_response")
                report = full_response['report'] if full_response else None

                if rnd == 1:
                    # Round 1: only append the assistant report
                    if report:
                        conv.append({
                            "role": "assistant",
                            "content": report,
                        })
                    else:
                        logger.warning(f"Q{qid} missing report in round {rnd}, skipping")
                        to_delete.append(qid)
                        continue
                else:
                    # Round 2+: append user(signal_k) then assistant(report_k)
                    if signal_prev:
                        conv.append({
                            "role": "user",
                            "content": signal_prev,
                        })
                    else:
                        logger.warning(f"Q{qid} missing signal in round {rnd}, skipping")
                        to_delete.append(qid)
                        continue

                    if report:
                        conv.append({
                            "role": "assistant",
                            "content": report,
                        })
                    else:
                        logger.warning(f"Q{qid} missing report in round {rnd}, skipping")
                        to_delete.append(qid)
                        continue

            # Remove any questions that could not be reconstructed
            for qid in to_delete:
                conversations.pop(qid, None)

    # 3. Append the current round signal, determined by current_round_type
    if current_round_type == "reflection":
        for qid, conv in conversations.items():
            feedback_text = (
                "Reflect on your current report and refine it. "
                "Please only return the refined report without any other text."
            )
            signal = {
                "feedback_text": feedback_text,
                "targeted_item": [],
                "targeted_id": [],
            }
            conv.append({
                "role": "user",
                "content": feedback_text,
            })
            signals[qid] = signal

    elif current_round_type == "formal_feedback":
        # formal_feedback: generate formal feedback using GPT based on previous reports
        needed_qids = list(conversations.keys())
        questions_by_id = {str(q["id"]): q["question"] for q in questions}

        feedbacks = get_feedbacks(
            base_output_dir=base_output_dir,
            model_name=model_name,
            current_round_type=current_round_type,
            current_round=current_round,
            needed_qids=needed_qids,
            questions_by_id=questions_by_id,
            num_items=num_items,
        )

        to_delete = []
        for qid, conv in conversations.items():
            feedback = feedbacks.get(qid)
            if feedback:
                feedback_text = feedback.get("feedback_text", "")
                conv.append({
                    "role": "user",
                    "content": feedback_text,
                })
                signals[qid] = feedback
            else:
                logger.warning(f"Q{qid} missing formal feedback, skipping")
                to_delete.append(qid)

        for qid in to_delete:
            conversations.pop(qid, None)

    elif current_round_type == "checklist_feedback":
        # checklist_feedback: use checklist-based feedback generated from checklist evaluations
        needed_qids = list(conversations.keys())
        questions_by_id = {str(q["id"]): q["question"] for q in questions}

        feedbacks = get_feedbacks(
            base_output_dir=base_output_dir,
            model_name=model_name,
            current_round_type=current_round_type,
            current_round=current_round,
            needed_qids=needed_qids,
            questions_by_id=questions_by_id,
            num_items=num_items,
        )

        to_delete = []
        for qid, conv in conversations.items():
            feedback = feedbacks.get(qid)
            if feedback:
                feedback_text = feedback.get("feedback_text", "")
                conv.append({
                    "role": "user",
                    "content": feedback_text,
                })
                signals[qid] = feedback
            else:
                logger.warning(f"Q{qid} missing checklist feedback, skipping")
                to_delete.append(qid)

        for qid in to_delete:
            conversations.pop(qid, None)
    

    elif current_round_type.startswith("refined_"):
        # -------------------- Load or generate refined feedback --------------------
        needed_qids = list(conversations.keys())
        questions_by_id = {str(q["id"]): q["question"] for q in questions}

        feedbacks = get_feedbacks(
            base_output_dir=base_output_dir,
            model_name=model_name,
            current_round_type=current_round_type,
            current_round=current_round,
            needed_qids=needed_qids,
            questions_by_id=questions_by_id,
            num_items=num_items,
        )

        to_delete = []
        for qid, conv in conversations.items():
            feedback = feedbacks.get(qid)
            if feedback:
                feedback_text = feedback.get("feedback_text", "")
                conv.append({
                    "role": "user",
                    "content": feedback_text,
                })
                signals[qid] = feedback
            else:
                logger.warning(f"Q{qid} missing refined feedback, skipping")
                to_delete.append(qid)

        for qid in to_delete:
            conversations.pop(qid, None)

    elif current_round_type != "init":
        logger.warning(f"Invalid or unsupported round type: '{current_round_type}'")
        return {}, {}

    return conversations, signals


def load_prev_snippet_maps(
    base_output_dir: str,
    model_name: str,
    current_round: int,
    current_round_type: str,
    num_items: int = 1,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load snippet_map from the previous round for dr-tulu models.

    Parameters:
    - base_output_dir: Base directory for all outputs of the given dataset
    - model_name: Name of the DRA model
    - current_round: Index of the current round (starting from 1)
    - current_round_type: Type of the current round
    - num_items: Number of checklist items (used for file path generation)

    Returns:
    Mapping from question ID (string) to snippet_map dict.

    Raises:
    - FileNotFoundError: If the previous round file does not exist
    - RuntimeError: If loading the previous round file fails
    """
    
    prev_round = current_round - 1
    if prev_round == 1:
        prev_round_type = "init"
    else:
        prev_round_type = current_round_type

    # Determine the file path for the previous round
    prev_file = get_file_path(
        base_output_dir,
        model_name,
        prev_round,
        prev_round_type,
        num_items,
    )

    if not os.path.exists(prev_file):
        raise FileNotFoundError(
            f"Previous round file not found: {prev_file}. "
            f"Cannot load snippet_map for dr-tulu model at round {current_round}."
        )

    try:
        prev_data = load_jsonl(prev_file)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load previous round file {prev_file}: {e}"
        ) from e

    prev_snippet_maps: Dict[str, Dict[str, Dict[str, str]]] = {}
    for item in prev_data:
        qid_str = str(item.get("id"))
        if qid_str:
            full_response = item.get("full_response", {})
            snippet_map = full_response.get("snippet_map", {})
            if snippet_map:
                prev_snippet_maps[qid_str] = snippet_map

    logger.info(f"Loaded snippet_map from round {prev_round} for {len(prev_snippet_maps)} questions")
    return prev_snippet_maps


def merge_snippet_maps(
    prev_snippet_map: Dict[str, Dict[str, str]],
    new_snippet_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """
    Merge previous and new snippet_maps, with old entries taking precedence.

    Parameters:
    - prev_snippet_map: Snippet map from the previous round
    - new_snippet_map: Snippet map from the current round

    Returns:
    Merged snippet map where old entries take precedence for duplicate keys
    """
    
    merged = dict(prev_snippet_map)
    for key, value in new_snippet_map.items():
        if key not in merged:
            merged[key] = value
    return merged


def call_dra_api_single_round(
    questions,
    model_name: str,
    base_output_dir: str = None,
    current_round: int = 1,
    current_round_type: str = "init",
    num_items: int = 1,
    batch_size: int = 10,
    refine: bool = False,
    use_reviser: bool = False,
    **kwargs,
):
    """
    Call the Deep Research Agent (DRA) for a single round of processing.

    Functionality:
    - Optionally resumes from an existing output file
    - Constructs conversations for each question
    - Issues asynchronous calls to the DRA
    - Waits for completion of all requests
    - Persists results in JSONL format

    Parameters:
    - questions: List of question objects from the dataset
    - model_name: Name of the DRA model to use
    - base_output_dir: Base directory for outputs
    - current_round: Index of the current round (starting from 1)
    - current_round_type: Type of the current round ("init", "reflection", "checklist_feedback", "formal_feedback")
    - num_items: Number of checklist items (only for checklist_feedback)
    - batch_size: Number of requests per batch (default: 10)
    - refine: Whether to refine feedback
    - use_reviser: Whether to use ReviserAgent for report revision
    - kwargs: Additional arguments passed to DRA constructor

    Returns:
    Tuple containing:
    - List of result objects for all processed questions
    - List of question IDs that timed out
    - List of question IDs that failed due to errors
    """
    
    # Initialize Agent
    if use_reviser:
        agent = ReviserAgent()
        logger.info("Using ReviserAgent for report revision")
    else:
        agent = DRA(model_name=model_name, **kwargs)

    # Track timed out and failed requests
    timed_out_qids = []
    failed_qids = []

    # Modify current_round_type if refining is enabled
    if refine:
        current_round_type = f"refined_{current_round_type}"

    # Current round output file
    output_file = get_file_path(base_output_dir, model_name, current_round, current_round_type, num_items, use_reviser)

    # Support resuming from checkpoint
    current_results = []
    if os.path.exists(output_file):
        try:
            current_results = load_jsonl(output_file)
            logger.info(f"Resuming: {len(current_results)} completed from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            current_results = []

    # Load previous round's snippet_map only for dr-tulu models (if current_round > 1)
    is_dr_tulu = "tulu" in model_name.lower()
    prev_snippet_maps = {}
    if is_dr_tulu and current_round > 1:
        prev_snippet_maps = load_prev_snippet_maps(
            base_output_dir=base_output_dir,
            model_name=model_name,
            current_round=current_round,
            current_round_type=current_round_type,
            num_items=num_items,
        )

    # Build conversations for all questions
    conversations, signals = construct_conversations(
        base_output_dir=base_output_dir,
        model_name=model_name,
        current_round=current_round,
        current_round_type=current_round_type,
        questions=questions,
        num_items=num_items,
    )
    
    # Filter pending questions (not processed and have conversation)
    pending_questions = []
    skipped_ids = []
    for question_data in questions:
        question_id = question_data["id"]
        qid_str = str(question_id)
        
        # Skip if already processed
        processed = False
        for item in current_results:
            if str(item.get("id")) == qid_str and "error" not in item and "full_response" in item:
                processed = True
                break
        if processed:
            skipped_ids.append(str(question_id))
            continue
        
        # Skip if no conversation
        if conversations.get(qid_str) is None:
            logger.warning(f"Q{question_id} has no conversation, skipping")
            continue
        
        pending_questions.append(question_data)
    
    if skipped_ids:
        logger.info(f"[SKIP] {len(skipped_ids)} already done - Q[{', '.join(skipped_ids)}]")
    logger.info(f"Processing {len(pending_questions)} questions (round {current_round}, {current_round_type})")
    
    # Process questions in batches
    total_batches = (len(pending_questions) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(pending_questions), batch_size):
        batch = pending_questions[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"Batch {batch_num}/{total_batches} — submitting {len(batch)} requests")
        
        # Submit batch requests
        batch_requests = {}
        for question_data in batch:
            question_id = question_data["id"]
            qid_str = str(question_id)
            conversation_text = conversations.get(qid_str)
            signal = signals.get(qid_str)
            
            # Asynchronous request
            if use_reviser:
                # Extract question, report, and feedback for ReviserAgent
                question = question_data["question"]
                # Get the last assistant message as the report to revise
                report = ""
                for msg in reversed(conversation_text[:-1]):  # Exclude the last feedback message
                    if msg["role"] == "assistant":
                        report = msg["content"]
                        break
                feedback = signal.get("feedback_text", "") if signal else ""
                request_id = agent(question, report, feedback)
            else:
                request_id = agent(conversation_text)
            logger.info(f"Q{question_id} → request {request_id}")
            
            batch_requests[request_id] = {
                "question_data": question_data,
                "signal": signal,
                "start_time": time.time(),
            }
        
        # Use polling mechanism to wait for batch completion
        pending = batch_requests.copy()
        poll_interval = POLL_INTERVAL_SECONDS
        batch_timeout_minutes = BATCH_TIMEOUT_MINUTES
        batch_start_time = time.time()
        total_batch_requests = len(pending)
        completed_batch_requests = 0
        
        # Create progress bar for this batch
        pbar = tqdm(total=total_batch_requests, desc=f"Batch {batch_num}", unit="req", leave=True)
        
        while pending:
            batch_elapsed = (time.time() - batch_start_time) / 60
            if batch_elapsed > batch_timeout_minutes:
                pending_ids = [str(m["question_data"]["id"]) for m in pending.values()]
                logger.warning(f"[BATCH TIMEOUT] Batch {batch_num} timeout after {batch_elapsed:.1f}m, skipping Q[{', '.join(pending_ids)}]")
                break
            
            to_remove = []
            
            for request_id, meta in pending.items():
                question_data = meta["question_data"]
                question_id = question_data["id"]
                elapsed_seconds = time.time() - meta.get("start_time", time.time())
                elapsed = elapsed_seconds / 60
                
                # Check for timeout
                if elapsed_seconds > REQUEST_TIMEOUT_SECONDS:
                    logger.warning(f"Q{question_id} timed out after {elapsed:.1f} minutes")
                    timed_out_qids.append(question_id)
                    to_remove.append(request_id)
                    continue
                
                try:
                    # Use poll instead of wait_for_completion to avoid blocking
                    is_completed, response = agent.poll(request_id)
                    
                    if is_completed:
                        logger.info(f"[DONE] Q{question_id} completed ({elapsed:.1f}m)")
                        signal = meta["signal"]
                        
                        # Merge snippet_map from previous round for dr-tulu
                        if is_dr_tulu and isinstance(response, dict):
                            qid_str = str(question_id)
                            prev_snippet_map = prev_snippet_maps.get(qid_str, {})
                            new_snippet_map = response.get("snippet_map", {})
                            response["snippet_map"] = merge_snippet_maps(prev_snippet_map, new_snippet_map)
                        
                        # Check if metadata exists in question_data
                        metadata = question_data.get("metadata", {})
                        result = {
                            "id": question_id,
                            "question": question_data["question"],
                            "question_metadata": metadata,
                            "signal": signal,
                            "full_response": response,
                        }
                        
                        # Update or append
                        updated = False
                        for i, item in enumerate(current_results):
                            if str(item.get("id")) == str(question_id):
                                current_results[i] = {"id": question_id, **result}
                                updated = True
                                break
                        if not updated:
                            current_results.append(result)
                        
                        completed_batch_requests += 1
                        pbar.update(1)
                        pbar.set_postfix_str(f"Q{question_id} done ({elapsed:.1f}m)")
                        
                        output_dir = os.path.dirname(output_file)
                        os.makedirs(output_dir, exist_ok=True)
                        save_jsonl(current_results, output_file)
                        
                        logger.info(f"Saved {len(current_results)} results → {os.path.basename(output_file)}")
                        to_remove.append(request_id)
                    else:
                        time.sleep(1)
                
                except ConnectionError as e:
                    # Server connection issue - likely server crashed or overloaded
                    logger.error(f"Q{question_id} connection error: {e}")
                    logger.error(
                        "[SERVER ISSUE] The ODR server may have crashed. "
                        "Consider restarting with: ./scripts/start_odr.sh --clean"
                    )
                    failed_qids.append(question_id)
                    to_remove.append(request_id)
                    # Track consecutive connection errors
                    meta["connection_errors"] = meta.get("connection_errors", 0) + 1
                        
                except RuntimeError as e:
                    # Run failed on server side (error/interrupted/timeout status)
                    logger.error(f"Q{question_id} run failed: {e}")
                    failed_qids.append(question_id)
                    to_remove.append(request_id)
                    
                except Exception as e:
                    logger.error(f"Q{question_id} unexpected error: {e}")
                    failed_qids.append(question_id)
                    to_remove.append(request_id)
            
            # Remove completed requests
            for rid in to_remove:
                del pending[rid]
            
            # Sleep before next poll if there are still pending requests
            if pending:
                pending_ids = [str(m["question_data"]["id"]) for m in pending.values()]
                logger.info(f"[POLL] {len(pending)} pending - Q[{', '.join(pending_ids)}]")
                time.sleep(poll_interval)
        
        pbar.close()
        time.sleep(60)
        logger.info(f"Batch {batch_num} completed, sleeping for 60 seconds to avoid rate limiting")

    # Log timeout and failure summary at the end
    if timed_out_qids:
        logger.warning(f"Total timeouts: {len(timed_out_qids)}")
        logger.warning(f"Timed out question IDs: {timed_out_qids}")
    if failed_qids:
        logger.warning(f"Total failures: {len(failed_qids)}")
        logger.warning(f"Failed question IDs: {failed_qids}")

    return current_results, timed_out_qids, failed_qids


def main():
    parser = argparse.ArgumentParser(description="Deep Research Agent Processing")
    parser.add_argument("--round", type=int, required=True, help="Current round number (1, 2, 3, ...)")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["init", "reflection", "checklist_feedback", "formal_feedback"],
        help="Round type: init, reflection, checklist_feedback, formal_feedback.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--num_questions",
        type=int,
        default=-1,
        help="Number of questions to process (default: -1 for all)",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=1,
        help="Number of checklist items to get feedback in each question "
             "(only used in checklist_feedback mode; default: 1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of questions to process in each batch (default: 10)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether to refine the feedback (only applicable for checklist_feedback or formal_feedback types)",
    )
    parser.add_argument(
        "--use_reviser",
        action="store_true",
        help="Use ReviserAgent to revise reports based on feedback instead of "
             "sending feedback as another conversation turn to DRA.",
    )

    args = parser.parse_args()

    base_output_dir = f"output/{args.dataset}"
    question_data = f"data/{args.dataset}.jsonl"

    questions = load_jsonl(question_data)
    logger.info(f"Loaded {len(questions)} questions from {args.dataset}")

    # Limit number of questions if specified
    if args.num_questions > 0:
        random.seed(42)
        random.shuffle(questions)
        questions = questions[:args.num_questions]
        logger.info(f"Sampled {len(questions)} questions (seed=42)")


    try:
        results, timed_out_qids, failed_qids = call_dra_api_single_round(
            questions=questions,
            model_name=args.model,
            base_output_dir=base_output_dir,
            current_round=args.round,
            current_round_type=args.type,
            num_items=args.num_items,
            batch_size=args.batch_size,
            use_reviser=args.use_reviser,
            refine=args.refine,
        )

        logger.info(f"✓ Completed round {args.round}: {len(results)} questions processed")

    except Exception as e:
        logger.exception(f"Error during processing: {e}")


if __name__ == "__main__":
    main()