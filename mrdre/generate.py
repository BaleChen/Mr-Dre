"""
Report generation: build each question's conversation from the earlier turns and the
new feedback, send it to an engine in batches, and write one report record per question.
"""

import importlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from mrdre.feedback import feedback_signal
from mrdre.utils import append_jsonl, load_partial_jsonl, save_jsonl

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "1800"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
BATCH_TIMEOUT_MINUTES = int(os.getenv("BATCH_TIMEOUT_MINUTES", "300"))
# Pause between batches to stay under the providers' rate limits
BATCH_PAUSE_SECONDS = 60

Messages = List[Dict[str, str]]


class FunctionAgent:
    """
    Run a user's function behind the submit-and-poll interface of the engines.

    The function takes the message list of one question and returns the report, either
    as a string or as a dict with a "report" key plus any JSON-serializable fields to keep.
    Each call runs in a background thread, so a call that hangs past the request timeout
    does not keep the command from finishing.
    """

    def __init__(self, function: Callable[[Messages], Any], name: str):
        self.function = function
        self.name = name
        self.results: Dict[str, Tuple[bool, Any]] = {}

    def __call__(self, messages: Messages) -> str:
        request_id = uuid.uuid4().hex
        threading.Thread(target=self._run, args=(request_id, messages), daemon=True).start()
        return request_id

    def _run(self, request_id: str, messages: Messages) -> None:
        try:
            self.results[request_id] = (True, self.function(messages))
        except Exception as e:
            self.results[request_id] = (False, e)

    def poll(self, request_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if request_id not in self.results:
            return False, None
        succeeded, result = self.results.pop(request_id)
        if not succeeded:
            raise result
        if isinstance(result, str):
            result = {"report": result}
        if not isinstance(result, dict) or not isinstance(result.get("report"), str):
            raise ValueError(
                f"{self.name} must return the report as a string or as a dict with a 'report' "
                f"string, not {type(result).__name__}"
            )
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{self.name} returned a dict that cannot be saved as JSON: {e}") from e
        return True, result


def load_engine(name: str):
    """
    Create the engine for --engine.

    Args:
        name: A built-in model name (e.g. "o3-deep-research", "sonar-deep-research",
            "dr-tulu"), or "module:function" for a function in the user's code, where
            module is a module importable from the working directory or a path to a .py file

    Returns:
        An engine with the submit-and-poll interface of DeepResearchAgentBase
    """
    if ":" not in name:
        from mrdre.engine import DRA
        return DRA(model_name=name)

    module_name, function_name = name.rsplit(":", 1)
    try:
        if module_name.endswith(".py"):
            # Like `python path/to/agent.py`: the file's folder is on the import path
            sys.path.insert(0, str(Path(module_name).resolve().parent))
            spec = importlib.util.spec_from_file_location(Path(module_name).stem, module_name)
            if spec is None:
                raise ImportError(f"cannot import {module_name}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Import the user's module from the working directory, as `python -m` would
            sys.path.insert(0, os.getcwd())
            module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError, TypeError, OSError) as e:
        raise ValueError(f"cannot load the engine function {name}: {e}") from e
    return FunctionAgent(function, name)


def build_conversations(
    questions: List[Dict[str, Any]],
    previous_turns: Sequence[List[Dict[str, Any]]] = (),
    feedback_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Messages], Dict[str, Optional[Dict[str, Any]]]]:
    """
    Build the conversation of each question.

    The conversation starts with the question. Each earlier turn adds its report as an
    assistant message, preceded from turn 2 on by the feedback that turn answered (the
    "signal" of its record). A revision turn ends with the new feedback as a user
    message. Questions missing from an earlier turn or from the feedback are skipped.

    Args:
        questions: Dataset items
        previous_turns: Report records of each earlier turn, oldest first
        feedback_records: Feedback on the latest turn; None for the first turn

    Returns:
        Conversations and the signals of the new feedback (None for the first turn),
        both indexed by question ID as a string
    """
    if previous_turns and any(record.get("signal") for record in previous_turns[0]):
        raise ValueError(
            "the first earlier turn holds revised reports (its records answer feedback in 'signal'); "
            "pass the reports of all earlier turns, oldest first"
        )
    conversations = {str(q["id"]): [{"role": "user", "content": q["question"]}] for q in questions}

    for turn, records in enumerate(previous_turns, start=1):
        records_by_id = {str(record["id"]): record for record in records}
        for qid in list(conversations):
            record = records_by_id.get(qid)
            report = record["full_response"].get("report") if record else None
            feedback_text = (record.get("signal") or {}).get("feedback_text") if record else None
            if not report or (turn > 1 and not feedback_text):
                missing = "report" if not report else "feedback (signal)"
                logger.warning(f"Q{qid} has no {missing} in turn {turn}, skipping")
                del conversations[qid]
                continue
            if turn > 1:
                conversations[qid].append({"role": "user", "content": feedback_text})
            conversations[qid].append({"role": "assistant", "content": report})

    signals: Dict[str, Optional[Dict[str, Any]]] = {qid: None for qid in conversations}
    if feedback_records is not None:
        feedback_by_id = {str(item["id"]): item for item in feedback_records}
        for qid in list(conversations):
            if qid not in feedback_by_id:
                logger.warning(f"Q{qid} has no feedback, skipping")
                del conversations[qid], signals[qid]
                continue
            signals[qid] = feedback_signal(feedback_by_id[qid])
            conversations[qid].append({"role": "user", "content": signals[qid]["feedback_text"]})
    return conversations, signals


def merge_snippet_maps(
    prev_snippet_map: Dict[str, Dict[str, str]],
    new_snippet_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """
    Merge previous and new snippet_maps, with old entries taking precedence.

    Args:
        prev_snippet_map: Snippet map from the previous turn
        new_snippet_map: Snippet map from the current turn

    Returns:
        Merged snippet map where old entries take precedence for duplicate keys
    """
    merged = dict(prev_snippet_map)
    for key, value in new_snippet_map.items():
        if key not in merged:
            merged[key] = value
    return merged


def generate_reports(
    agent: Any,
    questions: List[Dict[str, Any]],
    output_file: Path,
    previous_turns: Sequence[List[Dict[str, Any]]] = (),
    feedback_records: Optional[List[Dict[str, Any]]] = None,
    use_reviser: bool = False,
    batch_size: int = 10,
    overwrite: bool = False,
) -> List[Any]:
    """
    Generate one turn of reports and append one record per question to output_file.

    Each record holds "id", "question", "question_metadata", "signal" (the feedback the
    report answers; None in the first turn) and "full_response" (the engine's response,
    with the report under "report"). Questions that already have a record in output_file
    are skipped, so an interrupted run resumes where it stopped.

    Args:
        agent: Engine from load_engine, or a ReviserAgent
        questions: Dataset items to generate reports for
        output_file: Output JSONL file
        previous_turns: Report records of each earlier turn, oldest first
        feedback_records: Feedback on the latest turn; None for the first turn
        use_reviser: agent is a ReviserAgent, which edits the latest report according
            to the feedback instead of continuing the conversation
        batch_size: Number of requests submitted at a time
        overwrite: Regenerate the reports of these questions that are already in output_file

    Returns:
        IDs of the questions whose requests failed or timed out; running again retries them
    """
    output_file = Path(output_file)
    question_ids = {str(q["id"]) for q in questions}
    existing = load_partial_jsonl(output_file)
    if any("full_response" not in record for record in existing):
        raise ValueError(f"{output_file} holds records that are not generated reports; use a new output file")
    # Records of failed requests ("error") are generated again
    kept = [
        record for record in existing
        if "error" not in record and not (overwrite and str(record["id"]) in question_ids)
    ]
    if len(kept) < len(existing):
        save_jsonl(kept, str(output_file))
    done_ids = {str(record["id"]) for record in kept}
    if done_ids & question_ids:
        logger.info(f"Resuming: {len(done_ids & question_ids)} reports already in {output_file}")

    conversations, signals = build_conversations(
        [q for q in questions if str(q["id"]) not in done_ids], previous_turns, feedback_records
    )
    pending = [q for q in questions if str(q["id"]) in conversations]
    if not pending:
        logger.info(f"No reports to generate in {output_file}")
        return []
    # DR Tulu cites snippet IDs; its revised reports keep citing the snippets of the previous turn
    snippet_maps = {
        str(record["id"]): record["full_response"].get("snippet_map") or {} for record in previous_turns[-1]
    } if previous_turns else {}

    logger.info(f"Generating {len(pending)} reports (turn {len(previous_turns) + 1}) → {output_file}")
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    written, timed_out, failed = 0, [], []
    for batch_number, batch in enumerate(batches, start=1):
        logger.info(f"Batch {batch_number}/{len(batches)}: submitting {len(batch)} requests")
        requests = {}
        for question in batch:
            qid = str(question["id"])
            try:
                if use_reviser:
                    # The latest report is the last assistant message, before the new feedback
                    request_id = agent(question["question"], conversations[qid][-2]["content"],
                                       signals[qid]["feedback_text"])
                else:
                    request_id = agent(conversations[qid])
            except Exception as e:
                logger.error(f"Q{qid} could not be submitted: {e}")
                failed.append(question["id"])
                continue
            logger.info(f"Q{qid} → request {request_id}")
            requests[request_id] = (question, time.time())
        requests_sent = bool(requests)

        batch_start = time.time()
        with tqdm(total=len(requests), desc=f"Batch {batch_number}", unit="req") as pbar:
            while requests:
                if time.time() - batch_start > BATCH_TIMEOUT_MINUTES * 60:
                    ids = [question["id"] for question, _ in requests.values()]
                    logger.warning(f"Batch {batch_number} timed out after {BATCH_TIMEOUT_MINUTES} minutes, skipping Q{ids}")
                    timed_out.extend(ids)
                    break

                for request_id, (question, start_time) in list(requests.items()):
                    qid = str(question["id"])
                    elapsed = time.time() - start_time
                    if elapsed > REQUEST_TIMEOUT_SECONDS:
                        logger.warning(f"Q{qid} timed out after {elapsed / 60:.1f} minutes")
                        timed_out.append(question["id"])
                        del requests[request_id]
                        continue
                    try:
                        completed, response = agent.poll(request_id)
                    except Exception as e:
                        logger.error(f"Q{qid} failed: {e}")
                        failed.append(question["id"])
                        del requests[request_id]
                        continue
                    if not completed:
                        time.sleep(1)
                        continue

                    if isinstance(response, dict) and (snippet_maps.get(qid) or "snippet_map" in response):
                        response["snippet_map"] = merge_snippet_maps(
                            snippet_maps.get(qid, {}), response.get("snippet_map") or {}
                        )
                    append_jsonl({
                        "id": question["id"],
                        "question": question["question"],
                        "question_metadata": question.get("metadata", {}),
                        "signal": signals[qid],
                        "full_response": response,
                    }, str(output_file))
                    written += 1
                    del requests[request_id]
                    pbar.update(1)
                    logger.info(f"Q{qid} done ({elapsed / 60:.1f}m)")

                if requests:
                    time.sleep(POLL_INTERVAL_SECONDS)

        if batch_number < len(batches) and requests_sent:
            logger.info(f"Batch {batch_number} done; pausing {BATCH_PAUSE_SECONDS}s before the next batch")
            time.sleep(BATCH_PAUSE_SECONDS)

    if timed_out:
        logger.warning(f"Timed out: Q{timed_out}")
    if failed:
        logger.warning(f"Failed: Q{failed}")
    logger.info(f"✓ {written} reports → {output_file}")
    return timed_out + failed
