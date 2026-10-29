"""
Single-turn evaluation of research reports: comprehensiveness (checklist),
factuality (citation) and presentation (rubric).
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from mrdre.dr_eval.checklist_eval import evaluate_checklist, aggregate_checklist_metrics
from mrdre.dr_eval.citation_eval import evaluate_citations, aggregate_citation_metrics
from mrdre.dr_eval.rubric_eval import evaluate_rubric, aggregate_rubric_metrics
from mrdre.dr_eval.eval_prompts import (
    CHECKLIST_EVAL_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CONTENT_SUMMARIZATION_SYSTEM_PROMPT,
    RUBRIC_EVAL_SYSTEM_PROMPT,
    SUPPORTED_JUDGE_SYSTEM_PROMPT,
)
from mrdre.engine.dr_tulu import parse_and_format_citations
from mrdre.engine.oai import GPT
from mrdre.utils import append_jsonl, load_jsonl, load_partial_jsonl, save_jsonl

logger = logging.getLogger(__name__)

EVAL_MODES = ("checklist", "citation", "rubric")
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_SUMMARIZATION_MODEL = "gpt-4.1-nano"
# The reports scored in each output directory of `mrdre evaluate`, as {"id", "full_response"}
REPORTS_FILE = "scored_reports.jsonl"
# Fields of the per-report records that name the judge models of each mode
JUDGE_MODEL_FIELDS = {
    "checklist": ("checklist_evaluator_model",),
    "citation": ("claim_extraction_model", "supported_judge_model"),
    "rubric": ("judge_model",),
}

AGGREGATORS = {
    "checklist": aggregate_checklist_metrics,
    "citation": aggregate_citation_metrics,
    "rubric": aggregate_rubric_metrics,
}


def load_reports(
    report_file: str,
    dataset: Dict[Any, Dict[str, Any]],
    question_ids: Optional[Set[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load a reports JSONL file into the record format the judges read.

    A record needs "id" and the report text, either as "report" or as
    "full_response.report" (the format written by generation). IDs are matched to
    the dataset's IDs by their string form, and a missing "question" is filled in
    from the dataset.

    Args:
        report_file: Path to the reports JSONL file
        dataset: Dataset items indexed by question ID
        question_ids: If given, keep only reports for these questions

    Returns:
        List of report records with "id", "question" and "full_response.report"
    """
    reports = load_jsonl(report_file)
    # Match IDs by their string form, so "1" in a reports file matches 1 in the dataset
    dataset_ids = {str(qid): qid for qid in dataset}
    unknown_ids = [report.get("id") for report in reports if str(report.get("id")) not in dataset_ids]
    if unknown_ids:
        raise ValueError(f"{report_file}: {len(unknown_ids)} report IDs are not in the dataset, e.g. {unknown_ids[:5]}")
    for report in reports:
        report["id"] = dataset_ids[str(report["id"])]

    if question_ids is not None:
        reports = [report for report in reports if report["id"] in question_ids]

    missing = []
    for report in reports:
        if "full_response" not in report and "report" in report:
            report["full_response"] = {"report": report.pop("report")}
        full_response = report.get("full_response")
        if not isinstance(full_response, dict) or not isinstance(full_response.get("report"), str):
            missing.append(report["id"])
        report.setdefault("question", dataset[report["id"]]["question"])
    if missing:
        raise ValueError(f"{report_file}: {len(missing)} records have no report text in 'report' or "
                         f"'full_response.report', e.g. {missing[:5]}")
    return reports


def check_output_dir(output_dir: Path, reports: List[Dict[str, Any]], overwrite: bool = False) -> None:
    """
    Make sure an output directory only holds scores of these reports, and record them.

    Scores are resumed by question ID, so a directory that already holds scores of a
    different report for the same question would silently be reused. The directory
    keeps the reports it scored in scored_reports.jsonl, which a later revision
    evaluation also reads (load_evaluation). A mismatch is an error unless overwrite is
    set; the scores of the mismatched reports are then removed in every mode.

    Args:
        output_dir: Output directory of evaluate_reports, created if missing
        reports: Report records from load_reports
        overwrite: Whether the reports will be re-evaluated (ignore_existing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    changed = changed_reports(output_dir, reports)
    if changed and not overwrite:
        raise ValueError(
            f"{output_dir} holds scores of different reports for {len(changed)} questions "
            f"(e.g. Q{', Q'.join(changed[:5])}); use a new output directory or --overwrite"
        )
    # Remove the scores of replaced reports in all modes, including modes this run does not score
    for mode in EVAL_MODES if changed else ():
        score_file = output_dir / f"{mode}.jsonl"
        if score_file.exists():
            kept = [record for record in load_partial_jsonl(score_file) if str(record.get("id")) not in changed]
            save_jsonl(kept, str(score_file))

    stored = _stored_reports(output_dir)
    for report in reports:
        full_response = report["full_response"]
        stored[str(report["id"])] = {
            "id": report["id"],
            "full_response": {
                "report": full_response["report"],
                **({"snippet_map": full_response["snippet_map"]} if full_response.get("snippet_map") else {}),
            },
        }
    # Write through a temporary file, so an interruption cannot leave a truncated file
    temp_file = output_dir / f"{REPORTS_FILE}.tmp"
    save_jsonl(list(stored.values()), str(temp_file))
    os.replace(temp_file, output_dir / REPORTS_FILE)


def changed_reports(output_dir: Path, reports: List[Dict[str, Any]]) -> List[str]:
    """Return the IDs of reports that differ from the ones scored in output_dir."""
    stored = _stored_reports(Path(output_dir))
    return sorted(
        str(report["id"]) for report in reports
        if str(report["id"]) in stored and _report_hash(stored[str(report["id"])]) != _report_hash(report)
    )


def load_evaluation(
    eval_dir: Path,
    dataset: Dict[Any, Dict[str, Any]],
    question_ids: Set[Any],
    modes: Sequence[str],
    judge_model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Load the reports and scores of an earlier `mrdre evaluate` run, to compare a revision with.

    Nothing is re-scored: every question must already have a report and a score of each
    mode in eval_dir, from the same judge model.

    Args:
        eval_dir: Output directory of the earlier run
        dataset: Dataset items indexed by question ID
        question_ids: Questions to load (those of the revised reports)
        modes: Modes whose scores are needed
        judge_model: Judge model of the current run

    Returns:
        The scored reports and their per-report records by mode, for question_ids
    """
    eval_dir = Path(eval_dir)
    reports_file = eval_dir / REPORTS_FILE
    if not reports_file.is_file():
        raise ValueError(
            f"{eval_dir} is not an output directory of mrdre evaluate (it has no {REPORTS_FILE}); "
            "pass the --out directory of the previous turn's evaluation, not its reports file"
        )
    reports = load_reports(str(reports_file), dataset, question_ids)
    missing = {"report": question_ids - {report["id"] for report in reports}}

    records_by_mode = {}
    for mode in modes:
        path = eval_dir / f"{mode}.jsonl"
        records_by_id = {record["id"]: record for record in (_read_jsonl(path) if path.is_file() else [])}
        records_by_mode[mode] = [records_by_id[qid] for qid in question_ids if qid in records_by_id]
        missing[f"{mode} score"] = question_ids - set(records_by_id)
        _check_judge_models(
            records_by_mode[mode], {field: judge_model for field in JUDGE_MODEL_FIELDS[mode]}, path,
            "both turns must be scored with the same --judge-model",
        )

    for what, ids in missing.items():
        if ids:
            examples = ", Q".join(str(qid) for qid in sorted(ids, key=str)[:5])
            raise ValueError(
                f"{eval_dir} has no {what} for {len(ids)} of the revised reports (e.g. Q{examples}); "
                "evaluate the previous turn into it first, with the same --mode and --limit"
            )
    return reports, records_by_mode


def evaluate_reports(
    reports: List[Dict[str, Any]],
    dataset: Dict[Any, Dict[str, Any]],
    output_dir: Path,
    modes: Sequence[str] = EVAL_MODES,
    file_suffix: str = "",
    evaluator_model_name: str = DEFAULT_JUDGE_MODEL,
    claim_extraction_model_name: str = DEFAULT_JUDGE_MODEL,
    supported_judge_model_name: str = DEFAULT_JUDGE_MODEL,
    webpage_summarization_model_name: str = DEFAULT_SUMMARIZATION_MODEL,
    rubric_judge_model_name: str = DEFAULT_JUDGE_MODEL,
    ignore_existing: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate reports and write one record per report to {output_dir}/{mode}{file_suffix}.jsonl.

    Reports that already have a record in that file are skipped, so an interrupted
    run resumes where it stopped.

    Args:
        reports: Report records from load_reports
        dataset: Dataset items indexed by question ID
        output_dir: Directory for the per-report output files
        modes: Any of "checklist", "citation", "rubric"
        file_suffix: Suffix of the output file names (e.g. "_turn1")
        evaluator_model_name: Model for checklist evaluation
        claim_extraction_model_name: Model for claim extraction
        supported_judge_model_name: Model for judging claim support
        webpage_summarization_model_name: Model for summarizing webpage content
        rubric_judge_model_name: Model for rubric evaluation
        ignore_existing: If True, discard existing output files and re-evaluate all reports

    Returns:
        Mapping from mode to the per-report records of all evaluated reports
    """
    reports = _format_dr_tulu_citations(reports)

    judges: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
    if "checklist" in modes:
        evaluator_model = GPT(evaluator_model_name, system_prompt=CHECKLIST_EVAL_SYSTEM_PROMPT)
        judges["checklist"] = lambda report: evaluate_checklist(
            report, dataset[report["id"]], evaluator_model
        )
    if "citation" in modes:
        claim_extraction_model = GPT(claim_extraction_model_name, system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT)
        supported_judge_model = GPT(supported_judge_model_name, system_prompt=SUPPORTED_JUDGE_SYSTEM_PROMPT)
        webpage_summarization_model = GPT(webpage_summarization_model_name, system_prompt=CONTENT_SUMMARIZATION_SYSTEM_PROMPT)
        judges["citation"] = lambda report: evaluate_citations(
            report,
            claim_extraction_model,
            supported_judge_model,
            webpage_summarization_model=webpage_summarization_model,
        )
    if "rubric" in modes:
        rubric_judge_model = GPT(rubric_judge_model_name, system_prompt=RUBRIC_EVAL_SYSTEM_PROMPT)
        judges["rubric"] = lambda report: evaluate_rubric(
            report, dataset[report["id"]], rubric_judge_model, rubric_dimensions=["presentation"]
        )

    output_dir = Path(output_dir)
    if not ignore_existing:
        # Scores resumed from output_dir must come from the same judges as new ones
        judge_models = {
            "checklist": {"checklist_evaluator_model": evaluator_model_name},
            "citation": {
                "claim_extraction_model": claim_extraction_model_name,
                "supported_judge_model": supported_judge_model_name,
            },
            "rubric": {"judge_model": rubric_judge_model_name},
        }
        report_ids = {report["id"] for report in reports}
        for mode in judges:
            output_file = output_dir / f"{mode}{file_suffix}.jsonl"
            existing = [record for record in load_partial_jsonl(output_file) if record.get("id") in report_ids]
            _check_judge_models(
                existing, judge_models[mode], output_file, "use the same judge model or a new output directory"
            )

    return {
        mode: _evaluate_mode(mode, judge, reports, output_dir / f"{mode}{file_suffix}.jsonl", ignore_existing)
        for mode, judge in judges.items()
    }


def summarize(records_by_mode: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate the per-report records of each mode into dataset-level metrics."""
    return {mode: AGGREGATORS[mode](records) for mode, records in records_by_mode.items()}


def _evaluate_mode(
    mode: str,
    judge: Callable[[Dict[str, Any]], Dict[str, Any]],
    reports: List[Dict[str, Any]],
    output_file: Path,
    ignore_existing: bool,
) -> List[Dict[str, Any]]:
    """Run one judge over the reports, resuming from the records already in output_file."""
    report_ids = {report["id"] for report in reports}
    existing = load_partial_jsonl(output_file)
    if ignore_existing:
        # Drop the old records of these reports; records of other questions stay in the file
        if output_file.exists():
            logger.info(f"Re-evaluating {len(report_ids)} reports in {output_file}")
            save_jsonl([record for record in existing if record.get("id") not in report_ids], str(output_file))
        records: List[Dict[str, Any]] = []
    else:
        records = [record for record in existing if record.get("id") in report_ids]
        if records:
            logger.info(f"Resuming {mode}: {len(records)} existing")

    done_ids = {record["id"] for record in records}
    for report in tqdm(reports, desc=f"{mode.capitalize()} eval", unit="report"):
        if report["id"] in done_ids:
            continue
        try:
            record = judge(report)
        except Exception as e:
            logger.error(f"{mode.capitalize()} eval failed for Q{report['id']}: {e}")
            continue
        done_ids.add(report["id"])
        records.append(record)
        append_jsonl(record, str(output_file))

    logger.info(f"{mode.capitalize()} done → {output_file}")
    return records


def _check_judge_models(
    records: List[Dict[str, Any]], judge_models: Dict[str, str], source: Path, hint: str
) -> None:
    """Raise ValueError if a record names another judge than judge_models (record field -> model)."""
    for field, model in judge_models.items():
        others = {record.get(field) for record in records} - {model, None}
        if others:
            raise ValueError(f"{source} holds scores from {', '.join(sorted(others))}, not {model}; {hint}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file of an evaluation directory, naming the file if it cannot be parsed."""
    try:
        return load_jsonl(str(path))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} cannot be parsed ({e}); rerun the evaluation that wrote it") from e


def _stored_reports(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return the reports scored in output_dir, indexed by question ID as a string."""
    reports_file = output_dir / REPORTS_FILE
    return {str(report["id"]): report for report in _read_jsonl(reports_file)} if reports_file.exists() else {}


def _report_hash(report: Dict[str, Any]) -> str:
    """Fingerprint of a report's text and DR Tulu snippet map."""
    full_response = report["full_response"]
    content = json.dumps([full_response["report"], full_response.get("snippet_map") or None], sort_keys=True)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _format_dr_tulu_citations(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolve DR Tulu citations before evaluation.

    DR Tulu reports cite snippet IDs, which are resolved to numbered URL citations
    through the snippet_map stored with each report. Other reports are returned as is,
    and the input records are not modified.
    """
    formatted = []
    for report in reports:
        full_response = report["full_response"]
        snippet_map = full_response.get("snippet_map")
        if snippet_map:
            report = {
                **report,
                "full_response": {
                    **full_response,
                    "report": parse_and_format_citations(full_response["report"], snippet_map),
                },
            }
        formatted.append(report)
    return formatted
