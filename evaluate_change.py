"""
Compare two generation runs in the output/<dataset>/<turn type>/<model>/ folder layout
of the paper's runs and emit a single CSV-friendly summary row with: Break Rate,
Incorporation Rate, Checklist Coverage Delta, Citation Faithfulness Delta, Claim
Groundedness Delta, Presentation Delta.

The newer `mrdre evaluate --previous-eval ... --feedback ...` command compares any two evaluated turns.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Load .env before importing mrdre modules: the disk caches read MCP_CACHE_DIR when they are imported
load_dotenv()

from mrdre.revision import COLUMNS, compute_revision_metrics  # noqa: E402
from mrdre.utils import load_jsonl  # noqa: E402

GEN_FILE_PATTERN = re.compile(r"gen_turn(\d+)\.jsonl$", re.IGNORECASE)


def parse_run(gen_file: str) -> Tuple[Path, Path, int]:
    path = Path(gen_file).expanduser().resolve()
    turn = int(GEN_FILE_PATTERN.search(path.name).group(1))
    parts = list(path.parts)
    out_idx = parts.index("output")
    run_dir = Path(*parts[: out_idx + 4])
    return path, run_dir, turn


def load_metrics(run_dir: Path, turn: int) -> Dict[str, List[Dict]]:
    metrics = {}
    for mode in ("checklist", "citation", "rubric"):
        path = run_dir / f"{mode}_turn{turn}.jsonl"
        metrics[mode] = load_jsonl(str(path)) if path.is_file() else []
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two generation runs.")
    parser.add_argument("gen_file_v1", type=str, help="Generation file for version 1")
    parser.add_argument("gen_file_v2", type=str, help="Generation file for version 2")
    parser.add_argument("--format-feedback", default=False, action="store_true",
                        help="Judge the incorporation rate with the pairwise LLM judge (format feedback); "
                             "otherwise it is the fix rate of the targeted checklist items (content feedback).")
    parser.add_argument("--output-file", type=str, default="incorporation_results.jsonl",
                        help="Output file to save detailed scores")
    args = parser.parse_args()

    gen_v1, run_v1, turn_v1 = parse_run(args.gen_file_v1)
    gen_v2, run_v2, turn_v2 = parse_run(args.gen_file_v2)

    # The revised reports carry the feedback they answered in their "signal" field
    values = compute_revision_metrics(
        load_jsonl(str(gen_v1)),
        load_jsonl(str(gen_v2)),
        load_metrics(run_v1, turn_v1),
        load_metrics(run_v2, turn_v2),
        feedback_type="format" if args.format_feedback else "content",
        judge_output_file=Path(args.output_file),
    )

    writer = csv.writer(sys.stdout)
    writer.writerow(COLUMNS)
    writer.writerow([f"{values[column]:.4f}" for column in COLUMNS])


if __name__ == "__main__":
    main()
