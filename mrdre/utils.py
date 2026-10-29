"""
Utility functions for the multi-turn deep research agent project.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Benchmark files shipped with the repository
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def setup_logging(stream=sys.stderr):
    """Configure console logging for the command-line entry points."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        stream=stream,
    )
    # Suppress noisy HTTP library logs
    for name in ("httpx", "urllib3", "requests"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file into a list of dictionaries.
    
    Args:
        filepath: Path to the JSONL file
    
    Returns:
        List of dictionaries, one per line in the JSONL file
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_partial_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load the records an earlier, possibly interrupted run appended to a JSONL file.

    A line cut off by an interrupted write is dropped and the file is rewritten
    without it. A missing file has no records.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    records, broken = [], 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1
    if broken:
        logger.warning(f"Dropping {broken} unreadable lines from {filepath}")
        save_jsonl(records, str(filepath))
    return records


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """
    Save list of dictionaries to JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path where the JSONL file should be saved
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(entry: Dict[str, Any], filepath: str):
    """
    Append a single dictionary entry to a JSONL file, creating it if necessary.
    
    Args:
        entry: Dictionary to append
        filepath: Path where the JSONL file should be stored
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into a dictionary.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary containing the JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path where the JSON file should be saved
        indent: Number of spaces for indentation (default: 2)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_dataset(dataset_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load dataset JSONL file and index by question ID.
    
    Args:
        dataset_path: Path to dataset JSONL file
    
    Returns:
        Dict mapping question ID to dataset item
    """
    dataset_items = load_jsonl(dataset_path)
    return {item["id"]: item for item in dataset_items}


def resolve_dataset(name_or_path: str) -> Path:
    """
    Resolve a dataset argument to a JSONL file.

    Args:
        name_or_path: A benchmark name in data/ (case-insensitive, e.g. "researcherbench")
            or a path to a dataset JSONL file

    Returns:
        Path to the dataset JSONL file
    """
    path = Path(name_or_path)
    if path.is_file():
        return path
    candidates = sorted(DATA_DIR.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No benchmark files in {DATA_DIR}. Install Mr Dre from a clone with `pip install -e .`, "
            "or pass a path to a dataset JSONL file."
        )
    for candidate in candidates:
        if candidate.stem.lower() == name_or_path.lower():
            return candidate
    names = ", ".join(c.stem for c in candidates)
    raise FileNotFoundError(f"Unknown dataset '{name_or_path}'. Use one of {names}, or a path to a JSONL file.")


def sample_questions(questions: List[Dict[str, Any]], num_questions: int) -> List[Dict[str, Any]]:
    """
    Select a fixed random subset of questions (seed 42).

    Generation, feedback and evaluation therefore work on the same questions. The
    subset is the one used in our paper (e.g. the 100 RigorousBench questions).

    Args:
        questions: Dataset items in file order
        num_questions: Subset size; -1 (or any value <= 0) keeps all questions

    Returns:
        The selected dataset items
    """
    if num_questions <= 0:
        return questions
    questions = list(questions)
    random.Random(42).shuffle(questions)
    return questions[:num_questions]


def model_name(model: Any) -> str:
    """
    Extract model name from a model object or string.
    
    Args:
        model: Model object (with model_name attribute), string, or any object
    
    Returns:
        String representation of the model name
    """
    if hasattr(model, "model_name"):
        return getattr(model, "model_name")
    if isinstance(model, str):
        return model
    return getattr(model, "__class__", type(model)).__name__
