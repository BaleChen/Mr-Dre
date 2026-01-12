"""
Utility functions for the multi-turn deep research agent project.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


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

