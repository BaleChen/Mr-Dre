#!/usr/bin/env python3
"""
Feedback refinement utilities.

This module implements functions to refine feedback for research reports,
making it more specific and actionable by targeting exact locations in the report.
"""

import json
import os
import logging
from typing import List, Dict, Any

from tqdm import tqdm
from engine.oai import GPT
from dotenv import load_dotenv

load_dotenv()

# Get module logger
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Model configuration
REFINE_FEEDBACK_GPT_MODEL = "gpt-4.1"

FEEDBACK_REFINER_SPROMPT = r"""
You are an expert technical editor. Your task is to translate high-level user feedback into a minimal, localized edit plan.

Input You Will Receive:
1) Original Query  
2) Full Research Report  
3) Original Feedback (often vague or high-level)

Your Goal:
Create a structured **edit plan** that enables a research agent to take specific, localized editing actions without ambiguity.  
Do NOT write or fabricate final factual content — your job is to **identify where** and **what type** of content needs to be added/changed.

---

Editing Constraints:

- Only use the following atomic actions: **DELETE / INSERT / MODIFY**
- Every edit must specify an exact location with:
  - `Section` name (must match exactly)
  - `Subsection` name (or N/A if not applicable)
  - `Anchor` quote: A short (≤18 words) **verbatim** sentence/phrase from the current report that clearly identifies **where** the edit should occur.
- Reference the Anchor in your **Content Spec** to clarify where in the text the change happens.  
- INSERT actions can create new sections/subsections, but only if explicitly specified in the feedback. 
- Do NOT invent specific facts (names, numbers, dates, benchmarks, claims).

---

Output Format (Markdown):

Feedback:
[Insert original feedback exactly as received]

Edit Actions:
1) Action: DELETE | INSERT | MODIFY  
   Location:
   - Section: "[Exact section name]"
     *(For new sections, use format: `NEW: [Section Name]`)*  
   - Subsection: "[Exact subsection name]" (or N/A)
     *(For new subsections, use format: `NEW: [Subsection Name]`)*
   - Anchor: "[Short verbatim quote from report]"  
     *(For new sections/subsections, specify relative location, e.g., "After section 'Discussion')*
   Content Spec:
   - What to change: Describe required content, not final prose    
   - Must-include: Specific elements that must be part of the edit
"""


def load_feedback(feedback_path: str) -> List[Dict[str, Any]]:
    """
    Load feedback from a JSONL file.
    
    Args:
        feedback_path: Path to the feedback JSONL file
        
    Returns:
        List of feedback entries
    """
    feedbacks = []
    with open(feedback_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                feedbacks.append(json.loads(line))
    return feedbacks


def load_gen_files(gen_dir: str, round_num: int = 1) -> Dict[int, Dict[str, Any]]:
    """
    Load gen_turn{round_num}.jsonl file and build an ID to content mapping.
    
    Args:
        gen_dir: Directory containing gen_turn{round_num}.jsonl file
        round_num: Round number to load (default: 1)
        
    Returns:
        Mapping from ID to gen entry (containing question and report)
    """
    gen_map = {}
    
    # Load gen_turn{round_num}.jsonl file
    filename = f"gen_turn{round_num}.jsonl"
    filepath = os.path.join(gen_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        gen_id = entry["id"]
                        # Build entry with question and report
                        question = entry["question"]
                        # Get report from full_response.report if available, otherwise from response or report field
                        full_response = entry.get("full_response", {})
                        report = full_response.get("report", "")
                        if not report:
                            report = entry.get("response", "")
                        if not report:
                            report = entry.get("report", "")
                        
                        gen_map[gen_id] = {
                            "question": question,
                            "report": report
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line in {filepath}: {e}")
                    except KeyError as e:
                        logger.error(f"Missing key {e} in entry: {entry}")
    else:
        logger.error(f"{filename} not found in {gen_dir}")
    
    return gen_map


from typing import Dict, Any

HARD_CODED_CONSTRAINTS = """
You are given:
1) The original user feedback
2) A structured list of localized Edit Actions derived from that feedback

Each Edit Action includes:
- Action: One of DELETE, INSERT, or MODIFY — the atomic type of edit to apply.
- Section / Subsection: The precise location in the document where the edit applies. If the action introduces a new section/subsection, it will be labeled as `NEW: [Name]`.
- Anchor: A short verbatim quote from the report identifying the exact insertion/modification point. For new sections/subsections, this is a relative reference (e.g., "After section 'Discussion'").
- Content Spec: A short explanation of what to change, localized to the Anchor location. This is NOT final content — only a structural and intent-level guide.

Non-negotiable editing constraints:
- Apply ONLY the actions listed under "Edit Actions".
- Do NOT infer, add, or modify edits beyond what is explicitly specified.
- Do NOT reinterpret or expand the original feedback.
- Do NOT rewrite sections wholesale; keep edits strictly local to the specified Anchor quote.

Your Task:
Apply the Edit Actions to improve the report by making precise, localized edits at the specified locations, adhering strictly to all constraints above. Please only output the revised report and no other text such as comments or explanations.
""".strip()


# - Do NOT introduce new sections, reorder headings, or modify formatting beyond the listed actions unless an edit action explicitly instructs so.
def refine_feedback(feedback: Dict[str, Any], gen_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refine feedback to be more specific and actionable using GPT.

    Args:
        feedback: Original feedback entry
        gen_entry: Corresponding gen entry with question and report

    Returns:
        Refined feedback entry
    """
    gpt = GPT(REFINE_FEEDBACK_GPT_MODEL, system_prompt=FEEDBACK_REFINER_SPROMPT)

    # Build prompt (keep it tight; the system prompt already enforces structure)
    feedback_prompt = f"""
Original Query:
{gen_entry.get('question', '')}

Full Research Report:
{gen_entry.get('report', '')}

Original Feedback:
{feedback.get('feedback', '')}

Task:
Refine the original feedback into a minimal, localized edit plan using the required output format.
""".strip()

    refined_feedback = gpt(feedback_prompt).strip()
    refined_feedback = refined_feedback.rstrip() + "\n\n" + HARD_CODED_CONSTRAINTS + "\n"

    return {
        "id": feedback["id"],
        "feedback": refined_feedback,
        "question": gen_entry.get("question", ""),
        "original_feedback": feedback.get("feedback", ""),
        "item": feedback.get("item", []),
        "item_id": feedback.get("item_id", [])
    }



def process_feedback(feedback_path: str, gen_dir: str = None, output_path: str = None, limit: int = None) -> None:
    """
    Process feedback entries and generate refined feedback.
    
    Args:
        feedback_path: Path to input feedback file
        gen_dir: Directory containing gen_turn1.jsonl file (defaults to feedback file directory)
        output_path: Path to output refined feedback file (defaults to feedback file directory with refined_ prefix)
        limit: Maximum number of feedback entries to process (None for all)
    """
    # 自动确定gen_dir和output_path
    feedback_dir = os.path.dirname(feedback_path)
    feedback_filename = os.path.basename(feedback_path)
    
    if not gen_dir:
        gen_dir = feedback_dir
    
    if not output_path:
        # 在文件名前添加refined_前缀
        base_name, ext = os.path.splitext(feedback_filename)
        output_filename = f"refined_{base_name}{ext}"
        output_path = os.path.join(feedback_dir, output_filename)
    # Load feedback and gen files
    feedbacks = load_feedback(feedback_path)
    gen_map = load_gen_files(gen_dir)
    
    # Apply limit if specified
    if limit is not None:
        feedbacks = feedbacks[:limit]
        logger.info(f"Processing first {limit} feedback entries")
    
    # Refine each feedback entry
    refined_feedbacks = []
    for feedback in tqdm(feedbacks, desc="Refining feedback"):
        feedback_id = feedback["id"]
        if feedback_id in gen_map:
            gen_entry = gen_map[feedback_id]
            refined_feedback = refine_feedback(feedback, gen_entry)
            refined_feedbacks.append(refined_feedback)
        else:
            logger.warning(f"No gen entry found for feedback ID {feedback_id}")
    
    # Write refined feedback to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in refined_feedbacks:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Refined feedback written to {output_path}")


def main():
    """
    Main entry point for the feedback refinement script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine feedback using GPT-4.1-mini")
    parser.add_argument("--feedback-path", required=True, help="Path to input feedback file")
    parser.add_argument("--gen-dir", default=None, help="Directory containing gen_turn1.jsonl file (defaults to feedback file directory)")
    parser.add_argument("--output-path", default=None, help="Path to output refined feedback file (defaults to feedback file directory with refined_ prefix)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of feedback entries to process (None for all)")
    
    args = parser.parse_args()
    
    process_feedback(
        feedback_path=args.feedback_path,
        gen_dir=args.gen_dir,
        output_path=args.output_path,
        limit=args.limit
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()   