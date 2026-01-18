"""
Feedback generation utilities.

This module implements helper functions to synthesize natural language
feedback for research reports, based on evaluation results and the
original user queries.
"""

import json
import logging
import random
from typing import List, Dict, Any

from tqdm import tqdm
from engine.oai import GPT
from dotenv import load_dotenv

load_dotenv()

# Get module logger (inherits config from main entry point)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

CHECKLIST_FEEDBACK_GPT_MODEL = "gpt-4.1-mini"
FORMAL_FEEDBACK_GPT_MODEL = "gpt-4.1-mini"


USER_SIMULATOR_SPROMPT = """
You are a user providing feedback to a research report writing agent.

You will be provided with:
    1. The original query that you asked
    2. A specific evaluation rule that was used to assess the report where the agent achieves a suboptimal score
    3. The coverage status (whether the rule was covered in the report)
    4. The weight of the rule (positive means the rule should be covered, negative means the rule should NOT be covered)
    5. The evaluator's explanation for the score

Your task is to provide natural, constructive, concrete feedback that a normal user would give to improve the report based on this specific evaluation rule.

Guidelines:
    - Be converstational and natural as possible. Imagine you are talking to a collaborator who is helping you to write the report.
    - Occasionally, you can use some colloquial language to make the feedback more realistic.
    - Your feedback should be in 1-2 sentences that is concise and to the point.
    - Don't repeat the evaluation explanation verbatim but use it as a reference to help you provide the feedback.
"""

USER_SIMULATOR_SPROMPT_MULTI = """
You are a user providing feedback to a research report writing agent.

You will be provided with:
    1. The original query that you asked
    2. Several specific evaluation rules that were used to assess the report where the agent achieves suboptimal scores
    3. For each rule, the coverage status (whether the rule was covered in the report)
    4. For each rule, the weight of the rule (positive means the rule should be covered, negative means the rule should NOT be covered)
    5. The evaluator's explanation for each score

Your task is to provide natural, constructive, concrete feedback that a normal user would give to improve the report based on these specific evaluation rules.

Guidelines:
    - Be converstational and natural as possible. Imagine you are talking to a collaborator who is helping you to write the report.
    - Occasionally, you can use some colloquial language to make the feedback more realistic.
    - Your feedback for each evaluation point should be in 1-2 sentences that is concise and to the point.
    - Don't repeat the evaluation explanations verbatim but use them as a reference to help you provide the feedback.
"""

def generate_feedback_for_point(
    original_query: str,
    point_obj: Dict[str, Any],
) -> str:
    """
    Call GPT to generate user-style feedback for a single evaluation point that needs improvement.

    Args:
        original_query: The original question/query text for this sample.
        point_obj: A dictionary describing one evaluation point, expected to contain
            at least the keys:
                - "item": the evaluation criterion
                - "covered": whether the criterion was covered in the report
                - "weight": weight of the criterion (positive=should be covered, negative=should NOT be covered)
                - "justification": the evaluator's explanation for the score

    Returns:
        A single feedback string (stripped), written as 1–2 concise, natural sentences.
    """
    point_text = point_obj.get("item", "")
    justification = point_obj.get("justification", "")
    
    # Get covered value
    covered_value = point_obj.get("covered", False)
    try:
        covered = float(covered_value)
    except (TypeError, ValueError):
        covered = 0.0
    
    # Get weight value
    weight = point_obj.get("weight", 1)
    try:
        w = float(weight)
    except (TypeError, ValueError):
        w = 1.0
    
    # Determine if the point should be covered based on weight
    should_be_covered = w >= 0
    
    gpt = GPT(CHECKLIST_FEEDBACK_GPT_MODEL, system_prompt=USER_SIMULATOR_SPROMPT)

    feedback_prompt = f"""
Original Query:
{original_query}

Evaluation Point: (the criteria that need improvement):
{point_text}

Coverage Status:
{covered} (1.0 means covered, 0.0 means not covered, 0.5  means partially covered)

Point Weight:
{w} (positive means the point SHOULD be covered, negative means the point should NOT be covered)

Evaluator's Explanation:
{justification}

This point is being flagged for improvement because:
- For positive weight points: the point was not covered (should be covered)
- For negative weight points: the point was covered (should not be covered)

Please provide 1–2 sentences of natural, constructive feedback that you,as a normal user / collaborator, would give to improve the report with respect to this point. Be concise and conversational.
"""

    feedback = gpt(feedback_prompt).strip()

    # Optionally append a global suffix if defined in the runtime environment.
    if "USER_FEEDBACK_SUFFIX" in globals():
        feedback = feedback + "\n\n" + USER_FEEDBACK_SUFFIX

    return feedback.strip()

def generate_feedback_for_points(
    original_query: str,
    points: List[Dict[str, Any]],
) -> str:
    """
    Call GPT to generate user-style feedback for multiple evaluation points that need improvement.

    Args:
        original_query: The original question/query text for this sample.
        points: A list of evaluation point dicts, each expected to contain at least:
            - "id": point id
            - "item": the evaluation criterion
            - "covered": whether the criterion was covered in the report (0=no, 1=yes)
            - "weight": weight of the criterion (positive=should be covered, negative=should NOT be covered)
            - "justification": the evaluator's explanation for the score

    Returns:
        A single feedback string (stripped), written as 1–2 (or a few) concise, natural sentences for each point.
    """
    lines = []
    for idx, p in enumerate(points, start=1):
        item = p.get("item", "")
        justification = p.get("justification", "")
        pid = p.get("id")
        
        # Get covered value
        covered_value = p.get("covered", False)
        try:
            covered = float(covered_value)
        except (TypeError, ValueError):
            covered = 0.0
        
        # Get weight value
        weight = p.get("weight", 1)
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 1.0
        
        lines.append(
            f"Evaluation Point {idx}:\n"
            f"  Criterion: {item}\n"
            f"  Coverage Status: {covered} (1.0 means covered, 0.0 means not covered, 0.5 means partially covered)\n"
            f"  Point Weight: {w} (positive=should be covered, negative=should NOT be covered)\n"
            f"  Evaluator's Explanation: {justification}\n"
        )
    points_block = "\n".join(lines)

    gpt = GPT(CHECKLIST_FEEDBACK_GPT_MODEL, system_prompt=USER_SIMULATOR_SPROMPT_MULTI)

    feedback_prompt = f"""
Original Query:
{original_query}

Evaluation Points (the criteria that need improvement):
{points_block}

For each point, it's being flagged because:
- For positive weight points: the point was not covered (should be covered)
- For negative weight points: the point was covered (should not be covered)

Please provide 1–2 sentences of natural, constructive feedback that you, as a normal user / collaborator, would give to improve the report with respect to each point. Be concise and conversational.
"""

    feedback = gpt(feedback_prompt).strip()

    if "USER_FEEDBACK_SUFFIX" in globals():
        feedback = feedback + "\n\n" + USER_FEEDBACK_SUFFIX

    return feedback.strip()


def is_failed_checklist_point(point: Dict[str, Any]) -> bool:
    """
    Decide whether a checklist point should be treated as "failed"
    given its coverage and weight.

    - For normal / positive items (weight >= 0):
        fail  <=> not covered (covered < 1)
    - For negative items         (weight < 0):
        fail  <=> covered (covered >= 1)

    If weight is missing or invalid, default to positive-item behavior.
    If covered is missing or invalid, default to not covered.
    """
    covered_value = point.get("covered", False)
    
    # Convert covered_value to float, default to 0.0 if conversion fails
    try:
        covered = float(covered_value)
    except (TypeError, ValueError):
        covered = 0.0
    
    # Check if covered is at least 1.0 to be considered covered
    is_covered = covered >= 1.0
    
    weight = point.get("weight", 1)

    try:
        w = float(weight)
    except (TypeError, ValueError):
        w = 1.0

    if w < 0:
        return is_covered
    else:
        return not is_covered


def generate_feedback_batch(
    qids: List[str],
    eval_map: Dict[str, Dict[str, Any]],
    questions_by_id: Dict[str, str],
    num_points: int = 1,
) -> List[Dict[str, Any]]:

    """
    Generate feedback for a batch of question IDs, given evaluation results and question texts.

    This function looks up each question ID in:
      - `eval_map` to find the corresponding evaluation result, and
      - `questions_by_id` to find the original question text.

    For each question:
      1. It collects all evaluation points where `covered == False`.
      2. It randomly selects one such point.
      3. It calls `generate_feedback_for_point` to synthesize a feedback message.

    Args:
        qids:
            List of question IDs (as strings or values convertible to strings)
            that require feedback.
        eval_map:
            Mapping from question ID (string) to the evaluation item, e.g.:
            {
              "1": {
                "id": 1,
                "checklist_details": [
                    {"item": "...", "covered": true/false, "justification": "...", "weight": ...},
                    ...
                    ]
              },
              ...
            }
        questions_by_id:
            Mapping from question ID (string) to the original question text:
            { "1": "original question text", ... }
        num_points:
            Number of evaluation points to sample for each question. Defaults to 1.

    Returns:
        A list of dictionaries, one per successfully generated feedback, each of the form:
            {
              "id": <original numeric or string id>,
              "feedback": "<generated feedback>",
              "question": "<original question text>",
              "item": [ "<point text 1>", ... ],
              "item_id": [ <point id 1>, ... ],
            }
    """
    outputs: List[Dict[str, Any]] = []
    qid_list = sorted(set(str(q) for q in qids))

    for qid_str in tqdm(qid_list, desc="Checklist feedback", unit="q"):
        eval_item = eval_map.get(qid_str)
        question_text = questions_by_id.get(qid_str)

        if eval_item is None:
            logger.warning(f"Q{qid_str} not in eval_map, skipping")
            continue
        if question_text is None:
            logger.warning(f"Q{qid_str} not in questions_by_id, skipping")
            continue

        # By convention, checklist-based evaluation results are stored under "checklist_details".
        coverage_results = eval_item.get("checklist_details", [])

        # Select only points that were not fully covered.
        false_points = [p for p in coverage_results if is_failed_checklist_point(p)]
        if not false_points:
            logger.debug(f"Q{qid_str} has no uncovered points, skipping")
            continue
            
        # If fewer points are available than requested, skip this question.
        if len(false_points) < num_points:
            logger.debug(f"Q{qid_str} has {len(false_points)} uncovered points (< {num_points}), skipping")
            continue

        # ---- Single point case: Call the original function, output wrapped in a list structure ----
        if num_points == 1:
            selected_point = random.choice(false_points)
            feedback = generate_feedback_for_point(question_text, selected_point)
            outputs.append(
                {
                    "id": eval_item.get("id", qid_str),
                    "feedback": feedback,  
                    "question": question_text,
                    "item": [selected_point.get("item", "")], 
                    "item_id": [selected_point.get("id", "")],
                }
            )

        # ---- Multiple points case: Randomly sample num_points, call the multi version ----
        else:
            selected_points = random.sample(false_points, k=num_points)
            feedback = generate_feedback_for_points(question_text, selected_points)
            items_list = [p.get("item", "") for p in selected_points]
            item_ids = [p.get("id", "") for p in selected_points]

            outputs.append(
                {
                    "id": eval_item.get("id", qid_str),
                    "feedback": feedback,  
                    "question": question_text,
                    "item": items_list,    # list of str
                    "item_id": item_ids,   # list of ids
                }
            )

    return outputs

# Formal feedback examples for seed selection
FORMAL_FEEDBACK_EXAMPLES = [
    "Please rewrite this so the language is clearer and more straightforward, suitable for a reader with no prior knowledge.",
    "Whenever you introduce a technical concept, add a simple and real-world analogy to illustrate it.",
    "Standardize heading levels and naming so similar sections use parallel phrasing (e.g., 'Approach', 'Results', 'Limitations').",
    "Make sure that each section ends with a short summary sentence that emphasizes the main takeaway.",
    "Add a concise TL;DR at the beginning of the report that states the main question and key takeaways from the report.",
    "It would help if the report indicated which parts are essential reading and which parts are optional background.",
    "Highlight key sentences or phrases (e.g., with bold) so I can quickly find the most important takeaways.",
    "Please add short 'section previews' at the start of each main section, summarizing in 1–2 lines what will be covered.",
    "Please keep the core sections concise and move extended explanations, detailed justifications, and long background passages into clearly labeled 'Appendix' sections at the end.",
    "Consider adding transition sentences between sections to show how each part connects to the next.",
    "Add subheadings to help readers navigate and find information quickly.",
    "Include a glossary of key terms at the end for readers who want quick reference.",
    "Consider using bullet points or numbered lists when presenting multiple related items rather than embedding them in prose.",
    "Add visual breaks like pull quotes to highlight critical insights so that it's easier to find takeaways.",
    "Apply bold formatting to critical findings, main conclusions, and essential terms on first mention, while using italics for secondary emphasis, technical terms in context, or when citing specific examples.",
    "Add a 'How to Read This Report' section that explains the document's structure and what different readers should focus on.",
    "Vary sentence length and structure to maintain reader interest and create rhythm.",
    "Use 'we' as much as possible than 'you' or third-person pronouns to create connection with readers rather than maintaining complete detachment.",
    "Add a brief 'Why This Matters' box at the start of technical sections to motivate readers.",
    "Close with actionable next steps or recommendations for related information so readers know what to do or read next.",
    "Create a separate 'Frequently Asked Questions' section to address common points of confusion."
]

# Combined system prompt for selecting and rewriting feedback
FORMAL_FEEDBACK_SPROMPT = """
You are a user providing feedback on the report's writing, structure, and presentation only, not on its facts, reasoning, or conclusions.

You will be provided with:
1. The original query given to the agent
2. The agent-generated research report
3. Three seed feedback examples from a predefined list

Your task is to:
1. First, select which of the three feedback examples would be most suitable and relevant for improving this report. It should be targeting an aspect that the report misses or did not do well. Do not give feedback to what is already done in the report. For example, if the report already uses subheadings or bulleted lists to organize information, you should not give feedback on that but select another aspect to ask for improvement.
2. Then, start from the selected feedback and either (a) rewrite it to be more specific and tailored to the actual content of the report while preserving its core suggestion, or (b) if a slightly different but closely related suggestion would better improve this particular report, adapt it into that alternative while staying within the same improvement category.

Your final feedback must adhere to the following specific desiderata:
- **Content-preserving**: Your feedback must not require any edits to existing content in the current draft. It should only incur changes in the form, structure, organization, tone, or style of the writing. Do NOT ask for new evidence, new arguments, or different conclusions.
- **Naturalness**: The language and wording should be natural and human-like, as if it was a natural follow-up response from the user themselves, or a thoughtful peer/supervisor. Give exactly one coherent suggestion (1–2 sentences) that feels like a natural follow-up from the user.
- **Draft-specific**: The feedback should be tailored to the original query and the current draft of the report, targeting aspects that the current draft misses and have clear room for improvement.
- **Actionability**: The feedback should be concrete and actionable, phrasing as implementable suggestions and avoiding vague comments such as "improve clarity" without explaining how. Make it specific to this draft and clearly implementable.

Please only respond with the final rewritten feedback, without any additional explanation or commentary.
"""

def generate_formal_feedback(original_query: str, report: str) -> str:
    """
    Call GPT to generate a single piece of formal feedback for a report.
    
    Steps:
    1. Randomly select 3 feedback examples from the predefined list
    2. Ask GPT to select the most appropriate one and rewrite it to be more specific to the report in a single call
    """
    # Step 1: Randomly select 3 feedback examples
    selected_examples = random.sample(FORMAL_FEEDBACK_EXAMPLES, 3)
    
    # Step 2: Ask GPT to select and rewrite the feedback in a single call
    gpt = GPT(FORMAL_FEEDBACK_GPT_MODEL, system_prompt=FORMAL_FEEDBACK_SPROMPT)
    combined_prompt = f"""
Original Query:
{original_query}

Report:
{report}

Feedback Examples:
1. {selected_examples[0]}
2. {selected_examples[1]}
3. {selected_examples[2]}

Please first select the most appropriate feedback example and then rewrite it to be more specific and tailored to this report. Provide only the final rewritten feedback (1–2 sentences).
"""
    
    final_feedback = gpt(combined_prompt).strip()
    return final_feedback



def generate_formal_feedback_batch(
    qids: List[str],
    reports_by_id: Dict[str, str],
    questions_by_id: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Generate formal feedback for a batch of question IDs, given the reports and original questions.

    Args:
        qids: List of question IDs that require feedback.
        reports_by_id: Mapping from question ID to the research report text.
        questions_by_id: Mapping from question ID to the original question text.

    Returns:
        A list of dictionaries, one per successfully generated feedback, each of the form:
            {
              "id": <question ID>,
              "feedback": <generated formal feedback>,
              "question": <original question text>,
              "item": [],
              "item_id": [],
            }
    """
    outputs: List[Dict[str, Any]] = []
    qid_list = sorted(set(str(q) for q in qids))

    for qid_str in tqdm(qid_list, desc="Formal feedback", unit="q"):
        report = reports_by_id.get(qid_str)
        question_text = questions_by_id.get(qid_str)

        if report is None:
            logger.warning(f"Q{qid_str} report not found, skipping")
            continue
        if question_text is None:
            logger.warning(f"Q{qid_str} question not found, skipping")
            continue

        try:
            feedback = generate_formal_feedback(question_text, report)
            outputs.append(
                {
                    "id": qid_str,
                    "feedback": feedback,
                    "question": question_text,
                    "item": [],
                    "item_id": [],
                }
            )
        except Exception as e:
            logger.warning(f"Q{qid_str} formal feedback failed: {e}")
            continue

    return outputs