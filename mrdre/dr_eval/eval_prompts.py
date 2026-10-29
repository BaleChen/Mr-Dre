from typing import Any, Dict, List

###########################################################
# Checklist-based evaluation prompts
###########################################################

CHECKLIST_EVAL_SYSTEM_PROMPT = (
    "You will be given a question the user asked (in <question></question> tags) and the corresponding report (in <report></report> tags) given as a response to the question by an assistant. You will then be given a specific criterion to evaluate the report against (in <criterion></criterion> tags. It could be a yes/no question or a statement about the report that you should judge whether it's true or not).\n"
    "Your task is to score the report based on whether it satisfies the criterion or not on a three-point scale: 1.0 if the report satisfies the criterion, 0.5 if the report partially satisfies the criterion, 0.0 if the report does not satisfy the criterion. Judge only the specified aspect(s), not any other qualities of the report. Please also provide a short (2-3 sentences maximum) justification for your score. Note: A criterion might be positive or negative. Satisfying the criterion means that the report contains the content that is described by the criterion, which should not be confused with satisfying the user's request.\n"
    "Output only a JSON string with the following format: {\"score\": float, \"justification\": string}. Do not include any other text or comments in your response."
)

CHECKLIST_EVAL_USER_PROMPT = (
    "Evaluate the report based on the given criterion.\n"
    "<question>\n{question}\n</question>\n"
    "<report>\n{report}\n</report>\n"
    "<criterion>\n{criterion}\n</criterion>\n"
)

NEGATIVE_REMINDER = (
    "Note: this is a negative criterion. Your score should be 1.0 only it describes something that is present or true to the report. If the report did not contain the content described by the criterion, your score should be 0.0.\n"
)

def get_checklist_eval_prompts(question: str, report: str, checklist: List[Dict[str, Any]]) -> List[str]:
    """
    Generate prompt for checklist-based evaluation.
    
    Args:
        question: The original question
        report: The generated report to evaluate
        checklist: List of checklist items with id, item, and weight
    
    Returns:
        List[str]: List of formatted evaluation prompts
    """
    prompts = []
    for item in checklist:
        prompt = CHECKLIST_EVAL_USER_PROMPT.format(question=question, report=report, criterion=item["item"])
        if item["weight"] < 0:
            prompt += NEGATIVE_REMINDER
        prompts.append(prompt)
    return prompts


###########################################################
# Citation-based evaluation prompts
# Claim extraction prompt is adapted from VeriScore
###########################################################

CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You will be provided with a research report (in <report></report> tags). The body of the report will contain many factual claims and citations to references. A section of the report will be highlighted in <highlighted_section></highlighted_section> tags.\n"
    "Your task is to extract all factual claims from and only from this highlighted section, along with the corresponding citation URLs if they exist.\n\n"
    "Extraction Guidelines:\n"
    "- You should ONLY extract claims from the highlighted section. Other parts of the report should only be used as context.\n"
    "- Each of these claims should be verifiable against external sources (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g.,\"would be\" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list.\n"
    "- All extracted claims should be standalone that can be understandable and verifiable without additional context. \n"
    "- You should preserve the original wording where possible, but provide necessary context to make the claim self-contained. Particularly, use the context to recover pronouns, anaphoric references (e.g. \"the paper\", \"the idea\"), and other such information to make the claim self-contained. Use the name of entities rather than anaphors whenever possible.\n"
    "- Along with the claims, you should also extract the corresponding citation URL(s) if they exist. Citations can be in different formats:\n"
    "  - A segment of text + [number], for example: \"Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels [15]\"\n"
    "  - A segment of text + [number†(some line numbers, etc.)], for example: \"Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels [15†L10][5L23][7†summary]\"\n"
    "  - [Citation Source](Citation Link), for example: \"Bolsonaro's rhetoric and frequent conflicting signals (e.g. encouraging gatherings) eroded public trust in institutions [pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC11042250/#:~:text=Conclusion).\"\n"
    "  If the citation format is among the first two, please refer to the references/sources section at the end to find the corresponding URLs for each claim.\n"
    "- If a claim has no corresponding citation to support it, return an empty list for the url field.\n"
    "- If multiple claims are associated with the same citation, extract them as separate entries. If a claim has multiple citations, include all citation URLs in the url list.\n\n"
    "Output format:\n"
    "Return a list of JSON objects of the following format: [{\"claim\": \"EXTRACTED CLAIM TEXT\", \"url\": [\"URL1\", \"URL2\", ...]}, ...].\n"
    "Output only the JSON list directly, without any chitchat or explanations. If the highlighted section does not contain any verifiable factual claims, please return an empty list. Please make sure the URLs are copied verbatim from the original citations. The \"url\" field should be an empty, single-item, or multi-item list."
)

CLAIM_EXTRACTION_USER_PROMPT = (
    "Extract the verifiable factual claims from the highlighted section of the report.\n"
    "<report>\n{report}\n</report>\n"
    "<highlighted_section>\n{highlighted_section}\n</highlighted_section>"
)


SUPPORTED_JUDGE_SYSTEM_PROMPT = (
    "You will be provided with a reference content (in <reference_content></reference_content> tags) and a claim or statement (in <claim></claim> tags). Your task is to determine whether the claim is 'supported', 'insufficient', or 'contradictory' with respect to the reference. Please note:\n"
    "- 'supported': the claim is clearly supported by the reference.\n"
    "- 'insufficient': the claim is weakly supported by the reference, or the reference is missing key evidence, or the claim is not related to the reference.\n"
    "- 'contradictory': the claim contradicts the reference.\n"
    "First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then the claim should be considered 'insufficient'. Then, carefully read the reference and the claim, and determine the relationship between the claim and the reference. The reference content can be from one or multiple webpages.\n\n"
    "Output Format: Return a JSON string with the following format: {\"result\": \"supported\" | \"insufficient\" | \"contradictory\"}. Do not include any other text or comments in your response. Please make sure the result is based purely on whether the claim is supported by the reference, not any other factors."
)

SUPPORTED_JUDGE_USER_PROMPT = (
    "Judge if the cited reference content supports the claim.\n"
    "<reference_content>\n{url_content}\n</reference_content>\n"
    "<claim>\n{claim}\n</claim>"
)

def get_citation_claim_extraction_prompts(report: str, highlighted_sections: List[str]) -> List[str]:
    """
    Generate prompts for extracting claims from report sections.
    
    Args:
        report: The generated report
        highlighted_sections: List of sections to extract claims from
    
    Returns:
        List[str]: List of formatted claim extraction prompts
    """
    prompts = []
    for section in highlighted_sections:
        prompt = CLAIM_EXTRACTION_USER_PROMPT.format(
            report=report.strip(),
            highlighted_section=section.strip(),
        )
        prompts.append(prompt)
    return prompts


def get_citation_supported_judge_prompts(claims: List[Dict[str, Any]]) -> List[str]:
    """
    Generate prompts for judging if claims are supported by citations.
    
    Args:
        claims: List of claim dictionaries, where each dict has:
            - "claim": str
            - "combined_content": str (concatenated content from all URLs)
    
    Returns:
        List[str]: List of formatted supported judgment prompts (one per claim)
    """
    prompts = []
    for claim in claims:
        prompt = SUPPORTED_JUDGE_USER_PROMPT.format(
            url_content=claim["combined_content"].strip(),
            claim=claim["claim"].strip(),
        )
        prompts.append(prompt)
    return prompts


###########################################################
# Webpage content summarization prompts
###########################################################

CONTENT_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a webpage summarization assistant. Your goal is to create a summary that preserves the most important information from the original web page. Given scraped webpage in markdown format (in <webpage_content></webpage_content> tags) and a list of claims (in <claims></claims> tags), "
    "extract and summarize the parts of the webpage that are relevant to the claims. "
    "Make sure you include all information that could support, contradict, or provide context for the claims. Also, preserve as much other key information in the webpage as possible to provide comprehensive context and be self-contained. "
    "Try to use the original wording of the webpage content as much as possible. "
    "If you find the webpage content is irrelevant to the claims, just generally summarize the web page content covering all key information. "
    "When you are summarizing, DO NOT use the third-person perspective (e.g. the webpage states that ..., the author says that ..., etc.). Just consider you are shortening the webpage as the author. "
    "Be as objective as possible and do not make any judgement or comments on the content. "
    "Aim for about 20 percent of the original length, unless the webpage is already concise."
)

CONTENT_SUMMARIZATION_USER_PROMPT = (
    "Summarize the webpage content that are potentially relevant to the claims.\n"
    "<webpage_content>\n{content}\n</webpage_content>\n"
    "<claims>\n{claims}\n</claims>\n"
    "Provide a summary of the webpage content. Preserve the original wording of the webpage content as much as possible, and include all meaningful details. Do not include any other text or explanations in your response."
)

def get_content_summarization_prompt(content: str, claims: List[Dict[str, Any]]) -> str:
    """
    Generate prompt for summarizing webpage content relevant to claims.
    
    Args:
        content: The webpage content to summarize
        claims: List of claim dictionaries with "id" and "claim" fields
    
    Returns:
        str: Formatted summarization prompt
    """
    claims_text = "\n".join([f"- {c['claim']}" for c in claims])
    return CONTENT_SUMMARIZATION_USER_PROMPT.format(content=content, claims=claims_text)

###########################################################
# Rubric-based evaluation prompts
###########################################################

RUBRIC_CHECKLISTS = {
    "presentation": [
        "Does the report follow a clear, logically ordered structure that is easy to navigate (e.g., problem → approach → results), with sections that match the report's stated purpose and directly addresses the research question?",
        "Do different sections logically follow or build on one another with minimal redundant restatement, and is any repetition clearly purposeful (e.g., brief recap before a new stage)?",
        "Where content is naturally parallel (steps, criteria, comparisons, key takeaways, etc.), does the report use lists and/or tables to present it in a scannable form rather than dense prose?",
        "Are headings/subheadings consistent in level and hierarchy (H1/H2/H3), and are comparable sections named with parallel phrasing (e.g., \"Method,\" \"Results\" rather than inconsistent mixes like \"How they did it,\" \"Findings\")?",
        "Does the report use concise transition sentences/phrases to signal why the proceeding content follows and to reduce abrupt jumps and make report easier to follow?",
        "If there are cross-references, are they consistent and unambiguous (figure/table numbers, section references, in-text citation), with no missing/duplicate numbering and no “see above/below” without an anchor? If no cross-references are present, score should be -1.",
        "If tables are included, are they structurally complete and interpretable on their own (no blank cells without notation, consistent units/precision, clear headers/labels/notes)? If no tables are included, score should be -1.",
        "Is report formatting correct and consistent (e.g., valid Markdown heading syntax, renderable Markdown tables, consistent numbering, consistent emphasis/code styling, consistent citation format if used)?",
        "Is the writing clear and professional at the sentence level (consistent tense/voice, minimal colloquialisms, avoids rhetorical exaggeration), with consistent terminology and abbreviation handling (define once, then reuse consistently)?",
        "Are key terms, symbols, and abbreviations formatted consistently (e.g., italicization, capitalization, acronym, bolding), and is there no “term drift” where the same concept is labeled multiple ways without intent?"
    ]
}

RUBRIC_EVAL_SYSTEM_PROMPT = (
    "You will be given a question the user asked (in <question></question> tags) and the corresponding report (in <report></report> tags) given as a response to the question by an assistant. You will then be given a specific criterion to evaluate the report against (in <criterion></criterion> tags).\n"
    "Your task is to score the report based on whether it satisfies the criterion or not: 1 if the report satisfies the criterion and 0 if the report does not satisfy the criterion. You might also be asked to give score=-1 when the criterion is not applicable to the report. Please do that when instructed. Judge only the specified aspect(s) in the criterion, not any other qualities of the report. Please also provide a short (2 sentences maximum) justification for your score.\n"
    "Output only a JSON string with the following format: {\"score\": int, \"justification\": string}. Do not include any other text or comments in your response."
)

RUBRIC_EVAL_USER_PROMPT = (
    "Evaluate the report based on the given criterion. If the criterion is not applicable to the report, score should be -1 instead of 0/1.\n"
    "<question>\n{question}\n</question>\n"
    "<report>\n{report}\n</report>\n"
    "<criterion>\n{criterion}\n</criterion>"
)

def get_rubric_checklist_items(rubric_dimension: str) -> List[Dict[str, Any]]:
    """
    Get checklist items for a rubric dimension.
    Rubrics are fixed (not question-specific) checklists with equal weights.
    
    Args:
        rubric_dimension: The rubric dimension (e.g., "presentation")
    
    Returns:
        List[Dict]: List of checklist items with id and item
    """
    
    if rubric_dimension not in RUBRIC_CHECKLISTS:
        raise ValueError(f"Unknown rubric dimension: {rubric_dimension}. Available dimensions: {list(RUBRIC_CHECKLISTS.keys())}")
    
    return RUBRIC_CHECKLISTS[rubric_dimension]


def get_rubric_eval_prompts(question: str, report: str, rubric_dimension: str) -> List[str]:
    """
    Generate prompts for rubric-based evaluation.
    
    Args:
        question: The original question
        report: The generated report to evaluate
        rubric_dimension: The rubric dimension to evaluate
    
    Returns:
        List[str]: List of formatted evaluation prompts for the rubric dimension's checklist items
    """
    checklist = get_rubric_checklist_items(rubric_dimension)
    prompts = []
    for item in checklist:
        prompt = RUBRIC_EVAL_USER_PROMPT.format(question=question, report=report, criterion=item)
        prompts.append(prompt)
    return prompts

###########################################################
# Pairwise feedback incorporation judgment prompts
###########################################################

PAIRWISE_JUDGE_SYSTEM_PROMPT = (
    "You will be given a research question the user asked (in <question></question> tags) and two versions of the report, original (in <report></report> tags) and revised (in <revised_report></revised_report> tags) that are generated by an assistant. The revised report is a revised version of the original report based on the feedback (in <feedback></feedback> tags) provided by the user.\n"
    "Your task is to score the revised report based on whether it incorporates the feedback provided by the user or not, comparing it to the original report: 1.0 if the revised report incorporates the feedback, 0.5 if the revised report partially incorporates the feedback, and 0.0 if the revised report does not incorporate the feedback.\n"
    "Output only a JSON string with the following format: {\"score\": float}. Do not include any other text or comments in your response. Please make sure the score is based purely on whether the feedback is reflected in the revised report compared to the original report, not any other factors."
)

PAIRWISE_JUDGE_USER_PROMPT = (
    "Score the revised report based on whether it incorporates the feedback provided by the user compared to the original report.\n"
    "<question>\n{question}\n</question>\n"
    "<report>\n{report}\n</report>\n"
    "<revised_report>\n{revised_report}\n</revised_report>\n"
    "<feedback>\n{feedback}\n</feedback>"
)