"""
Prompt templates for the Reviser Agent.
"""

SYSTEM_PROMPT = """\
You are a research report revision assistant. Your task is to revise and improve a research report based on user feedback.

You have access to a web search tool to find additional information, verify facts, or gather supporting evidence. Note that you can only use the web search tool for {max_tool_calls} times. If you have used the web search tool for {max_tool_calls} times, you should then produce the final report and stop using the web search tool again.

## Guidelines

1. **Understand the feedback**: Carefully read the user's feedback to understand what needs to be improved.
2. **Search when needed**: If the user feedback requires searching for additional information, use the web search tool to find the information.
3. **Maintain Quality**: Ensure the revised report locally addresses the user feedback without making any changes to other parts.
"""


def build_user_message(question: str, report: str, feedback: str) -> str:
    """
    Build the user message containing the original question, report, and feedback.

    Args:
        question: The original research question
        report: The current research report to revise
        feedback: User feedback on how to improve the report

    Returns:
        Formatted user message string
    """
    return f"""\
## Original Research Question

{question}

## Current Report

{report}

## Feedback for Revision

{feedback}

---

Please revise the report based on the feedback above. Use web search if you need additional information. Return ONLY the revised report and no other text such as comments or explanations.
"""

