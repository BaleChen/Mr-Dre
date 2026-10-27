"""
Template for plugging a new Deep Research Agent into Mr Dre.

Steps:
    1. Copy this file (e.g., to engine/my_dra.py) and implement `generate`.
    2. Register it in `_PROVIDER_REGISTRY` in engine/__init__.py, e.g.:
           ("my-dra", "my_dra", "MyDeepResearchAgent", True),
    3. Generate reports with: ./scripts/generate_report.sh ResearcherBench my-dra 1 init

See docs/evaluate_your_dra.md for the full guide.
"""

from typing import List, Union

from . import SimpleDeepResearchAgent


class MyDeepResearchAgent(SimpleDeepResearchAgent):
    """Minimal example of a synchronous DRA."""

    def __init__(self, model: str = "my-dra", **kwargs):
        super().__init__(model, **kwargs)
        # Initialize your client, tools, or local model here.

    def generate(self, messages: List[dict]) -> Union[str, dict]:
        """
        Args:
            messages: [{"role": "user", "content": question},
                       {"role": "assistant", "content": report_turn1},
                       {"role": "user", "content": feedback_turn2}, ...]

        Returns:
            The report text, or a dict such as
            {"report": str, "citations": list, "usage": dict, "metadata": dict}.
            Only "report" is required. Cite sources with URLs in the report text.
        """
        raise NotImplementedError("Call your Deep Research Agent here and return its report.")


__all__ = ["MyDeepResearchAgent"]
