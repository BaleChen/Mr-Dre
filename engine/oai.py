"""OpenAI Deep Research Agent implementation."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from . import DeepResearchAgentBase
from .cache import cached


class OpenAIDeepResearchAgent(DeepResearchAgentBase):
    """
    Deep Research Agent for OpenAI's background research models.

    Uses OpenAI's Responses API with background mode for long-running
    research tasks with web search capabilities.

    Example:
        >>> agent = OpenAIDeepResearchAgent(model="o4-mini-deep-research")
        >>> request_id = agent([
        ...     {"role": "user", "content": "Research topic..."},
        ... ])
        >>> result = agent.poll(request_id)
    """

    def __init__(
        self,
        model: str = "o4-mini-deep-research-2025-06-26",
        poll_interval: int = 5,
    ):
        """
        Initialize the OpenAI Deep Research Agent.

        Args:
            model: OpenAI model identifier.
            poll_interval: Seconds between status checks when polling.
        """
        self.client = openai.OpenAI(timeout=3600)
        self.model = model
        self.poll_interval = poll_interval

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=60),
    )
    def __call__(
        self,
        input_data: Union[str, List[dict]],
        previous_response_id: Optional[str] = None,
    ) -> str:
        """
        Start a deep research task in background mode.

        Args:
            input_data: User prompt string or list of message dicts.
            previous_response_id: Optional ID to continue a conversation.

        Returns:
            Response ID for tracking the background task.
        """
        response = self.client.responses.create(
            model=self.model,
            input=input_data,
            previous_response_id=previous_response_id,
            tools=[{"type": "web_search_preview"}],
            background=True,
        )
        return response.id

    def wait_for_completion(self, response_id: str) -> dict:
        """
        Block until the background task completes.

        Args:
            response_id: ID of the background task.

        Returns:
            Processed response dictionary.
        """
        while True:
            is_completed, result = self.poll(response_id)
            if is_completed:
                return result
            time.sleep(self.poll_interval)

    def poll(self, response_id: str) -> Tuple[bool, Optional[dict]]:
        """
        Check task status without blocking.

        Args:
            response_id: ID of the background task.

        Returns:
            Tuple of (is_completed, result).

        Raises:
            Exception: If the task failed, was canceled, or has unexpected status.
        """
        response = self.client.responses.retrieve(response_id)
        status = response.status

        if status == "completed":
            return True, self._postprocess_response(response)
        if status in ("queued", "in_progress"):
            return False, None
        if status in ("failed", "incomplete", "canceled"):
            raise Exception(f"Task {status}: {response_id}")
        raise Exception(f"Unexpected status '{status}': {response_id}")

    def get_response(self, response_id: str) -> Tuple[str, Any]:
        """
        Get the current status and raw response.

        Args:
            response_id: ID of the background task.

        Returns:
            Tuple of (status, raw_response).
        """
        response = self.client.responses.retrieve(response_id)
        return response.status, response

    def _postprocess_response(self, response: Any) -> dict:
        """
        Convert OpenAI response to unified format.

        Args:
            response: Raw OpenAI response object.

        Returns:
            Standardized response dictionary with report, citations, usage, and metadata.
        """
        tool_call_count = 0
        citations = []

        for item in response.output:
            if item.type in (
                "web_search_call",
                "code_interpreter_call",
                "file_search_call",
                "mcp_tool_call",
            ):
                tool_call_count += 1
            elif item.type == "message":
                citations = [anno.to_dict() for anno in item.content[0].annotations]

        return {
            "resp_id": response.id,
            "report": response.output_text,
            "citations": citations,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "cached_input_tokens": 0,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
                "total_tokens": response.usage.total_tokens,
                "tool_call_count": tool_call_count,
                "cost": {}, # TODO: add cost calculation
            },
            "metadata": {
                "previous_response_id": response.previous_response_id,
                "reasoning": response.reasoning.to_dict(),
                **response.metadata,
            },
        }


@cached()
def _cached_gpt_call(model_name: str, system_prompt: str, user_prompt: str) -> str:
    """
    Cached GPT call for cost saving and deterministic requests.

    Cache key is based on model_name, system_prompt, and user_prompt.
    """
    client = openai.OpenAI()
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    response = client.responses.create(
        model=model_name,
        input=messages,
        temperature=0,
    )
    return response.output_text


class GPT:
    """
    Simple GPT wrapper for general text generation tasks.

    Example:
        >>> gpt = GPT("gpt-4o", system_prompt="You are a helpful assistant.")
        >>> response = gpt("Explain quantum entanglement briefly.")
    """

    def __init__(self, model_name: str, system_prompt: str = ""):
        """
        Initialize the GPT wrapper.

        Args:
            model_name: OpenAI model identifier.
            system_prompt: Optional system prompt for all calls.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=60),
    )
    def __call__(self, user_prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            user_prompt: User's input prompt.

        Returns:
            Generated response text.
        """
        return _cached_gpt_call(self.model_name, self.system_prompt, user_prompt)
