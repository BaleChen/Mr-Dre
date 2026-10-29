"""Perplexity Sonar Deep Research Agent implementation."""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from . import DeepResearchAgentBase


class SonarDeepResearchAgent(DeepResearchAgentBase):
    """
    Deep Research Agent for Perplexity's Sonar models.

    Example:
        >>> agent = SonarDeepResearchAgent()
        >>> request_id = agent([
        ...     {"role": "user", "content": "Research topic..."},
        ... ])
        >>> result = agent.poll(request_id)
    """

    ASYNC_BASE_URL = "https://api.perplexity.ai/async/chat/completions"

    def __init__(
        self,
        model: str = "sonar-deep-research",
        api_key: Optional[str] = None,
        timeout: int = 600,
        poll_interval: int = 5,
        session: Optional[requests.Session] = None,
    ):
        """
        Initialize the Sonar Deep Research Agent.

        Args:
            model: Perplexity model identifier.
            api_key: API key. Falls back to PERPLEXITY_API_KEY env var.
            timeout: Request timeout in seconds.
            poll_interval: Seconds between status checks when polling.
            session: Optional requests session for connection reuse.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.model = model
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required: provide via api_key or PERPLEXITY_API_KEY env var"
            )
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = session or requests.Session()
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def __call__(self, input_data: Union[str, List[dict]]) -> str:
        """
        Start a Sonar Deep Research task in async mode.

        Args:
            input_data: User prompt string or list of message dicts.

        Returns:
            Request ID for tracking the background task.

        Raises:
            TypeError: If input_data is neither a string nor a list.
        """
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            raise TypeError("input_data must be a string or list of message dicts")

        payload = {"request": {"model": self.model, "messages": messages}}
        return self._submit_request(payload)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=5, min=5, max=60),
    )
    def _submit_request(self, payload: dict) -> str:
        """Submit async request with retry logic."""
        response = self.session.post(
            self.ASYNC_BASE_URL,
            json=payload,
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "FAILED":
            raise RuntimeError(f"Async request failed: {data}")
        return data.get("id")

    def get_response(self, request_id: str) -> Tuple[str, dict]:
        """
        Get the current status and raw response.

        Args:
            request_id: ID of the async request.

        Returns:
            Tuple of (status, response_data).
        """
        url = f"{self.ASYNC_BASE_URL}/{request_id}"
        response = self.session.get(url, headers=self._headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("status"), data

    def wait_for_completion(self, request_id: str) -> dict:
        """
        Block until the async request completes.

        Args:
            request_id: ID of the async request.

        Returns:
            Processed response dictionary.
        """
        while True:
            is_completed, result = self.poll(request_id)
            if is_completed:
                return result
            time.sleep(self.poll_interval)

    def poll(self, request_id: str) -> Tuple[bool, Optional[dict]]:
        """
        Check request status without blocking.

        Args:
            request_id: ID of the async request.

        Returns:
            Tuple of (is_completed, result).

        Raises:
            RuntimeError: If the request failed or was canceled.
        """
        status, data = self.get_response(request_id)

        if status == "COMPLETED":
            return True, self._postprocess_response(data["response"])

        if status in ("FAILED", "CANCELED", "ERROR") or data.get("failed_at"):
            error_msg = data.get("error_message", f"Request {status}")
            raise RuntimeError(f"Async request failed: {error_msg}")

        return False, None

    def _postprocess_response(self, response: dict) -> dict:
        """
        Convert Perplexity response to unified format.

        Args:
            response: Raw Perplexity response dict.

        Returns:
            Standardized response dictionary with report, citations, usage, and metadata.
        """
        thinking, report = self._parse_thinking(
            response["choices"][0]["message"]["content"]
        )

        # Append formatted citations since Perplexity doesn't include them in the report
        citations = response.get("search_results", [])
        report += self._format_sources_section(citations)

        usage = response.get("usage", {})
        return {
            "resp_id": response.get("id"),
            "report": report,
            "citations": citations,
            "usage": {
                "input_tokens": usage.get("prompt_tokens"),
                "cached_input_tokens": 0,
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "tool_call_count": usage.get("num_search_queries"),
                "cost": usage.get("cost"),
            },
            "metadata": {"thinking": thinking},
        }

    @staticmethod
    def _parse_thinking(content: str) -> Tuple[str, str]:
        """Extract thinking tags from content."""
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            thinking = match.group(1)
            report = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return thinking, report
        return "", content

    @staticmethod
    def _format_sources_section(citations: List[dict]) -> str:
        """Format citations as a sources section."""
        if not citations:
            return ""
        lines = ["\n\n**Sources:**"]
        for i, cite in enumerate(citations, 1):
            lines.append(f"[{i}] {cite.get('title', 'Untitled')} ({cite.get('url', '')})")
        return "\n".join(lines)


__all__ = ["SonarDeepResearchAgent"]
