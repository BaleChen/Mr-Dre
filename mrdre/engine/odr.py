"""LangChain Open Deep Research client for a LangGraph server that you run."""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import DeepResearchAgentBase

logger = logging.getLogger(__name__)

# Errors after which the session is rebuilt and the request is sent again
_CONNECTION_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
    ConnectionAbortedError,
)


class OpenDeepResearchAgent(DeepResearchAgentBase):
    """
    Client for an Open Deep Research LangGraph server.

    Each request creates a thread on the server and starts a run of the "Deep Researcher"
    graph on it with the conversation as input. The server runs the research in the
    background and decides which models and search tools it uses; poll() checks the run
    and, once it succeeds, reads the final report from the thread state.
    """

    def __init__(
        self,
        model: str = "odr",
        *,
        api_url: Optional[str] = None,
        assistant_id: Optional[str] = None,
        timeout: int = 600,
        poll_interval: int = 5,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        """
        Initialize the client and find the assistant of the Deep Researcher graph.

        Args:
            model: Engine name; the server's configuration sets the models it uses.
            api_url: Server URL. Falls back to the LANGGRAPH_API_URL env var, then
                http://localhost:2025.
            assistant_id: Assistant to run. Defaults to the one of the "Deep Researcher" graph.
            timeout: Timeout of each HTTP request in seconds.
            poll_interval: Seconds between status checks in wait_for_completion.
            max_retries: Retries of a request after a connection error.
            retry_backoff: Base of the exponential backoff between retries, in seconds.
        """
        self.model = model
        self.api_url = (api_url or os.getenv("LANGGRAPH_API_URL", "http://localhost:2025")).rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = self._create_session()
        self.assistant_id = assistant_id or self._resolve_assistant()
        # Thread of each pending run: run_id -> thread_id
        self._pending_runs: Dict[str, str] = {}

    def _create_session(self) -> requests.Session:
        """Create a session that also retries server errors (502, 503, 504)."""
        retry_strategy = Retry(
            total=self.max_retries,
            # A request that timed out may have reached the server, so it is not sent again
            read=False,
            backoff_factor=self.retry_backoff,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """
        Send a request to the server and return its JSON body.

        After a connection error, the session is rebuilt and the request is sent again, up to
        max_retries times with exponential backoff.

        Raises:
            ConnectionError: If the server is still unreachable after the retries.
            requests.HTTPError: If the server answers with an error status.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, f"{self.api_url}{path}", timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response.json()
            except _CONNECTION_ERRORS as e:
                if attempt == self.max_retries:
                    raise ConnectionError(
                        f"LangGraph server at {self.api_url} unreachable after {self.max_retries + 1} attempts"
                    ) from e
                wait_time = self.retry_backoff * (2 ** attempt)
                logger.warning(
                    f"Connection error on {method} {path} (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                    f"Retrying in {wait_time:.1f}s"
                )
                self.session.close()
                self.session = self._create_session()
                time.sleep(wait_time)

    def _resolve_assistant(self) -> str:
        """Find the assistant of the "Deep Researcher" graph, or else the first assistant."""
        assistants = self._request("POST", "/assistants/search", json={})
        if not assistants:
            raise RuntimeError(f"No assistants found at {self.api_url}")
        for assistant in assistants:
            if assistant.get("graph_id") == "Deep Researcher":
                return assistant["assistant_id"]
        return assistants[0]["assistant_id"]

    def __call__(self, input_data: Union[str, List[dict]]) -> str:
        """
        Start a research run on a new thread and return its run ID.

        Args:
            input_data: User prompt as a string, or a list of message dicts.

        Returns:
            Run ID for tracking the request.
        """
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            raise TypeError("input_data must be a string or a list of message dicts")

        thread_id = self._request("POST", "/threads", json={})["thread_id"]
        run = self._request(
            "POST",
            f"/threads/{thread_id}/runs",
            json={"assistant_id": self.assistant_id, "input": {"messages": messages}},
        )
        self._pending_runs[run["run_id"]] = thread_id
        return run["run_id"]

    def get_response(self, request_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current status and run information of a run.

        Args:
            request_id: Run ID returned by __call__.

        Returns:
            Tuple of (status, run_info).
        """
        thread_id = self._pending_runs.get(request_id)
        if thread_id is None:
            raise ValueError(f"No pending run for id '{request_id}'")
        run_info = self._request("GET", f"/threads/{thread_id}/runs/{request_id}")
        return run_info.get("status"), run_info

    def poll(self, request_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check whether a run has finished without blocking.

        Args:
            request_id: Run ID returned by __call__.

        Returns:
            Tuple of (is_completed, result). Result is the processed response if completed,
            None otherwise.

        Raises:
            RuntimeError: If the run failed.
        """
        status, run_info = self.get_response(request_id)
        if status in ("error", "interrupted", "timeout"):
            self._pending_runs.pop(request_id, None)
            raise RuntimeError(f"Run {request_id} failed with status: {status}")
        if status != "success":
            return False, None

        thread_id = self._pending_runs[request_id]
        state = self._request("GET", f"/threads/{thread_id}/state")
        self._pending_runs.pop(request_id, None)
        return True, self._postprocess_state(request_id, thread_id, state, run_info)

    def wait_for_completion(self, request_id: str) -> Dict[str, Any]:
        """
        Block until a run finishes and return the processed response.

        Args:
            request_id: Run ID returned by __call__.

        Returns:
            Processed response with the report and metadata.
        """
        while True:
            is_completed, result = self.poll(request_id)
            if is_completed:
                return result
            time.sleep(self.poll_interval)

    def _postprocess_state(
        self, run_id: str, thread_id: str, state: Dict[str, Any], run_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Take the final report from the thread state, or else the last AI message."""
        values = state.get("values") or {}
        report = values.get("final_report") or self._extract_last_ai_message(values.get("messages"))
        return {
            "resp_id": run_id,
            "report": report or "",
            "citations": [],
            # The LangGraph server does not report token usage
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "num_tool_calls": 0,
                "cost": {},
            },
            "metadata": {
                "thread_id": thread_id,
                "assistant_id": run_info.get("assistant_id"),
                "research_brief": values.get("research_brief"),
                "notes": values.get("notes", []),
                "raw_notes": values.get("raw_notes", []),
            },
        }

    def _extract_last_ai_message(self, messages: Any) -> str:
        """Return the text of the last AI message, or an empty string."""
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if message.get("type") in {"ai", "assistant"}:
                content = message.get("content", "")
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                    )
                return str(content)
        return ""


__all__ = ["OpenDeepResearchAgent"]
