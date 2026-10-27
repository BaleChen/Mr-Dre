"""
Deep Research Agent (DRA) Engine.

Provides a unified interface and router for different deep research agent implementations
"""

import importlib
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union


class DeepResearchAgentBase(ABC):
    """
    Abstract base class for Deep Research Agent implementations.

    All deep research agents must inherit from this class and implement
    the required methods to ensure a consistent interface across providers.
    """

    @abstractmethod
    def __init__(self, model: str, **kwargs):
        """
        Initialize the deep research agent.

        Args:
            model: Model name/identifier.
            **kwargs: Provider-specific initialization parameters.
        """
        pass

    @abstractmethod
    def __call__(self, input_data: Union[str, List[dict]], **kwargs) -> str:
        """
        Start a deep research task in async mode (non-blocking).

        Args:
            input_data: User prompt as a string, or a messages list.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Request/response ID for tracking the background task.
        """
        pass

    @abstractmethod
    def wait_for_completion(self, request_id: str) -> dict:
        """
        Block until the async request completes and return the final response.

        Args:
            request_id: ID of the async request.

        Returns:
            Processed response dictionary.
        """
        pass

    @abstractmethod
    def get_response(self, request_id: str) -> Tuple[str, Any]:
        """
        Get the current status and raw response of an async request.

        Args:
            request_id: ID of the async request.

        Returns:
            Tuple of (status, response_object).
        """
        pass

    @abstractmethod
    def poll(self, request_id: str) -> Tuple[bool, Optional[dict]]:
        """
        Check if an async request has completed without blocking.

        Args:
            request_id: ID of the async request.

        Returns:
            Tuple of (is_completed, result). Result is the processed response
            if completed, None otherwise.
        """
        pass

    def _postprocess_response(self, response: Any) -> dict:
        """
        Postprocess the raw response into a unified format.

        Override in subclasses for provider-specific processing.

        Args:
            response: Raw response object from the provider.

        Returns:
            Processed response dictionary.
        """
        return response


class SimpleDeepResearchAgent(DeepResearchAgentBase):
    """
    Base class for plugging in a DRA that runs synchronously (e.g., a local agent loop
    or a blocking HTTP endpoint). Subclasses implement only `generate`; this class runs
    each request in a thread pool and provides the asynchronous interface that
    call_dra.py expects (`__call__`, `poll`, `wait_for_completion`, `get_response`).

    Example:
        >>> class MyAgent(SimpleDeepResearchAgent):
        ...     def generate(self, messages):
        ...         report = my_agent.run(messages)  # your own agent
        ...         return {"report": report}
    """

    def __init__(self, model: str, max_workers: int = 10, **kwargs):
        """
        Args:
            model: Model name/identifier (the name passed to --model).
            max_workers: Maximum number of requests generated concurrently.
            **kwargs: Ignored; accepted for compatibility with the DRA router.
        """
        self.model = model
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}

    @abstractmethod
    def generate(self, messages: List[dict]) -> Union[str, dict]:
        """
        Produce a report for a conversation.

        Args:
            messages: OpenAI-style messages. The first is the user question; in later turns
                the list alternates previous reports (assistant) and feedback (user),
                ending with the current feedback.

        Returns:
            The report text, or a dict with at least a "report" key. Cited sources should
            appear in the report as URLs (inline links or a references section), since the
            citation evaluation extracts them from the report text.
        """
        pass

    def __call__(self, input_data: Union[str, List[dict]], **kwargs) -> str:
        messages = [{"role": "user", "content": input_data}] if isinstance(input_data, str) else input_data
        request_id = str(uuid.uuid4())
        self._futures[request_id] = self._executor.submit(self.generate, messages)
        return request_id

    def poll(self, request_id: str) -> Tuple[bool, Optional[dict]]:
        future = self._futures[request_id]
        if not future.done():
            return False, None
        # Re-raises any exception from generate(), which call_dra.py records as a failure
        result = self._postprocess_response(future.result())
        del self._futures[request_id]
        return True, result

    def wait_for_completion(self, request_id: str) -> dict:
        self._futures[request_id].result()
        return self.poll(request_id)[1]

    def get_response(self, request_id: str) -> Tuple[str, Any]:
        future = self._futures[request_id]
        if not future.done():
            return "in_progress", None
        return ("failed", future.exception()) if future.exception() else ("completed", future.result())

    def _postprocess_response(self, response: Union[str, dict]) -> dict:
        if isinstance(response, str):
            return {"report": response}
        if "report" not in response:
            raise ValueError("generate() must return the report text or a dict with a 'report' key")
        return response


# Provider registry: maps model prefix/pattern to (module_name, class_name, case_sensitive)
_PROVIDER_REGISTRY: List[Tuple[str, str, str, bool]] = [
    ("sonar-", "perplexity", "SonarDeepResearchAgent", True),
    ("odr-", "odr", "OpenDeepResearchAgent", True),
    ("dr-tulu", "dr_tulu", "DrTuluDeepResearchAgent", True),
    ("tongyi", "tongyi", "TongyiDeepResearchAgent", True),
    # Add new providers here (see docs/evaluate_your_dra.md and engine/template.py):
    # ("my-dra", "template", "MyDeepResearchAgent", True),
]

# OpenAI models require exact matching
_OPENAI_MODELS = {"o4-mini-deep-research", "o3-deep-research"}


class DRA:
    """
    Deep Research Agent (DRA) router.

    Routes requests to the appropriate provider-specific agent based on model name.

    Supported providers:
        - OpenAI: "o4-mini-deep-research", "o3-deep-research"
        - Perplexity: "sonar-*" (e.g., "sonar-deep-research")
        - Open Deep Research: "odr-*"
        - DR-Tulu: "dr-tulu*" (case-insensitive)
        - Tongyi: "tongyi*" (case-insensitive)
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the DRA router.

        Args:
            model_name: Model identifier used to select the appropriate agent.
            **kwargs: Arguments passed to the underlying agent.
        """
        self.model_name = model_name
        self.agent = self._create_agent(model_name, **kwargs)

    def _create_agent(self, model_name: str, **kwargs) -> DeepResearchAgentBase:
        """
        Create the appropriate agent instance based on model name.

        Args:
            model_name: Model identifier.
            **kwargs: Agent initialization parameters.

        Returns:
            Instance of the appropriate agent class.

        Raises:
            ValueError: If model_name doesn't match any known provider.
        """
        # Check OpenAI exact matches
        if model_name in _OPENAI_MODELS:
            from .oai import OpenAIDeepResearchAgent
            return OpenAIDeepResearchAgent(model=model_name, **kwargs)

        # Check prefix-based provider registry
        model_lower = model_name.lower()
        for prefix, module_name, class_name, case_sensitive in _PROVIDER_REGISTRY:
            check_name = model_name if case_sensitive else model_lower
            check_prefix = prefix if case_sensitive else prefix.lower()
            if check_name.startswith(check_prefix):
                module = importlib.import_module(f".{module_name}", package=__package__)
                agent_class = getattr(module, class_name)
                return agent_class(model=model_name, **kwargs)

        # Build helpful error message
        supported = list(_OPENAI_MODELS) + [p[0] + "*" for p in _PROVIDER_REGISTRY]
        raise ValueError(
            f"Unknown model: '{model_name}'. Supported patterns: {supported}"
        )

    def __call__(self, input_data: Union[str, List[dict]], **kwargs) -> str:
        """Start a deep research task. See DeepResearchAgentBase.__call__."""
        return self.agent(input_data, **kwargs)

    def wait_for_completion(self, request_id: str) -> dict:
        """Wait for completion. See DeepResearchAgentBase.wait_for_completion."""
        return self.agent.wait_for_completion(request_id)

    def get_response(self, request_id: str) -> Tuple[str, Any]:
        """Get response status. See DeepResearchAgentBase.get_response."""
        return self.agent.get_response(request_id)

    def poll(self, request_id: str) -> Tuple[bool, Optional[dict]]:
        """Poll for completion. See DeepResearchAgentBase.poll."""
        return self.agent.poll(request_id)


# TODO: add back when releasing reviser agent
# from .reviser_agent import ReviserAgent

__all__ = ["DRA", "DeepResearchAgentBase", "SimpleDeepResearchAgent"] # "ReviserAgent" TODO: add back when releasing reviser agent

