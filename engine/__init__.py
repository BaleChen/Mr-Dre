"""
Deep Research Agent (DRA) Engine.

Provides a unified interface and router for different deep research agent implementations
"""

import importlib
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union


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


# Provider registry: maps model prefix/pattern to (module_name, class_name, case_sensitive)
_PROVIDER_REGISTRY: List[Tuple[str, str, str, bool]] = [
    ("sonar-", "perplexity", "SonarDeepResearchAgent", True),
    ("odr-", "odr", "OpenDeepResearchAgent", True),
    ("dr-tulu", "dr_tulu", "DrTuluDeepResearchAgent", True),
    ("tongyi", "tongyi", "TongyiDeepResearchAgent", True),
    # Add new providers here:
    # ("my-provider-", "my_module", "MyProviderAgent", True),
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

__all__ = ["DRA", "DeepResearchAgentBase"] # "ReviserAgent" TODO: add back when releasing reviser agent

