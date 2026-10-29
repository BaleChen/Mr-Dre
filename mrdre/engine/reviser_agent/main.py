import asyncio
import json
import logging
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

import httpx

from .prompts import SYSTEM_PROMPT, build_user_message
from .tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)


class ReviserAgent:
    """
    A simple LLM agent that can revise research reports using web search.

    Uses vLLM's OpenAI-compatible API with function calling support.
    Follows the same interface pattern as other DRA agents (__call__ + poll).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8005/v1",
        model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        max_tool_calls: int = 10,
        timeout: float = 1200.0,
    ):
        """
        Initialize the Reviser Agent.

        Args:
            base_url: vLLM server URL (default: http://localhost:8005/v1)
            model: Model name on the vLLM server
            max_tool_calls: Maximum number of tool calls allowed (default: 10)
            timeout: HTTP request timeout in seconds (default: 1200)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout
        self._results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    async def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Make an async chat completion request to the vLLM server."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 16384,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def _revise_async(
        self,
        question: str,
        report: str,
        feedback: str,
    ) -> Dict[str, Any]:
        """Internal async method to revise a report with agentic loop."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_tool_calls=self.max_tool_calls)},
            {"role": "user", "content": build_user_message(question, report, feedback)},
        ]

        tool_call_history = []
        tool_call_count = 0

        while tool_call_count < self.max_tool_calls:
            response = await self._chat_completion(messages, tools=TOOLS)
            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "")
            tool_calls = message.get("tool_calls", [])

            if not tool_calls or finish_reason == "stop":
                return {
                    "report": message.get("content", ""),
                    "tool_calls": tool_call_history,
                }

            messages.append(message)

            for tool_call in tool_calls:
                tool_call_count += 1
                tool_id = tool_call.get("id", f"call_{tool_call_count}")
                if tool_call_count > self.max_tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": "You have reached the maximal number of web search calls. Please now produce the revised report based on the information you have and the user feedback.",
                    })
                    break
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")

                arguments_raw = function.get("arguments", "{}")
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = arguments_raw

                logger.info(f"Tool call [{tool_call_count}]: {tool_name}({arguments})")
                result = execute_tool(tool_name, arguments)

                tool_call_history.append({
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })

        # Final request without tools after max tool calls
        response = await self._chat_completion(messages, tools=None)
        return {
            "report": response["choices"][0]["message"].get("content", ""),
            "tool_calls": tool_call_history,
        }

    def _run_in_thread(self, request_id: str, question: str, report: str, feedback: str):
        """Run the async revision in a separate thread with its own event loop."""
        async def run():
            try:
                result = await self._revise_async(question, report, feedback)
                with self._lock:
                    self._results[request_id] = {"status": "completed", "result": result}
            except Exception as e:
                logger.error(f"Request {request_id} failed: {e}")
                with self._lock:
                    self._results[request_id] = {"status": "failed", "error": str(e)}

        # Create a new event loop for this thread and run the coroutine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

    def __call__(
        self,
        question: str,
        report: str,
        feedback: str,
    ) -> str:
        """
        Start a revision task (non-blocking).

        Args:
            question: The original research question
            report: The current research report to revise
            feedback: User feedback on how to improve the report

        Returns:
            str: Request ID for tracking the task
        """
        request_id = str(uuid.uuid4())

        with self._lock:
            self._results[request_id] = {"status": "pending"}

        # Start the task in a background thread
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(request_id, question, report, feedback),
            daemon=True,
        )
        thread.start()

        return request_id

    def poll(self, request_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Poll a request to check if it's completed.

        Args:
            request_id: ID of the request to poll

        Returns:
            tuple: (is_completed, result)
                - is_completed: True if the request has finished
                - result: The result dict if completed, None otherwise
        """
        with self._lock:
            entry = self._results.get(request_id)

        if entry is None:
            return True, {"error": f"Unknown request_id: {request_id}"}

        if entry["status"] == "completed":
            return True, entry["result"]
        elif entry["status"] == "failed":
            raise RuntimeError(f"Request {request_id} failed: {entry['error']}")
        else:
            # Still pending
            return False, None

    def wait_for_completion(self, request_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        """
        Block until a request completes and return the result.

        Args:
            request_id: ID of the request to wait for
            poll_interval: Seconds between polls (default: 1.0)

        Returns:
            The result dictionary
        """
        while True:
            is_done, result = self.poll(request_id)
            if is_done:
                return result
            time.sleep(poll_interval)
