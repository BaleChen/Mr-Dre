import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from . import DeepResearchAgentBase

PromptType = Sequence[Dict[str, str]]

###########################################################
# Helper functions for parsing and formatting citations   # 
# The functions are copied from the drb_formatter.py file #
###########################################################

def parse_and_format_citations(text: str, snippet_map: Dict[str, Dict[str, str]]) -> str:
    """
    Parses a string with custom <cite> tags, extracts citation IDs,
    and reformats the string with numbered citations and Sources section.

    Args:
        text: The input string containing <cite> tags.
        snippet_map: A dictionary mapping citation IDs to their metadata (title, url).

    Returns:
        The formatted string with numbered citations in-text and a Sources section appended.
    """
    citation_ids: List[str] = []
    id_to_text_map: Dict[str, List[str]] = {}
    id_to_number: Dict[str, int] = {}

    def _ensure_number(cid: str) -> int:
        if cid not in id_to_number:
            id_to_number[cid] = len(id_to_number) + 1
            citation_ids.append(cid)
        return id_to_number[cid]

    def replacer(match):
        ids = [cid.strip() for cid in match.group(1).split(",") if cid.strip()]
        text_content = match.group(2).strip()

        for cid in ids:
            id_to_text_map.setdefault(cid, []).append(text_content)

        bracket_str = "".join(f"[{_ensure_number(cid)}]" for cid in ids)
        if not text_content:
            return bracket_str
        return f"{text_content} {bracket_str}"

    pattern = re.compile(r'<cite id="([^"]+)">(.*?)</cite>', re.DOTALL)
    formatted_text = pattern.sub(replacer, text)
    
    # Build references from snippet_map
    references: List[str] = []
    for idx, cid in enumerate(citation_ids, start=1):
        source = snippet_map.get(cid, {})
        url = source.get("url", "")
        title = source.get("title") or source.get("Title") or url or cid
        references.append(f"[{idx}] {title}{f' ({url})' if url else ''}")
    
    # Append Sources section if there are references
    if references:
        body = formatted_text.strip()
        body += "\n\n**Sources:**\n" + "\n".join(references)
        return body
    else:
        return formatted_text


def parse_search_results(text: str) -> List[Dict[str, str]]:
    """
    Parses flat search result text into structured entries.
    """
    pattern = re.compile(
        r"Title: (.*?)\n"
        r"(?:URL: (.*?)\n)?"
        r"Snippet: (.*?)"
        r"(?=\nTitle: |\Z)",
        re.DOTALL,
    )
    matches = pattern.findall(text or "")
    results = []
    for title, url, snippet in matches:
        results.append(
            {
                "Title": title.strip(),
                "URL": (url or "").strip(),
                "Snippet": snippet.strip(),
            }
        )
    return results


def parse_xml_snippets(text: str) -> Dict[str, Dict[str, str]]:
    """
    Parses <snippet> blocks into a mapping keyed by snippet id.
    """
    pattern = re.compile(
        r"<snippet id=([^>]+)>\s*Title: (.*?)\s*(?:URL: (.*?))?\s*Snippet: (.*?)\s*</snippet>",
        re.DOTALL,
    )
    snippets: Dict[str, Dict[str, str]] = {}
    for match in pattern.findall(text or ""):
        snippet_id, title, url, snippet = match
        snippets[snippet_id.strip()] = {
            "Title": title.strip(),
            "URL": (url or "").strip(),
            "Snippet": snippet.strip(),
        }
    return snippets

###########################################################
# DrTuluDeepResearchAgent class                           # 
###########################################################


class DrTuluDeepResearchAgent(DeepResearchAgentBase):
    """
    Client for the local DR-Tulu auto_search server.

    The server exposes a simple async job API (see engine/dr-tulu/agent/scripts/launch_server.py).
    This client mirrors the interface used by other DeepResearchAgentBase implementations so it
    can be plugged into call_dra.py without additional changes.
    """

    def __init__(
        self,
        model: str = "dr-tulu",
        *,
        base_url: Optional[str] = None,
        poll_interval: int = 5,
        request_timeout: int = 1200,
        max_wait_seconds: int = 5400,
        dataset_name: Optional[str] = "long_form",
        config_overrides: Optional[Dict[str, Any]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("DR_TULU_API_URL")
            or "http://127.0.0.1:8321"
        )
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self.max_wait_seconds = max_wait_seconds
        self.default_dataset_name = dataset_name or os.getenv("DR_TULU_DATASET_NAME")
        self.config_overrides = config_overrides
        self.session = session or requests.Session()

        if not self.base_url.startswith("http"):
            raise ValueError("DR-Tulu base_url must be an HTTP endpoint.")

    def _normalize_messages(self, input_data: PromptType) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in input_data]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
    )
    def __call__(self, input_data: PromptType, **kwargs) -> str:
        """Submit a new job to the DR-Tulu server and return its request_id."""
        messages = self._normalize_messages(input_data)

        payload: Dict[str, Any] = {
            "messages": messages,
            "dataset_name": kwargs.get("dataset_name") or self.default_dataset_name,
        }
        if self.config_overrides:
            payload["config_overrides"] = self.config_overrides

        url = f"{self.base_url.rstrip('/')}/api/v1/jobs"
        response = self.session.post(url, json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        request_id = data.get("request_id")
        if not request_id:
            raise RuntimeError(f"DR-Tulu server did not return a request_id: {data}")
        return request_id

    def wait_for_completion(self, request_id: str):
        """Poll the server until the job finishes and return the formatted response."""
        start_time = time.time()
        while True:
            try:
                is_completed, result = self.poll(request_id)
                if is_completed:
                    return result
            except requests.RequestException:
                if time.time() - start_time > self.max_wait_seconds:
                    raise TimeoutError(f"Timed out waiting for job {request_id}")
                time.sleep(self.poll_interval)
                continue

    def poll(self, request_id: str) -> Tuple[bool, Optional[Any]]:
        """Poll the server for job status without blocking.
        
        Args:
            request_id (str): Request ID of the job to poll.
            
        Returns:
            Tuple[bool, Optional[Any]]: (is_completed, result) - is_completed is True if the job has completed,
                   result is the response object if completed, None otherwise
                   
        Raises:
            RuntimeError: If the job failed or has an error.
            requests.RequestException: If there was an error communicating with the server.
        """
        status, payload = self.get_response(request_id)

        if status == "completed":
            result = payload.get("result")
            if not result:
                raise RuntimeError(f"Job {request_id} completed without a result: {payload}")
            return True, self._postprocess_result(result)

        if status == "failed":
            raise RuntimeError(f"Job {request_id} failed: {payload.get('error')}")
            
        # For all other statuses, return False and None
        return False, None

    def get_response(self, request_id: str):
        """Fetch the latest job status from the server."""
        url = f"{self.base_url.rstrip('/')}/api/v1/jobs/{request_id}"
        response = self.session.get(url, timeout=self.request_timeout)
        if response.status_code == 404:
            raise RuntimeError(f"Job {request_id} not found on DR-Tulu server.")
        response.raise_for_status()
        data = response.json()
        return data.get("status"), data

    def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract snippet_map from trace and add it to the result object."""
        trace = (result.get("metadata") or {}).get("trace")
        snippet_map = self._extract_snippet_metadata(trace)
        result["snippet_map"] = snippet_map
        return result

    def _extract_snippet_metadata(self, trace: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        snippet_map: Dict[str, Dict[str, str]] = {}
        if not isinstance(trace, dict):
            return snippet_map

        def record(snippet_id: str, doc: Dict[str, Any]) -> None:
            if not snippet_id or snippet_id in snippet_map:
                return
            url = doc.get("url") or doc.get("URL") or ""
            title = doc.get("title") or doc.get("Title") or url
            snippet_map[snippet_id] = {"title": title or "", "url": url}

        tool_calls = trace.get("tool_calls") or []
        for call in tool_calls:
            call_id = call.get("call_id") or ""

            documents = call.get("documents") or []
            for idx, doc in enumerate(documents):
                if call_id:
                    record(f"{call_id}-{idx}", doc)

            raw_output = call.get("raw_output")
            if isinstance(raw_output, dict):
                for inner in raw_output.get("tool_outputs") or []:
                    inner_call_id = inner.get("call_id") or call_id
                    inner_documents = inner.get("documents") or []
                    for idx, doc in enumerate(inner_documents):
                        if inner_call_id:
                            record(f"{inner_call_id}-{idx}", doc)
                    if inner_documents or not inner_call_id:
                        continue
                    parsed_inner = parse_search_results(inner.get("output") or "")
                    for idx, doc in enumerate(parsed_inner):
                        record(f"{inner_call_id}-{idx}", doc)

            parsed = parse_search_results(call.get("output") or "")
            if call_id:
                for idx, doc in enumerate(parsed):
                    record(f"{call_id}-{idx}", doc)

            if not call_id:
                snippets = parse_xml_snippets(call.get("output") or "")
                for snippet_id, doc in snippets.items():
                    record(snippet_id, doc)

        return snippet_map


__all__ = ["DrTuluDeepResearchAgent", "parse_and_format_citations"]


