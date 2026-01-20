"""
Tool definitions for the Reviser Agent.
Provides web search capabilities using Serper API.
"""

import json
import os
from typing import Dict, Any

import requests


# Tool JSON schema for vLLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a given query. Use this to find facts, verify claims, or gather additional information to improve the research report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using Serper API and return formatted results.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)

    Returns:
        Formatted string with search results
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set."

    num_results = min(num_results, 10)  # Cap at 10

    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": num_results,
        "gl": "us",
        "hl": "en",
    })
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        if response.status_code != 200:
            return f"Error: Search API returned status {response.status_code}"

        data = response.json()
        results = []

        # Format organic results
        for i, item in enumerate(data.get("organic", []), 1):
            title = item.get("title", "No title")
            link = item.get("link", "")
            snippet = item.get("snippet", "No snippet")
            results.append(f"{i}. **{title}**\n   URL: {link}\n   {snippet}")

        if not results:
            return "No search results found."

        return "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to perform search - {str(e)}"


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments.

    Args:
        name: Tool name
        arguments: Tool arguments as a dictionary

    Returns:
        Tool execution result as a string
    """
    if name == "web_search":
        return web_search(
            query=arguments.get("query", ""),
            num_results=arguments.get("num_results", 5),
        )
    else:
        return f"Error: Unknown tool '{name}'"

