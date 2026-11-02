# Deep Research Agent Engines

Client code for the engines of `mrdre generate --engine`, and the reviser agent of `--use-reviser` (`reviser_agent/`, see [Reviser Agent](../../CLI-guide.md#reviser-agent)).

| `--engine` | Client | Setup |
| --- | --- | --- |
| `o3-deep-research`, `o4-mini-deep-research` | `oai.py` | `OPENAI_API_KEY` |
| `sonar-deep-research` | `perplexity.py` | `PERPLEXITY_API_KEY` |
| `dr-tulu` | `dr_tulu.py` | A DR Tulu server at `DR_TULU_API_URL` (default `http://127.0.0.1:8321`) |
| `odr-*` | `odr.py` | A [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) LangGraph server at `LANGGRAPH_API_URL` (default `http://localhost:2025`) |

## Adding an Engine

The simplest way is a Python function passed as `--engine module:function`; see [Bring Your Own Agent](../../CLI-guide.md#bring-your-own-agent).

An engine that submits jobs to a service and polls for their results can instead be added here as a subclass of `DeepResearchAgentBase` (in `__init__.py`) with these methods:

- `__call__(messages)`: submit the conversation of one question and return a request ID.
- `poll(request_id)`: return `(True, response)` once the request is done, where `response` is a JSON-serializable dict with the report under `"report"`; return `(False, None)` while it runs, and raise if it failed.
- `get_response(request_id)` and `wait_for_completion(request_id)`: the current status, and a blocking wait for the response.

Then add its model-name prefix to `_PROVIDER_REGISTRY` in `__init__.py`. `mrdre generate` creates the class with `model=<the --engine value>` only, so read other settings from environment variables.
