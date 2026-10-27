# Deep Research Agent Engines

This folder contains the client-side code for each supported DRA. `DRA` in `__init__.py` routes a model name to its engine through `_PROVIDER_REGISTRY`.

| Engine | Model names | File |
|---|---|---|
| OpenAI Deep Research | `o4-mini-deep-research`, `o3-deep-research` | `oai.py` |
| Perplexity Sonar | `sonar-*` | `perplexity.py` |
| Open Deep Research | `odr-*` | `odr.py` (to be released) |
| DR Tulu | `dr-tulu*` | `dr_tulu.py` (to be released) |
| Tongyi Deep Research | `tongyi*` | `tongyi.py` (to be released) |

## Adding a new DRA

Subclass `SimpleDeepResearchAgent`, implement `generate(messages)`, and register the class in `_PROVIDER_REGISTRY`. `engine/template.py` is a starting point, and [`docs/evaluate_your_dra.md`](../docs/evaluate_your_dra.md) walks through generation and evaluation end to end.

Every engine returns a dict with at least a `report` key. Cited sources must appear as URLs in the report text, because the factuality evaluation extracts them from there.
