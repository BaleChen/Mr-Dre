# The `mrdre` command

`mrdre` has three commands, one for each stage of a revision study (see the [Quick Start](README.md#quick-start) for a full run):

- [`mrdre evaluate`](#mrdre-evaluate) scores reports and, given the previous turn, the revision.
- [`mrdre feedback`](#mrdre-feedback) simulates one feedback message per report.
- [`mrdre generate`](#mrdre-generate) calls an agent to write reports or revise the previous turn's reports.

All three commands share these options:

- `--dataset`: a benchmark name (`researchrubrics`, `rigorousbench`, `researcherbench`) or the path to a JSONL file in the [data format](README.md#data).
- `--limit N`: a fixed random subset of N questions, the same in every command.
- `--out`: the output file or directory. A rerun skips the questions already in `--out`; `--overwrite` recomputes them.

## `mrdre evaluate`

Scores each report along the three dimensions of our protocol. `--mode` selects one of them (default: all three).

| `--mode` | Dimension | Metric |
| --- | --- | --- |
| `checklist` | Comprehensiveness | Coverage of the question's checklist |
| `citation` | Factuality | Citation faithfulness and claim groundedness, judged against the cited web pages |
| `rubric` | Presentation | A fixed set of presentation rubrics |

```bash
mrdre evaluate \
    --dataset researcherbench \
    --reports my_reports.jsonl \
    --out results/my_agent
```

### Reports from Your Own Agent

Write one record per question, with numbered citations and a source list that gives each URL:

```json
{"id": 1, "report": "... text with citations [1] ...\n\n**Sources:**\n[1] Title (https://...)"}
```

### Outputs

`--out` holds the per-report scores (`checklist.jsonl`, `citation.jsonl`, `rubric.jsonl`), the dataset-level scores (`summary.json`, also printed), and the scored reports (`scored_reports.jsonl`).

### Models

`--judge-model` sets every judge (default: `gpt-4.1-mini`). Web page summaries use `gpt-5.6-luna`. Our paper used `gpt-4.1-mini` for the judges and `gpt-4.1-nano` (now deprecated) for the summaries. More capable and cost-efficient models such as `gpt-5.6-luna` can replace both roles, but the original paper's results were obtained with the GPT-4.1 series models.

### Revision Metrics

Pass the previous turn's evaluation directory with `--previous-eval` and the feedback with `--feedback` (the previous turn will not be scored again). `summary.json` then also holds:

- **Incorporation Rate:** for content feedback, the fraction of targeted checklist items that the revised report satisfies; for format feedback, the score of an LLM judge that compares the two reports. Self-reflection has no incorporation rate.
- **Break Rate:** among the checklist items that the feedback did not target and the previous report at least partially satisfied, the fraction whose score drops.
- **Score deltas:** the change in checklist coverage, citation faithfulness, claim groundedness, and presentation.

## `mrdre feedback`

Writes one feedback message per report:

| `--feedback-type` | Feedback |
| --- | --- |
| `content` | Asks to address checklist items that the report misses; needs the report's checklist scores (`--eval`, an `mrdre evaluate` output directory) |
| `format` | Asks for one change in writing, structure, or presentation |
| `reflection` | Asks the agent to reflect on its report and refine it |

```bash
mrdre feedback \
    --dataset researcherbench \
    --reports runs/turn1.jsonl \
    --feedback-type format \
    --out runs/feedback_format_turn2.jsonl
```

`--num-items` sets the number of missed checklist items in each content feedback (default: 1). `--refine` rewrites content or format feedback into a plan of localized edits. `--seed` sets the random seed (default: 42).

## `mrdre generate`

Generates one report per question. Without `--previous`, the engine receives the question and writes the first version of the report. With `--previous` and `--feedback`, it receives the conversation so far, ending with the new feedback, and revises its report.

| `--engine` | Engine |
| --- | --- |
| `o3-deep-research`, `o4-mini-deep-research` | OpenAI Deep Research |
| `sonar-deep-research` | Perplexity Sonar Deep Research |
| `dr-tulu` | DR Tulu, through your DR Tulu server |
| `odr-*` | LangChain Open Deep Research, through your LangGraph server |

See [mrdre/engine/README.md](mrdre/engine/README.md) for the setup of each engine and for adding an engine to `mrdre/engine/`.

### Bring Your Own Agent

Pass a Python function as `--engine module:function` or `--engine path/to/file.py:function`. It receives the message list of one question and returns the report as a string, or as a dict with a `report` key.

```python
# my_agent.py
def run(messages):
    # Turn 1: [{"role": "user", "content": question}]
    # Later turns: each earlier report (assistant) followed by its feedback (user)
    return my_research_system.run(messages)
```

```bash
mrdre generate \
    --dataset researcherbench \
    --engine my_agent:run \
    --out runs/turn1.jsonl
```

### Reviser Agent

`--use-reviser` revises the latest reports with our reviser agent, which edits each report according to the feedback, with web search. The default calls `Qwen/Qwen3-30B-A3B-Instruct-2507` on a vLLM server at `http://localhost:8005/v1` (set in `mrdre/engine/reviser_agent/main.py`). We started the server on two GPUs with:

```bash
uv sync --extra serve
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8005 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9
```
