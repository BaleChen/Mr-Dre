# Evaluating Your Own Deep Research Agent

This guide explains how to evaluate a new Deep Research Agent (DRA) with Mr Dre, both on single-shot report writing and on multi-turn report revision. There are two ways to plug in an agent, and both lead to the same evaluation command and the same metrics:

| | **Option A: Bring your own reports** | **Option B: Wrap your agent as an engine** |
|---|---|---|
| What you write | A script that runs your agent and saves JSONL files | One Python class with a `generate(messages)` method |
| Where your agent runs | Anywhere (your own codebase, cluster, or API) | Inside this repository, called by `call_dra.py` |
| Multi-turn revision | `mrdre.py feedback` exports the next-turn prompts for you | Fully automated by `scripts/generate_report.sh` |
| Best when | Your agent has its own infrastructure, or you already have reports | Your agent is reachable from Python (an API or a local loop) |

All evaluation goes through a single command, `python mrdre.py evaluate`, which runs the three evaluation dimensions of the paper on every turn and computes the revision metrics between consecutive turns.

## Prerequisites

Follow the [Setup](../README.md#setup) section of the main README. Evaluation requires only two API keys in `.env`:

- `OPENAI_API_KEY`: the LLM judges (default `gpt-4.1-mini`) and the webpage summarizer (default `gpt-4.1-nano`).
- `JINA_API_KEY`: crawling the URLs that reports cite, for the factuality evaluation.

The judges are called through the OpenAI SDK, so any OpenAI-compatible endpoint can be used by setting `OPENAI_BASE_URL` and passing `--judge_model`. Note, however, that the numbers in the paper were obtained with the default judges, so results with other judges are not directly comparable.

## The report format

Each turn is one JSONL file with one object per question:

```json
{"id": 1, "report": "# Title\n\nBody text with citations [1] ...\n\n## References\n[1] https://example.com/source"}
```

- `id` must match an `id` in the dataset file (`data/ResearcherBench.jsonl`, `data/RigorousBench.jsonl`, or `data/ResearchRubrics.jsonl`). String and integer forms of the same ID are both accepted.
- `report` is the full report text. Cited sources must appear in the text as URLs, either inline (`[source](https://...)`) or in a references section that numbered citations (`[1]`) point to. The factuality evaluation extracts claims and their URLs from the report text, so citations that exist only as metadata are not evaluated.
- Questions without a report are simply not evaluated. To compare with the paper on RigorousBench, use the same 100-question subset that `call_dra.py --num_questions 100` samples (seed 42).
- Any additional fields (for example `usage` or `citations`) are kept with the report and ignored by the evaluation.

Files written by `call_dra.py` (with a `full_response.report` field) are also accepted as they are.

## Option A: Bring your own reports

### Single-shot evaluation

Run your agent on the dataset questions, save its reports in the format above, and evaluate them:

```bash
python mrdre.py evaluate \
    --dataset ResearcherBench \
    --reports my_dra_turn1.jsonl \
    --output_dir results/my-dra
```

This evaluates Comprehensiveness (checklist coverage), Factuality (citation faithfulness and claim groundedness), and Presentation, prints a table, and writes the results to `results/my-dra/`. The command is resumable: if it is interrupted, or if some reports fail (for example because of an API error), running the same command again evaluates only the missing reports.

### Multi-turn revision

A revision turn sends the agent its previous report together with simulated user feedback. `mrdre.py feedback` produces that feedback, and the full conversation for each question, from the reports of the current turn:

```bash
# 1. Simulate the feedback for turn 2 from the turn-1 reports in results/my-dra
python mrdre.py feedback \
    --dataset ResearcherBench \
    --output_dir results/my-dra \
    --turn 1 \
    --type content_feedback
```

This writes `results/my-dra/feedback_turn2.jsonl`, where each line holds the feedback and the conversation to send to your agent:

```json
{"id": 1, "turn": 2, "type": "content_feedback",
 "feedback": "Could you also cover ...?",
 "targeted_item": ["..."], "targeted_id": [18],
 "messages": [{"role": "user", "content": "<question>"},
              {"role": "assistant", "content": "<turn-1 report>"},
              {"role": "user", "content": "Could you also cover ...?"}]}
```

```bash
# 2. Run your agent on each record's "messages" and save {"id", "report"} per line
#    to my_dra_turn2.jsonl (this step is yours).

# 3. Evaluate both turns and the revision between them
python mrdre.py evaluate \
    --dataset ResearcherBench \
    --reports my_dra_turn1.jsonl my_dra_turn2.jsonl \
    --output_dir results/my-dra
```

The turn-2 reports are matched with their feedback automatically through `feedback_turn2.jsonl`, so they need no extra fields. Further turns follow the same pattern (`--turn 2` produces `feedback_turn3.jsonl`, and so on).

The three feedback types correspond to the settings in the paper:

| `--type` | Feedback | Incorporation rate |
|---|---|---|
| `content_feedback` | Asks for content that the report misses, targeting checklist items it fails (`--num_items` items per question, default 1) | Share of targeted checklist items satisfied after revision |
| `format_feedback` | Asks for a change in structure, style, or presentation that preserves the content | Pairwise LLM judgment of whether the feedback was incorporated |
| `reflection` | A fixed prompt asking the agent to reflect on and refine its report | Not defined |

Content feedback requires the checklist evaluation of the current turn, which `mrdre.py feedback` runs first if it has not been run yet. A question receives no content feedback when its report fails none of the checklist items.

## Option B: Wrap your agent as an engine

If your agent can be called from Python, implementing one method lets `call_dra.py` handle the questions, the multi-turn conversations, the feedback simulation, batching, and resumption.

1. **Implement the agent.** Copy `engine/template.py` to, for example, `engine/my_dra.py` and implement `generate`:

    ```python
    from . import SimpleDeepResearchAgent

    class MyDeepResearchAgent(SimpleDeepResearchAgent):
        def generate(self, messages):
            # messages: [{"role": "user", "content": question},
            #            {"role": "assistant", "content": previous_report},
            #            {"role": "user", "content": feedback}, ...]
            report = run_my_agent(messages)
            return {"report": report}  # or simply return the report string
    ```

    `SimpleDeepResearchAgent` runs `generate` in a thread pool (`max_workers=10` by default), so a blocking implementation is sufficient. An exception raised in `generate` marks that question as failed, and it is retried the next time the same command runs. Agents that already expose an asynchronous submit-and-poll API can instead subclass `DeepResearchAgentBase` directly, as `engine/oai.py` and `engine/perplexity.py` do.

2. **Register it.** Add a line to `_PROVIDER_REGISTRY` in `engine/__init__.py`. Any model name that starts with the prefix is routed to your class:

    ```python
    ("my-dra", "my_dra", "MyDeepResearchAgent", True),
    ```

3. **Generate reports.** The arguments are `<dataset> <model> <turn> <type>`:

    ```bash
    ./scripts/generate_report.sh ResearcherBench my-dra 1 init
    ```

    This writes `output/ResearcherBench/init/my-dra/gen_turn1.jsonl`.

4. **Evaluate.**

    ```bash
    python mrdre.py evaluate --dataset ResearcherBench --run_dir output/ResearcherBench/init/my-dra
    ```

5. **Revise.** Content feedback for turn 2 is generated from the checklist evaluation of turn 1, which step 4 has already produced. Turn 1 is shared by all feedback settings, so generating turn 2 reads it from the `init` folder:

    ```bash
    ./scripts/generate_report.sh ResearcherBench my-dra 2 content_feedback
    ```

    The revised reports are written to `output/ResearcherBench/content_feedback_k=1/my-dra/gen_turn2.jsonl`. To evaluate the revision, point `--reports` at both turns:

    ```bash
    python mrdre.py evaluate --dataset ResearcherBench \
        --reports output/ResearcherBench/init/my-dra/gen_turn1.jsonl \
                  output/ResearcherBench/content_feedback_k=1/my-dra/gen_turn2.jsonl \
        --output_dir results/my-dra-content
    ```

## Outputs and metrics

`mrdre.py evaluate` writes the following files to the output directory:

| File | Content |
|---|---|
| `summary.json`, `summary_turns.csv` | One row per turn with the scores below, the number of reports, and the number of failed evaluations |
| `summary_revisions.csv` | One row per consecutive pair of turns with the revision metrics below |
| `gen_turn{i}.jsonl` | The reports of turn *i* in Mr Dre's native format |
| `{checklist,citation,rubric}_turn{i}.jsonl` | Per-question results, including the judge's justification for every checklist item and every claim |
| `incorporation_turn{i}.jsonl` | Per-question incorporation judgments (format feedback only) |
| `feedback_turn{i}.jsonl` | Simulated feedback and conversations for turn *i* (written by `mrdre.py feedback`) |

**Per-turn scores** (all in [0, 1]):

- *Checklist Coverage* (Comprehensiveness): weighted score of the question-specific checklist, normalized by the total positive weight (items with negative weight penalize content that should not appear).
- *Citation Faithfulness* (Factuality): share of cited claims that their sources support.
- *Claim Groundedness* (Factuality): share of all factual claims that are supported by a cited source.
- *Presentation*: average score over a fixed set of presentation rubric items.

**Revision metrics** (turn *t-1* to turn *t*):

- *Incorporation Rate*: how well the revision addresses the feedback (see the feedback table above).
- *Break Rate*: share of checklist items that were satisfied before the revision, were not targeted by the feedback, and are no longer satisfied afterwards.
- *Checklist Coverage Delta*, *Citation Faithfulness Delta*, *Claim Groundedness Delta*, and *Presentation Delta*: change in each per-turn score.

The feedback type of each revision is inferred from the stored feedback; pass `--feedback_type` to set it explicitly.

## Cost and runtime

Evaluation calls the judge once per checklist item, several times per report for claim extraction, once per extracted claim, and once per presentation rubric item, and it crawls every cited URL. LLM responses and crawled pages are cached on disk (in `.cache/`, or the directory set by `MCP_CACHE_DIR`), so re-running an evaluation does not repeat finished calls. To check the setup before a full run, evaluate a few reports first, for example with `--eval_mode checklist` on a file that contains only a handful of questions.
