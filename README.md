# Mr-Dre: Evaluation Suite for Deep Research Report Writing and Multi-turn Revision

<p align="center">
    <a href="https://arxiv.org/PLACEHOLDER" target="_blank" rel="noopener noreferrer">
        <img alt="paper" src="https://img.shields.io/badge/paper-paper?logo=arxiv&logoColor=%23B31B1B&labelColor=white&color=%23B31B1B">
    </a>
</p>

---

Official repository for the paper *"Beyond Single-shot Writing: Deep Research Agents are Unreliable at Multi-turn Report Revision"*.

We introduce **Mr-Dre**, a new evaluation suite for Deep Research Agents (DRAs) on *multi-turn report writing and revision*. Specifically, Mr-Dre includes:

1. **A Unified Evaluation Protocol for Deep Research Reports:** We consolidated best practices from previous Deep Research evaluation works into a lean protocol spanning three dimensions: *Comprehensiveness*, *Factuality*, and *Presentation*.
2. **Feedback Simulation Pipeline for Multi-turn Report Revision:** To evaluate not only the single-shot writing capabilities of Deep Research Agents but also their multi-turn report revision ability, we introduce a reliable method to simulate content, format, and self-reflection feedback.

## Table of Contents

- [News](#news)
- [Setup](#setup)
- [Data](#data)
- [DRA Report Generation](#dra-report-generation)
- [DRA Report Evaluation](#dra-report-evaluation)
- [Release Progress](#release-progress)
- [Contact Us](#contact-us)
- [Citation](#citation)

## News

## Setup

We recommend using [uv](https://github.com/astral-sh/uv) to quickly set up and manage your environment:

```shell
# Clone the repository
git clone https://github.com/BaleChen/mt-dra.git
cd mt-dra

# Set up the environment
uv sync
```

Then, set your API keys as environment variables. We use the following 4 APIs throughout this repository:

```shell
export OPENAI_API_KEY="YOUR_KEY_HERE"
export PERPLEXITY_API_KEY="YOUR_KEY_HERE"
export JINA_API_KEY="YOUR_KEY_HERE"
export SERPER_API_KEY="YOUR_KEY_HERE"
```

> **Note:** Tongyi Deep Research, DR Tulu, and Open Deep Research require customized environment setup. We recommend isolating the server-side and client-side environments for best compatibility. See [`engine/README.md`](engine/README.md) for detailed instructions on setting up each DRA engine's server.

## Data


## DRA Report Generation


## DRA Report Evaluation


## Release Progress

- [x] Main evaluation protocol
- [ ] Setup quickstart
- [ ] Main engine and generation code
- [ ] Feedback simulation
- [ ] Datasets used in our paper
- [ ] Tongyi, DR Tulu, Open Deep Research server-side code
- [ ] "How to Add a New DRA Engine" guide
- [ ] "How to Add a New Dataset" guide

## Contact Us

If you have any questions, feel free to email me at `bale[dot]chen[at]nyu[dot]edu`. If you encounter any issues, you can also open an issue on this repository. Please include your setup details and a code snippet to reproduce the problem so we can help resolve it quickly!

## Citation

If you find our work useful, please cite us:

```bibtex
PLACEHOLDER
```