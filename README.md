# ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation

<p align="center">
  <a href="https://arxiv.org/abs/2601.02535"><img src="https://img.shields.io/badge/arXiv-2601.02535-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/deeplearning-wisc/ModeX/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

Official code for the ACL 2026 Main Track paper:

> **ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation**  
> Hyeong Kyu Choi and Sharon Li  
> [arXiv:2601.02535](https://arxiv.org/abs/2601.02535)

---

## Overview

ModeX is an evaluator-free framework for selecting the best output from a set of N independently sampled LLM responses. Instead of relying on a reward model or external judge, ModeX builds a **semantic similarity graph** over the candidates and identifies the *modal* output — the centroid of the dominant cluster — through **recursive spectral graph partitioning**.

**ModeX-Lite** is an efficient variant that integrates the same pruning logic directly into the token-by-token decoding loop, eliminating the need to generate all N responses to completion before selection.

Both methods are entirely evaluator-free, requiring no auxiliary model or additional inference beyond the N forward passes used to generate the candidates.

---

## Repository Structure

```
ModeX/
├── modex/              # ModeX: post-hoc selection via spectral graph clustering
│   ├── main.py         # Entry point and core algorithm
│   ├── utils.py        # Batched generation engine
│   ├── evaluator.py    # Task-specific answer extraction and scoring
│   ├── prompts.py      # Prompt templates
│   ├── dashboard.py    # Logging and result visualization
│   ├── model/          # Model wrappers (Qwen, Llama, CodeLlama)
│   └── data/           # Dataset loaders (CNN/DM, HumanEval, MATH-500, …)
│
├── modex-lite/         # ModeX-Lite: online pruning during decoding
│   ├── main.py         # Entry point (adds --new_decode, --prune_frequency)
│   ├── utils.py        # Generation engine with ModeX-Lite hook
│   ├── model/
│   │   └── ma_decoder.py   # Online similarity-based batch pruning
│   └── ...             # (same structure as modex/)
│
├── scripts/
│   ├── run_modex.sh        # Example commands for ModeX
│   └── run_modex_lite.sh   # Example commands for ModeX-Lite
│
├── environment.yml     # Conda environment
└── README.md
```

---

## Installation

```bash
git clone https://github.com/deeplearning-wisc/ModeX.git
cd ModeX
conda env create -f environment.yml
conda activate modex
```

For gated models (e.g., Llama), log in to HuggingFace:
```bash
huggingface-cli login
```
or place your access token in a file named `token` inside the `modex/` (or `modex-lite/`) directory.

---

## Quick Start

### ModeX (post-hoc selection)

```bash
cd modex/

# Summarization — Qwen2.5-7B, N=8
python main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data cnn_daily \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance

# Math reasoning — Llama3.1-8B, N=8
python main.py \
    --model llama3.1-8b \
    --num_agents 8 \
    --data math500 \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance
```

### ModeX-Lite (online pruning)

```bash
cd modex-lite/

# Code generation — Qwen2.5-7B, N=8, prune every 300 tokens
python main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data humaneval \
    --data_size 164 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance \
    --new_decode \
    --prune_frequency 300
```

See `scripts/` for more examples.

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen2.5-7b` | Model name (see supported models below) |
| `--num_agents` | `4` | Number of parallel samples N |
| `--data` | `math500` | Dataset (see supported datasets below) |
| `--data_size` | `300` | Number of test samples to evaluate |
| `--tau` | `0.8` | Early-stopping threshold (higher = more aggressive pruning) |
| `--goodness_of_cut` | `conductance` | Cut quality metric: `conductance`, `cutratio`, or `ngc` |
| `--adjacency` | `text` | Similarity type: `text` (n-gram Jaccard), `semantics` (sentence-transformers MiniLM), or `both` |
| `--multi_persona` | off | Assign diverse system prompts to agents (from DyLAN) |
| `--bae` | off | Use base answer extractor for evaluation |
| `--model_dir` | `None` | Local path to model weights (default: HuggingFace Hub) |
| `--out_dir` | `out/` | Directory for logs and plots |

**ModeX-Lite only:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--new_decode` | off | Enable online pruning during generation |
| `--prune_frequency` | `100` | Token interval between pruning steps |

---

## Supported Models

| Short name | HuggingFace ID |
|------------|---------------|
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` |
| `qwen2.5-32b` | `Qwen/Qwen2.5-32B-Instruct` |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` |
| `llama3.1-8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `llama3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` |
| `llama2-7b-chat` | `meta-llama/Llama-2-7b-chat-hf` |
| `llama2-13b-chat` | `meta-llama/Llama-2-13b-chat-hf` |
| `llama2-70b-chat` | `meta-llama/Llama-2-70b-chat-hf` |
| `codellama` | `meta-llama/CodeLlama-7b-Instruct-hf` |

---

## Supported Datasets

| Category | Dataset key |
|----------|------------|
| Math reasoning | `math500`, `gsm8k`, `arithmetics` |
| Multiple choice | `gpqa` |
| Summarization | `cnn_daily` |
| Code generation | `humaneval` |


---

## Citation

```bibtex
@inproceedings{choi2026modex,
  title     = {ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation},
  author    = {Choi, Hyeong Kyu and Li, Sharon},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026},
}
```
