#!/bin/bash
# Example commands for running ModeX-Lite (online pruning during generation).
#
# ModeX-Lite integrates similarity-based pruning directly into the token-by-token
# decoding loop. Pass --new_decode to enable online pruning and --prune_frequency
# to control how often (in tokens) pruning is applied.
#
# Usage:
#   bash scripts/run_modex_lite.sh

set -e

# ---------- Summarization ----------
CUDA_VISIBLE_DEVICES=0 python modex-lite/main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data cnn_daily \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance \
    --new_decode \
    --prune_frequency 300

# ---------- Math Reasoning ----------
CUDA_VISIBLE_DEVICES=0 python modex-lite/main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data math500 \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance \
    --new_decode \
    --prune_frequency 300

# ---------- Code Generation ----------
CUDA_VISIBLE_DEVICES=0 python modex-lite/main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data humaneval \
    --data_size 164 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance \
    --new_decode \
    --prune_frequency 300
