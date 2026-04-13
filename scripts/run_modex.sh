#!/bin/bash
# Example commands for running ModeX (post-hoc selection).
#
# ModeX generates N independent responses from the same model and selects the
# best one via recursive spectral graph clustering on the similarity graph.
#
# Usage:
#   cd modex/
#   bash ../scripts/run_modex.sh
#
# Key arguments:
#   --num_agents N        : number of parallel samples (N in Best-of-N)
#   --tau 0.8             : early-stopping threshold (recommended: 0.8)
#   --goodness_of_cut     : cut metric (conductance recommended)
#   --adjacency text      : similarity type (text / semantics / both)
#
# Place your HuggingFace access token in a file named "token" inside modex/
# if you are using gated models such as Llama.

set -e

# ---------- Summarization ----------
CUDA_VISIBLE_DEVICES=0 python modex/main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data cnn_daily \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance

# ---------- Code Generation ----------
CUDA_VISIBLE_DEVICES=0 python modex/main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data humaneval \
    --data_size 164 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance

# ---------- Math Reasoning ----------
CUDA_VISIBLE_DEVICES=0 python modex/main.py \
    --model qwen2.5-7b \
    --num_agents 8 \
    --data math500 \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance

# ---------- Llama 3.1 8B variants ----------
CUDA_VISIBLE_DEVICES=0 python modex/main.py \
    --model llama3.1-8b \
    --num_agents 8 \
    --data cnn_daily \
    --data_size 300 \
    --tau 0.8 \
    --adjacency text \
    --goodness_of_cut conductance
