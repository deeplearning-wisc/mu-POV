
import argparse
import copy
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from model.model_utils import get_agents
from utils import engine
from data.data_utils import load_data
from prompts import create_initial_messages
from evaluator import (
    get_instruction_suffix,
    evaluate_arithmetics,
    evaluate_mcq,
    evaluate_basic,
    base_evaluate_arithmetics,
    base_evaluate_mcq,
    evaluate_gen,
    evaluate_code,
)
from dashboard import Tee, display_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


CLOSED_ENDED_TASKS = ['gpqa']
MATH_REASONING_TASKS = ['arithmetics', 'gsm8k', 'math500']
TEXT_GENERATION_TASKS = ['cnn_daily']
CODE_GENERATION_TASKS = ['humaneval']


def setup_seeds(seed=42):
    """Setup random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_evaluator(args, data_name, use_bae=False):
    """Get the appropriate evaluation function based on dataset."""
    if data_name in args.math_reasoning_tasks:
        return base_evaluate_arithmetics if use_bae else evaluate_arithmetics
    elif data_name in args.closed_ended_tasks:
        return base_evaluate_mcq if use_bae else evaluate_mcq
    elif data_name in args.text_generation_tasks:
        return evaluate_gen
    elif data_name in args.code_generation_tasks:
        return evaluate_code
    else:
        raise NotImplementedError(f"Dataset {data_name} not supported")


def get_responses(args, question, agent, num_agents, personas, suffix, evaluate_func):
    """Generate the initial round of responses from all agents."""
    print("Gathering initial opinions...")

    messages = create_initial_messages(question, num_agents, personas, suffix)
    prompts, responses = engine(messages, agent, num_agents)

    agent_names = [f"Agent{i+1}" for i in range(num_agents)]
    agent_responses = dict(zip(agent_names, responses))

    return prompts, agent_responses, responses



def goodness_of_cut(args, A, group_indices, n_group_indices):
    """Compute a graph cut quality metric to decide when to stop pruning."""
    if args.goodness_of_cut == 'cutratio':
        cut_S_Sbar = float(np.sum(A[np.ix_(group_indices, n_group_indices)]))
        return cut_S_Sbar / (np.sum(A) - A.shape[0])

    elif args.goodness_of_cut == 'ngc':
        cut_S_Sbar = float(np.sum(A[np.ix_(group_indices, n_group_indices)]))
        vol_S = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_Sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return (cut_S_Sbar / vol_S) + (cut_S_Sbar / vol_Sbar)

    elif args.goodness_of_cut == 'conductance':
        cut_S_Sbar = float(np.sum(A[np.ix_(group_indices, n_group_indices)]))
        vol_S = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_Sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return cut_S_Sbar / (min(vol_S, vol_Sbar) + 1e-10)


def vote(args, question, answer, agent_responses, agent, num_agents, personas, suffix, evaluate_func):
    """
    ModeX selection: iteratively apply spectral graph cuts to identify the modal cluster,
    then return the highest-degree node within that cluster.
    """
    agent_names = list(agent_responses.keys())

    A = compute_adjacency_matrix(args, agent_names, agent_responses)

    _A = A.copy()
    current_names = agent_names.copy()

    depth = 0
    while True:
        info = graph_cut(args, _A, current_names)
        depth += 1

        if len(info['groups']['group_1']) > len(info['groups']['group_2']):
            group = info['groups']['group_1']
            n_group = info['groups']['group_2']
        elif len(info['groups']['group_2']) > len(info['groups']['group_1']):
            group = info['groups']['group_2']
            n_group = info['groups']['group_1']
        else:
            # Tie: pick group with higher total degree
            degrees = np.sum(_A, axis=1)
            g1_names = info['groups']['group_1']
            g2_names = info['groups']['group_2']
            g1_idx = [current_names.index(n) for n in g1_names]
            g2_idx = [current_names.index(n) for n in g2_names]
            g1_deg_sum = float(np.sum(degrees[g1_idx]))
            g2_deg_sum = float(np.sum(degrees[g2_idx]))
            group = g1_names if g1_deg_sum >= g2_deg_sum else g2_names
            n_group = g2_names if g1_deg_sum >= g2_deg_sum else g1_names

        prev_n = len(current_names)
        group_indices = [current_names.index(n) for n in group]
        n_group_indices = [current_names.index(n) for n in n_group]

        phi_S = goodness_of_cut(args, _A, group_indices, n_group_indices)
        print(phi_S)

        if phi_S >= args.tau:
            best_idx = int(np.argmax(np.sum(_A, axis=1)))
            best_name = current_names[best_idx]
            return {best_name: agent_responses[best_name]}, depth - 1

        _A = _A[np.ix_(group_indices, group_indices)]
        current_names = [current_names[i] for i in group_indices]

        if len(current_names) == prev_n or len(current_names) <= 1:
            break

    current_names = random.sample(current_names, 1)
    return {current_names[0]: agent_responses[current_names[0]]}, depth


def compute_adjacency_matrix(args, agent_names, agent_responses):
    """Build pairwise similarity adjacency matrix between agent responses."""
    if args.adjacency == 'text':
        return compute_text_adjacency_matrix(agent_names, agent_responses)
    elif args.adjacency == 'semantics':
        return compute_semantics_adjacency_matrix(agent_names, agent_responses)
    elif args.adjacency == 'both':
        return 0.5 * (compute_text_adjacency_matrix(agent_names, agent_responses)
                      + compute_semantics_adjacency_matrix(agent_names, agent_responses))


def compute_semantics_adjacency_matrix(agent_names, agent_responses):
    """Cosine similarity using a SentenceTransformer encoder."""
    emb_encoder = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    texts = [str(agent_responses[name]) for name in agent_names]
    embeddings = emb_encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    return np.dot(embeddings_norm, embeddings_norm.T)


def compute_text_adjacency_matrix(agent_names, agent_responses):
    """Pairwise similarity via combined unigram + bigram + trigram Jaccard."""
    adjacency_matrix = np.zeros((len(agent_names), len(agent_names)))

    def jaccard_similarity(a, b):
        a_set, b_set = set(str(a).lower().split()), set(str(b).lower().split())
        if not a_set and not b_set:
            return 1.0
        union = a_set | b_set
        return len(a_set & b_set) / len(union) if union else 0.0

    def ngram_jaccard(tokens_a, tokens_b, n):
        if len(tokens_a) < n and len(tokens_b) < n:
            return 1.0
        if len(tokens_a) < n or len(tokens_b) < n:
            return 0.0
        a_ngrams = set(tuple(tokens_a[i:i+n]) for i in range(len(tokens_a) - n + 1))
        b_ngrams = set(tuple(tokens_b[i:i+n]) for i in range(len(tokens_b) - n + 1))
        union = a_ngrams | b_ngrams
        return len(a_ngrams & b_ngrams) / len(union) if union else 0.0

    def combined_similarity(a, b):
        tokens_a, tokens_b = str(a).lower().split(), str(b).lower().split()
        return (jaccard_similarity(a, b) + (ngram_jaccard(tokens_a, tokens_b, 2) + ngram_jaccard(tokens_a, tokens_b, 3)) / 2.0) / 2.0

    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            adjacency_matrix[i, j] = combined_similarity(
                agent_responses[agent_names[i]],
                agent_responses[agent_names[j]],
            )

    return adjacency_matrix


def graph_cut(args, A, agent_names):
    """
    Spectral bi-partition via the Fiedler vector of the graph Laplacian.
    Returns group assignments for each agent.
    """
    degrees = np.sum(A, axis=1)
    L = np.diag(degrees) - A

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    fiedler_vector = eigenvectors[:, 1] if len(eigenvalues) >= 2 else eigenvectors[:, 0]

    threshold = 0.0
    if np.all(fiedler_vector >= 0) or np.all(fiedler_vector <= 0):
        threshold = float(np.median(fiedler_vector))

    group1_idx = np.where(fiedler_vector > threshold)[0]
    group2_idx = np.where(fiedler_vector <= threshold)[0]

    partition = {agent_names[i]: 0 for i in group2_idx}
    for i in group1_idx:
        partition[agent_names[i]] = 1

    return {
        'A': A,
        'fiedler_vector': fiedler_vector,
        'partition': partition,
        'groups': {
            'group_1': [agent_names[i] for i in group1_idx],
            'group_2': [agent_names[i] for i in group2_idx],
        },
    }


def get_best_of_n_scores(agent_responses, y, evaluate_func):
    """Oracle Best-of-N: return the score of the best individual response."""
    bon_scores = [evaluate_func({name: resp}, y)[2] for name, resp in agent_responses.items()]
    return np.max(bon_scores, axis=0), np.argmax(bon_scores, axis=0)


def run_experiment(args, test_X, test_Y, agent, personas, exp_name):

    evaluate_func = get_evaluator(args, args.data, args.bae)
    suffix = get_instruction_suffix(args)

    all_results = []
    for i, (x, y) in enumerate(tqdm(zip(test_X, test_Y), total=len(test_X))):

        print(f'\n\nQuestion {i+1}: {x + suffix}\n\n')

        sample_results = {'question': x, 'correct_answer': y}

        # Generate initial responses and run ModeX selection
        prompts, agent_responses, responses = get_responses(
            args, x, agent, args.num_agents, personas, suffix, evaluate_func
        )
        voted_response, depth = vote(
            args, x, y, agent_responses, agent, args.num_agents, personas, suffix, evaluate_func
        )

        # Evaluate
        if args.data in CLOSED_ENDED_TASKS + MATH_REASONING_TASKS:
            _, _, mv_scores = evaluate_func(agent_responses, y)
            sample_results['mv_scores'] = mv_scores
        _, final_response, scores = evaluate_func(voted_response, y)

        bon_scores, bon_idx = get_best_of_n_scores(agent_responses, y, evaluate_func)
        if 'int' not in str(type(bon_idx)):
            bon_idx = Counter(bon_idx).most_common()[0][0]

        single_agent_scores = []
        for name, resp in agent_responses.items():
            _, _resp, single_agent_score = evaluate_func({name: resp}, y)
            single_agent_scores.append(single_agent_score)
            print(f"\n### {name} Response \n{resp}\n\n")
        print(f"\n\n### Voted Response \n{final_response}\n\n")
        print(f"Target:\n{y}\n\n")

        sample_results['agent_responses'] = agent_responses
        sample_results['voted_response'] = final_response
        sample_results['scores'] = scores
        sample_results['bon_scores'] = bon_scores
        sample_results['single_agent_scores'] = single_agent_scores
        sample_results['depth'] = depth
        all_results.append(sample_results)

        display_results(args, all_results, exp_name)

    print("\n\nExperiment completed")


def main(args):

    # Build experiment name from key hyperparameters
    exp_name = (
        f"{args.data}_{args.data_size}__{args.model}*{args.num_agents}"
        f"_adj={args.adjacency}_goc={args.goodness_of_cut}"
    )
    exp_name += f"_tau={args.tau}"

    setup_seeds(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    exp_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_file_path = os.path.join(exp_dir, "log.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(log_file, original_stdout)
    sys.stderr = Tee(log_file, original_stderr)

    try:
        print("Loading data...")
        test_X, test_Y = load_data(args, split='test')

        print("Loading agents...")
        agent, personas = get_agents(args)
        personas = None if not args.multi_persona else personas

        run_experiment(args, test_X, test_Y, agent, personas, exp_name)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation")

    # General
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--out_dir', type=str, default="out/", help="Output directory")

    # Data
    parser.add_argument('--data_dir', type=str, default="data/", help="Directory containing dataset files")
    parser.add_argument('--data', type=str, default='math500',
                        choices=['arithmetics', 'gsm8k', 'math500', 'gpqa',
                                 'cnn_daily', 'humaneval'],
                        help="Dataset to evaluate on")
    parser.add_argument('--data_size', type=int, default=300, help="Number of test samples")

    # Agent
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                        choices=['qwen2.5-1.5b', 'qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b',
                                 'llama3.1-8b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b',
                                 'codellama'],
                        help="Model to use")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Local cache directory containing model weights (None = use HuggingFace Hub)")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--num_agents', type=int, default=4, help="Number of parallel samples (N in Best-of-N)")
    parser.add_argument('--multi_persona', action='store_true', help="Use diverse system prompts (personas) per agent")

    # ModeX similarity graph
    parser.add_argument('--adjacency', type=str, default='text',
                        choices=['text', 'semantics', 'both'],
                        help="Method for computing pairwise response similarity")
    parser.add_argument('--goodness_of_cut', type=str, default='conductance',
                        choices=['cutratio', 'ngc', 'conductance'],
                        help="Graph cut quality metric for early stopping")
    parser.add_argument('--tau', type=float, default=0.8,
                        help="Early-stopping threshold tau (higher = more aggressive pruning)")

    parser.add_argument('--bae', action='store_true', help="Use base answer extractor for evaluation")

    args = parser.parse_args()

    # Load HuggingFace token if present (needed for gated models such as Llama)
    try:
        with open('token', 'r') as f:
            args.token = f.read().strip()
    except FileNotFoundError:
        args.token = None

    # Attach task-type lists
    args.closed_ended_tasks = CLOSED_ENDED_TASKS
    args.math_reasoning_tasks = MATH_REASONING_TASKS
    args.text_generation_tasks = TEXT_GENERATION_TASKS
    args.code_generation_tasks = CODE_GENERATION_TASKS

    main(args)
