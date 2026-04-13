"""
Multi-agent decoding utilities for filtering batches based on graph-based selection.
"""

import torch
import numpy as np


def filter_batch_by_multi_agent_decoding(
    args,
    gen_responses,
    batch_size,
    next_tokens,
    finished,
    all_generated_tokens,
    all_embeddings,
    past_key_values,
    input_ids,
    attention_mask,
    use_cache,
    huggingface_model,
    verbose=False
):
    """
    Filter the active batch to the modal cluster identified by one step of
    spectral graph partitioning (ModeX-Lite online pruning).

    Returns a dict with keys: new_batch_size, next_tokens, finished,
    all_generated_tokens, filtered_gen_responses, filtered_embeddings,
    past_key_values, input_ids, attention_mask.
    """

    original_input_ids = input_ids.clone()
    original_attention_mask = attention_mask.clone()

    A = compute_text_adjacency_matrix(gen_responses)
    new = multi_agent_decoding(args, A)

    # Default result — no filtering
    result = {
        'new_batch_size': batch_size,
        'next_tokens': next_tokens,
        'finished': finished,
        'all_generated_tokens': all_generated_tokens,
        'all_embeddings': all_embeddings,
        'filtered_gen_responses': gen_responses,
        'filtered_embeddings': torch.stack([all_embeddings[i][-1] for i in range(batch_size)], dim=0),
        'past_key_values': past_key_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'filtered': False
    }

    if len(new) < batch_size and len(new) > 0:
        new = [i for i in new if 0 <= i < batch_size]
        if len(new) == 0:
            new = list(range(batch_size))

        new_indices = torch.tensor(new, dtype=torch.long, device=input_ids.device)

        filtered_input_ids = input_ids[new_indices]
        filtered_attention_mask = attention_mask[new_indices]
        filtered_next_tokens = next_tokens[new_indices]
        filtered_finished = finished[new_indices]
        filtered_all_generated_tokens = [all_generated_tokens[i] for i in new]
        filtered_embeddings = torch.stack([all_embeddings[i][-1] for i in new], dim=0)
        filtered_gen_responses = [gen_responses[i] for i in new]

        filtered_past_key_values = past_key_values
        if past_key_values is not None:
            if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                cache_type = type(past_key_values)
                try:
                    if hasattr(past_key_values, 'to_legacy_cache'):
                        legacy_cache = past_key_values.to_legacy_cache()
                        filtered_legacy = tuple(
                            (legacy_cache[i][0][new_indices.to(legacy_cache[i][0].device)],
                             legacy_cache[i][1][new_indices.to(legacy_cache[i][1].device)])
                            if legacy_cache[i] is not None else None
                            for i in range(len(legacy_cache))
                        )
                        if hasattr(cache_type, 'from_legacy_cache'):
                            filtered_past_key_values = cache_type.from_legacy_cache(filtered_legacy)
                        else:
                            filtered_past_key_values = filtered_legacy
                    else:
                        # Rebuild cache via forward pass
                        filtered_input_ids_list = []
                        for idx in new:
                            seq_tokens = original_input_ids[idx:idx+1].clone()
                            if len(all_generated_tokens[idx]) > 0:
                                gen_tokens = torch.tensor(
                                    [all_generated_tokens[idx]],
                                    dtype=seq_tokens.dtype,
                                    device=seq_tokens.device
                                )
                                seq_tokens = torch.cat([seq_tokens, gen_tokens], dim=1)
                            filtered_input_ids_list.append(seq_tokens.squeeze(0))

                        if len(filtered_input_ids_list) > 0:
                            max_len = max(seq.shape[0] for seq in filtered_input_ids_list)
                            padded_list = []
                            for seq in filtered_input_ids_list:
                                if seq.shape[0] < max_len:
                                    pad_token = seq[-1].expand(max_len - seq.shape[0])
                                    padded_list.append(torch.cat([seq, pad_token], dim=0))
                                else:
                                    padded_list.append(seq)
                            filtered_input_ids_full = torch.stack(padded_list, dim=0)
                        else:
                            filtered_input_ids_full = torch.empty(
                                (0,), dtype=original_input_ids.dtype, device=original_input_ids.device)

                        filtered_attention_mask_full = torch.ones(
                            filtered_input_ids_full.shape,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                        rebuild_output = huggingface_model(
                            input_ids=filtered_input_ids_full,
                            attention_mask=filtered_attention_mask_full,
                            use_cache=use_cache,
                            return_dict=True
                        )
                        filtered_past_key_values = rebuild_output.past_key_values
                        filtered_input_ids = filtered_input_ids_full
                        filtered_attention_mask = filtered_attention_mask_full

                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not filter cache, resetting: {e}")
                    filtered_past_key_values = None
            else:
                # Tuple format — filter directly
                filtered_past_key_values_list = []
                for layer_past in past_key_values:
                    if layer_past is not None:
                        if isinstance(layer_past, tuple) and len(layer_past) == 2:
                            key, value = layer_past
                            filtered_key = key[new_indices] if key is not None and key.dim() > 0 else key
                            filtered_value = value[new_indices] if value is not None and value.dim() > 0 else value
                            filtered_past_key_values_list.append((filtered_key, filtered_value))
                        else:
                            filtered_past_key_values_list.append(
                                layer_past[new_indices] if hasattr(layer_past, 'dim') and layer_past.dim() > 0 else layer_past
                            )
                    else:
                        filtered_past_key_values_list.append(None)
                filtered_past_key_values = tuple(filtered_past_key_values_list)

        result.update({
            'new_batch_size': len(new),
            'next_tokens': filtered_next_tokens,
            'finished': filtered_finished,
            'all_generated_tokens': filtered_all_generated_tokens,
            'filtered_gen_responses': filtered_gen_responses,
            'filtered_embeddings': filtered_embeddings,
            'past_key_values': filtered_past_key_values,
            'input_ids': filtered_input_ids,
            'attention_mask': filtered_attention_mask,
        })

    return result


def compute_text_adjacency_matrix(agent_responses):
    adjacency_matrix = np.zeros((len(agent_responses), len(agent_responses)))

    def jaccard_similarity(a, b):
        a_set = set(str(a).lower().split())
        b_set = set(str(b).lower().split())
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
        tokens_a = str(a).lower().split()
        tokens_b = str(b).lower().split()
        ngram_sim = (ngram_jaccard(tokens_a, tokens_b, 2) + ngram_jaccard(tokens_a, tokens_b, 3)) / 2.0
        return (jaccard_similarity(a, b) + ngram_sim) / 2.0

    for i in range(len(agent_responses)):
        for j in range(len(agent_responses)):
            adjacency_matrix[i, j] = combined_similarity(agent_responses[i], agent_responses[j])

    return adjacency_matrix


def multi_agent_decoding(args, A):
    _A = A.copy()
    agent_names = [str(i) for i in range(A.shape[0])]
    current_names = agent_names.copy()

    info = graph_cut(_A, current_names)

    if len(info['groups']['group_1']) > len(info['groups']['group_2']):
        group, n_group = info['groups']['group_1'], info['groups']['group_2']
    elif len(info['groups']['group_2']) > len(info['groups']['group_1']):
        group, n_group = info['groups']['group_2'], info['groups']['group_1']
    else:
        degrees = np.sum(_A, axis=1)
        g1_idx = [current_names.index(n) for n in info['groups']['group_1']]
        g2_idx = [current_names.index(n) for n in info['groups']['group_2']]
        if float(np.sum(degrees[g1_idx])) >= float(np.sum(degrees[g2_idx])):
            group, n_group = info['groups']['group_1'], info['groups']['group_2']
        else:
            group, n_group = info['groups']['group_2'], info['groups']['group_1']

    group_indices = [current_names.index(n) for n in group]
    n_group_indices = [current_names.index(n) for n in n_group]

    phi_S = goodness_of_cut(args, _A, group_indices, n_group_indices)

    if phi_S >= args.tau:
        return [int(n) for n in current_names]
    else:
        return [int(current_names[i]) for i in group_indices]


def goodness_of_cut(args, A, group_indices, n_group_indices):
    cut_S_Sbar = float(np.sum(A[np.ix_(group_indices, n_group_indices)]))
    if args.goodness_of_cut == 'conductance':
        vol_S = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_Sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return cut_S_Sbar / min(vol_S, vol_Sbar)
    elif args.goodness_of_cut == 'ngc':
        vol_S = float(np.sum(A[group_indices, :]) - len(group_indices))
        vol_Sbar = float(np.sum(A[n_group_indices, :]) - len(n_group_indices))
        return (cut_S_Sbar / vol_S) + (cut_S_Sbar / vol_Sbar)
    elif args.goodness_of_cut == 'cutratio':
        return cut_S_Sbar / (np.sum(A) - A.shape[0])


def graph_cut(A, agent_names):
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
        }
    }
