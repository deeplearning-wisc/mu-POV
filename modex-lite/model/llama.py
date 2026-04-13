
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.ma_decoder import filter_batch_by_multi_agent_decoding, compute_text_adjacency_matrix


def load_model(args, model_name_or_path, llama_version=3):
    if llama_version == 3:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, device_map='auto',
            token=args.token, cache_dir=args.model_dir)
    else:
        # Llama 2 requires bfloat16 + flash attention to avoid NaN activations
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, device_map='auto',
            token=args.token, cache_dir=args.model_dir, attn_implementation='flash_attention_2')
    return model


class LlamaWrapper:
    def __init__(self, args, model_dir, lora_adapter_path=None, memory_for_model_activations_in_gb=2, llama_version=3):
        self.args = args
        self.name = model_dir
        self.huggingface_model = load_model(args, model_dir, llama_version)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def generate(self, input_ids, attention_mask, pad_token_id, max_new_tokens=64,
                 return_logits=True, verbose=False, do_sample=False, temperature=None,
                 top_p=None, top_k=None, repetition_penalty=None, **custom_decoding_kwargs):
        batch_size = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        all_generated_tokens = [[] for _ in range(batch_size)]
        all_scores = [] if return_logits else None
        all_embeddings = [[] for _ in range(batch_size)]
        past_key_values = None
        use_cache = True
        prev_tokens_tensor = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_key_values is None:
                    model_output = self.huggingface_model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=use_cache, output_hidden_states=True, return_dict=True)
                    past_key_values = model_output.past_key_values
                    next_token_logits = model_output.logits[:, -1, :]
                    next_token_embeddings = (model_output.hidden_states[-1][:, -1, :]
                                             if model_output.hidden_states is not None else None)
                else:
                    model_output = self.huggingface_model(
                        input_ids=prev_tokens_tensor.unsqueeze(1), attention_mask=None,
                        past_key_values=past_key_values, use_cache=use_cache,
                        output_hidden_states=True, return_dict=True)
                    past_key_values = model_output.past_key_values
                    next_token_logits = model_output.logits[:, -1, :]
                    next_token_embeddings = (model_output.hidden_states[-1][:, -1, :]
                                             if model_output.hidden_states is not None else None)

                if return_logits and all_scores is not None:
                    all_scores.append(next_token_logits.cpu())

                if next_token_embeddings is not None:
                    for i in range(batch_size):
                        if not finished[i]:
                            try:
                                all_embeddings[i].append(next_token_embeddings[i].cpu())
                            except Exception:
                                pass

                if repetition_penalty is not None and repetition_penalty != 1.0:
                    for i in range(batch_size):
                        if not finished[i] and len(all_generated_tokens[i]) > 0:
                            for token_id in set(all_generated_tokens[i]):
                                if next_token_logits[i, token_id] < 0:
                                    next_token_logits[i, token_id] *= repetition_penalty
                                else:
                                    next_token_logits[i, token_id] /= repetition_penalty

                if temperature is not None and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k is not None and top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    top_k_mask = torch.full_like(next_token_logits, float('-inf'))
                    top_k_mask.scatter_(-1, top_k_indices, top_k_values)
                    next_token_logits = top_k_mask

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                next_tokens = torch.where(finished, pad_token_id, next_tokens)

                for i in range(batch_size):
                    if not finished[i]:
                        all_generated_tokens[i].append(next_tokens[i].item())

                gen_responses, gen_embeddings = self.decode_generated_tokens(
                    all_generated_tokens, all_embeddings, batch_size)

                if step % self.args.prune_frequency == self.args.prune_frequency - 1:
                    filter_result = filter_batch_by_multi_agent_decoding(
                        args=self.args, gen_responses=gen_responses, batch_size=batch_size,
                        next_tokens=next_tokens, finished=finished,
                        all_generated_tokens=all_generated_tokens, all_embeddings=all_embeddings,
                        past_key_values=past_key_values, input_ids=input_ids,
                        attention_mask=attention_mask, use_cache=use_cache,
                        huggingface_model=self.huggingface_model, verbose=verbose)
                    batch_size = filter_result['new_batch_size']
                    next_tokens = filter_result['next_tokens']
                    finished = filter_result['finished']
                    all_generated_tokens = filter_result['all_generated_tokens']
                    gen_responses = filter_result['filtered_gen_responses']
                    gen_embeddings = filter_result['filtered_embeddings']
                    past_key_values = filter_result['past_key_values']
                    input_ids = filter_result['input_ids']
                    attention_mask = filter_result['attention_mask']

                finished = finished | (next_tokens == eos_token_id)

                if finished.all():
                    eos_token = self.tokenizer.convert_ids_to_tokens(eos_token_id)
                    gen_responses = [x.replace(eos_token, "") for x in gen_responses]
                    break

                prev_tokens_tensor = next_tokens

        A = compute_text_adjacency_matrix(gen_responses)
        final_idx = A.mean(0).argmax()
        return [gen_responses[final_idx]], gen_embeddings[final_idx]

    def decode_generated_tokens(self, all_generated_tokens, all_embeddings, batch_size):
        gen_responses = []
        gen_embeddings = []
        for i in range(batch_size):
            if len(all_generated_tokens[i]) > 0:
                gen_text = self.tokenizer.decode(all_generated_tokens[i], skip_special_tokens=False)
                gen_responses.append(gen_text)
                if len(all_embeddings[i]) > 0:
                    gen_embeddings.append(torch.stack(all_embeddings[i], dim=0))
                else:
                    gen_embeddings.append(None)
            else:
                gen_responses.append('')
                gen_embeddings.append(None)
        return gen_responses, gen_embeddings
