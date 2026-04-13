
import torch
from transformers import set_seed


def engine(messages, agent, num_agents=1, stop_sequences=None, judge=False):
    """
    Run batched generation for a list of chat messages.

    When ``agent.args.new_decode`` is ``True`` (ModeX-Lite mode), generation is
    delegated to the model wrapper's custom ``generate()`` method, which
    performs online similarity-based pruning of the decoding batch.
    Otherwise, standard HuggingFace ``generate()`` is used.

    Args:
        messages: List of chat message dicts.
        agent: Model wrapper exposing `.tokenizer`, `.huggingface_model`, and
               optionally a custom `.generate()` for ModeX-Lite decoding.
        num_agents: Number of sequences to decode.
        judge: If True, skip ModeX-Lite decoding (used for LLM-judge calls).

    Returns:
        prompts: Tokenized prompt strings.
        responses: Decoded response strings.
        ppl: None (perplexity not computed here).
        output_embeddings: Tensor of shape (batch, hidden_dim), or None.
    """
    try:
        prompts = [agent.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages]
    except Exception:
        prompts = [agent.tokenizer.apply_chat_template([msgs], tokenize=False, add_generation_prompt=True) for msgs in messages]

    inputs = agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(agent.huggingface_model.device)
    attention_mask = inputs['attention_mask'].to(agent.huggingface_model.device)

    if 'PhiWrapper' in type(agent).__name__:
        temperature, top_p = 0.8, 0.95
    elif 'Qwen' in type(agent).__name__:
        temperature, top_p = 1.2, 0.95
    elif 'Llama' in type(agent).__name__:
        temperature, top_p = 1.0, 0.95
    else:
        temperature, top_p = 1.0, 0.9

    set_seed(42)

    # ModeX-Lite: delegate to custom generate() for online pruning
    if getattr(agent.args, 'new_decode', False) and not judge:
        responses, _ = agent.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=agent.tokenizer.eos_token_id,
            max_new_tokens=3072 if 'PhiWrapper' in type(agent).__name__ else 2048,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return prompts, responses

    # Standard generation
    outputs = agent.huggingface_model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=agent.tokenizer.eos_token_id,
        max_new_tokens=3072 if 'PhiWrapper' in type(agent).__name__ else 2048,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
    )

    responses = []
    for input_id, sequence in zip(input_ids, outputs.sequences):
        gen_only = sequence[len(input_id):]
        responses.append(agent.tokenizer.decode(gen_only, skip_special_tokens=True))

    return prompts, responses
