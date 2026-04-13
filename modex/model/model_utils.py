
import torch
from transformers import set_seed


# Mapping from short model names to HuggingFace model IDs
MODEL_REGISTRY = {
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-70b-chat': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3.3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'codellama': 'meta-llama/CodeLlama-7b-Instruct-hf',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct',
}


def get_agents(args, peft_path=None):
    """
    Load the specified model and return (agent, personas).

    The agent is a thin wrapper around a HuggingFace model that exposes
    `.tokenizer` and `.huggingface_model`.  Personas are optional system
    prompts drawn from the DyLAN paper (https://arxiv.org/pdf/2310.02170).
    """
    model_id = MODEL_REGISTRY.get(args.model)
    if model_id is None:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # # Override hub ID with a local path if one was provided
    # if args.model_dir is not None:
    #     import os
    #     model_id = args.model_dir#os.path.join(args.model_dir, model_id.replace('/', os.sep))

    mem = args.memory_for_model_activations_in_gb

    if args.model in ['llama3.1-8b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
        from model.llama import LlamaWrapper
        agent = LlamaWrapper(args, model_id, memory_for_model_activations_in_gb=mem, lora_adapter_path=peft_path, llama_version=3)
    elif args.model in ['llama2-70b-chat', 'llama2-13b-chat', 'llama2-7b-chat', 'codellama']:
        from model.llama import LlamaWrapper
        agent = LlamaWrapper(args, model_id, memory_for_model_activations_in_gb=mem, lora_adapter_path=peft_path, llama_version=2)
    elif args.model in ['qwen2.5-1.5b', 'qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b']:
        from model.qwen import QwenWrapper
        agent = QwenWrapper(args, model_id, memory_for_model_activations_in_gb=mem, lora_adapter_path=peft_path)
    else:
        raise ValueError(f"No wrapper implemented for model '{args.model}'.")

    # Ensure a pad token exists
    if agent.tokenizer.pad_token is None:
        agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        agent.huggingface_model.resize_token_embeddings(len(agent.tokenizer))

    # Personas drawn from DyLAN: https://arxiv.org/pdf/2310.02170
    if args.multi_persona:
        personas = {
            "Assistant":         "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
            "Mathematician":     "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist":         "You are an economist. You are good at economics, finance, and business.",
            "Psychologist":      "You are a psychologist. You are good at psychology, sociology, and philosophy.",
            "Lawyer":            "You are a lawyer. You are good at law, politics, and history.",
            "Doctor":            "You are a doctor. You come up with creative treatments for illnesses or diseases.",
            "Programmer":        "You are a programmer. You are good at computer science, engineering, and physics.",
            "Historian":         "You are a historian. You research and analyze cultural, economic, political, and social events in the past.",
            "PythonAssistant":   "You are a Python writing assistant. You only respond with Python code, NOT English.",
            "AlgorithmDeveloper":"You are an algorithm developer. You respond with Python code, no free-flowing text.",
            "ComputerScientist": "You are a computer scientist. You write high-performance Python code.",
            "CodingArtist":      "You are a coding artist. You write Python code that is functional and aesthetically pleasing.",
            "SoftwareArchitect": "You are a software architect, skilled in designing scalable and maintainable code.",
        }
        # Task-specific persona subsets
        if args.data in ['arithmetics', 'gsm8k']:
            personas = {k: personas[k] for k in ["Assistant", "Mathematician", "Lawyer", "Economist", "Programmer"]}
    else:
        personas = {"None": ""}

    return agent, personas
