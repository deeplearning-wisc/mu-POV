
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        self.name = model_dir
        self.huggingface_model = load_model(args, model_dir, llama_version)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
