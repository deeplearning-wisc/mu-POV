
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(args, model_name_or_path):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, device_map='auto',
            token=args.token, cache_dir=args.model_dir, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: could not load model locally for {model_name_or_path}, downloading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, device_map='auto',
            token=args.token, cache_dir=args.model_dir, trust_remote_code=True)
    return model


class QwenWrapper:
    def __init__(self, args, model_dir, lora_adapter_path=None, memory_for_model_activations_in_gb=2):
        self.name = model_dir
        self.huggingface_model = load_model(args, model_dir)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, token=args.token, cache_dir=args.model_dir, local_files_only=True, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: could not load tokenizer locally for {model_dir}, downloading: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, token=args.token, cache_dir=args.model_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
