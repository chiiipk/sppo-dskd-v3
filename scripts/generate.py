from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from vllm import LLM, SamplingParams

import argparse
import torch
import json
import os
from pathlib import Path
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument("--prompts", type=str, default="d:/Python/GenAI/DSKD/data/dolly/train.jsonl")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--data_frac", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--knowledge_distillation", action="store_true", help="Enable knowledge distillation mode")
    return parser.parse_args()


def apply_template(text, tokenizer, knowledge_distillation=False):
    if knowledge_distillation:
        return text
    else:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
                tokenize=False, add_generate_prompt=True
            ).split("None")[0]
        else:
            return text


def split_prompts(prompts, frac_len, data_frac):
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return prompts[:]


def main():
    args = parse_arguments()
    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts.endswith('.jsonl'):
        from datasets import Dataset
        data_list = []
        with open(args.prompts, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data_list.append(item)
        data = Dataset.from_list(data_list)
        print(f"Loaded {len(data)} examples from {args.prompts}")
    else:
        data = load_dataset(args.prompts, split="train")

    if "gpt2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif "qwen" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
    elif "mistral" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif "llama-3" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "gemma-2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    else:
        raise ValueError("Model not supported")
    tokenizer.pad_token = tokenizer.eos_token

    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.world_size,
    )
    prompts = [apply_template(data[idx]["prompt"], tokenizer, args.knowledge_distillation) for idx in range(len(data))]
    print(prompts[0])
    data_frac, frac_len = args.data_frac, args.frac_len
    prompts = split_prompts(prompts, frac_len, data_frac)

    pairs = args.pairs

    os.makedirs(args.output_dir, exist_ok=True)

    for p in range(pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        with open(f"{args.output_dir}/responses_{data_frac}_{p}.json", "w") as f:
            json.dump(output, f)


if __name__ == "__main__":
    main()
