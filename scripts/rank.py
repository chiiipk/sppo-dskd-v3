from datasets import load_dataset
import json
import pandas as pd
import argparse
import llm_blender
import os
import numpy as np
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument("--numgpu", type=int, default=8)
    parser.add_argument('--prompts', type=str, default='d:/Python/GenAI/DSKD/data/dolly/train.jsonl')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)  # local rank
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--use_teacher_llm", action="store_true", help="Use teacher LLM instead of PairRM")
    parser.add_argument("--teacher_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Teacher model for scoring (HuggingFace model)")
    parser.add_argument("--scoring_prompt_template", type=str, default=None, help="Path to scoring prompt template")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for teacher LLM scoring")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism in vLLM")
    
    parser.add_argument("--bt_conversion_method", type=str, default="bradley_terry_mle",
                       choices=["bradley_terry_mle", "score_difference", "elo_simulation", "percentile_ranking"],
                       help="Method to convert absolute scores to Bradley-Terry format")
    
    return parser.parse_args()



def create_scoring_prompt(prompt, response, all_responses, template_path=None):
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template = f.read()
        return template.format(prompt=prompt, response=response, all_responses=all_responses)
    
    other_responses = [r[:200] + "..." if len(r) > 200 else r for r in all_responses if r != response]
    context = "\n\n".join([f"Reference Response {i+1}: {resp}" for i, resp in enumerate(other_responses[:3])])
    
    return f"""Please evaluate the following response to the given prompt on a scale from 1 to 10, considering how it compares to the reference responses shown below.

Consider the following criteria:
- Helpfulness and relevance to the prompt
- Accuracy and correctness
- Clarity and coherence
- Completeness of the response

Prompt: {prompt}

Reference Responses:
{context}

Response to Evaluate: {response}

Rate this response from 1-10 compared to the reference responses. A score of 5-6 means it's about average compared to the references, 1-4 means it's worse, and 7-10 means it's better.

Please provide only a numerical score between 1 and 10.

Score:"""

def simulate_pairwise_from_scores(absolute_scores, method="bradley_terry_mle"):
    """
    Simulate pairwise comparison results from absolute scores using various methods.
    
    Args:
        absolute_scores: Array of absolute scores for responses
        method: Method to use for simulation
    
    Returns:
        Array of Bradley-Terry compatible scores
    """
    scores = np.array(absolute_scores, dtype=float)
    
    if method == "bradley_terry_mle":
        temperature = 1.0
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        bt_scores = np.log(probabilities)
        
        bt_scores = bt_scores - np.mean(bt_scores)
        
    elif method == "score_difference":
        n = len(scores)
        pairwise_probs = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    score_diff = scores[i] - scores[j]
                    pairwise_probs[i][j] = 1 / (1 + np.exp(-score_diff))
        
        avg_win_prob = np.mean(pairwise_probs, axis=1)
        
        epsilon = 1e-10
        avg_win_prob = np.clip(avg_win_prob, epsilon, 1 - epsilon)
        bt_scores = np.log(avg_win_prob / (1 - avg_win_prob))
        
    elif method == "percentile_ranking":
        from scipy.stats import rankdata
        ranks = rankdata(scores, method='average')
        percentiles = (ranks - 1) / (len(scores) - 1)
        
        epsilon = 1e-10
        percentiles = np.clip(percentiles, epsilon, 1 - epsilon)
        bt_scores = np.log(percentiles / (1 - percentiles))
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return bt_scores



def score_with_teacher_vllm(prompts, all_responses, llm, max_length=1024, batch_size=4):
    """Score responses using vLLM for efficient batched scoring."""
    import re
    
    all_scores = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = all_responses[i:i+batch_size]
        
        batch_scoring_prompts = []
        batch_indices = []
        
        for batch_idx, (prompt, responses) in enumerate(zip(batch_prompts, batch_responses)):
            for response_idx, response in enumerate(responses):
                scoring_prompt = create_scoring_prompt(prompt, response, responses)
                batch_scoring_prompts.append(scoring_prompt)
                batch_indices.append((batch_idx, response_idx))
        
        if not batch_scoring_prompts:
            continue
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=10,
            stop=None
        )
        
        outputs = llm.generate(batch_scoring_prompts, sampling_params)
        
        batch_scores = []
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            score_match = re.search(r'(\d+\.?\d*)', generated_text)
            if score_match:
                score = float(score_match.group(1))
                batch_scores.append(score)
            else:
                print(f"Warning: Could not parse score from response: {generated_text}")
                batch_scores.append(5.0)
        
        current_prompt_scores = []
        current_prompt_idx = 0
        
        for j, (batch_idx, response_idx) in enumerate(batch_indices):
            if batch_idx != current_prompt_idx:
                all_scores.append(current_prompt_scores)
                current_prompt_scores = []
                current_prompt_idx = batch_idx
            current_prompt_scores.append(batch_scores[j])
        
        if current_prompt_scores:
            all_scores.append(current_prompt_scores)
    
    return all_scores



def score_with_teacher_llm(prompts, candidates, teacher_model, batch_size=4, template_path=None, bt_method="bradley_terry_mle", tensor_parallel_size=1):
    """Score responses using vLLM for efficient batched scoring then convert to Bradley-Terry format."""
    print(f"Scoring {len(prompts)} prompts with vLLM batched scoring using {teacher_model}")
    print(f"Batch size: {batch_size}, Bradley-Terry conversion method: {bt_method}, Tensor parallel size: {tensor_parallel_size}")
    
    try:
        llm = LLM(
            model=teacher_model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=2048
        )
    except Exception as e:
        print(f"Error loading model {teacher_model} with vLLM: {e}")
        raise
    
    absolute_scores_list = score_with_teacher_vllm(prompts, candidates, llm, batch_size=batch_size)
    
    all_scores = []
    for absolute_scores in absolute_scores_list:
        bt_scores = simulate_pairwise_from_scores(absolute_scores, method=bt_method)
        all_scores.append(bt_scores)
    
    return np.array(all_scores)


def ranking(args, prompts, candidates):
    """Rank responses using either PairRM or teacher LLM with vLLM."""
    if args.use_teacher_llm:
        scores = score_with_teacher_llm(
            prompts,
            candidates,
            args.teacher_model,
            args.batch_size,
            args.scoring_prompt_template,
            getattr(args, 'bt_conversion_method', 'bradley_terry_mle'),
            args.tensor_parallel_size
        )
    else:
        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM")
        scores = blender.rank(prompts, candidates, return_scores=True, batch_size=1)

    np.save(f"ranking/{args.output_dir}/{args.gpu}_{args.data_frac}.npy", scores)


def split_prompts(prompts, frac_len, data_frac):
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return prompts[:]


def apply_template(text, tokenizer):
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
            tokenize=False, add_generate_prompt=True
        ).split("None")[0]
    else:
        return text



def main(args):
    if args.prompts.endswith('.jsonl'):
        from datasets import Dataset
        data_list = []
        with open(args.prompts, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data_list.append(item)
        data = Dataset.from_list(data_list)

    if args.use_teacher_llm and args.teacher_model:
        if "qwen" in args.teacher_model.lower():
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
        elif "mistral" in args.teacher_model.lower():
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        elif "llama" in args.teacher_model.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        elif "gemma" in args.teacher_model.lower():
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    else:
        if "mistral" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        elif "llama-3" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        elif "gemma-2" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        elif "gpt2" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            raise ValueError("Must contain model name in the model argument. Supported models: Mistral/Llama-3/Gemma-2/GPT-2")

    tokenizer.pad_token = tokenizer.eos_token

    prompts_all = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]
    print(prompts_all[0])
    pairs = args.pairs
    all_generated = []

    for i in range(pairs):
        file_path = f"generated/{args.output_dir}/responses_{i}.json"
        with open(file_path) as f:
            gen = json.load(f)
            all_generated.append(gen)

    candidates_texts = list(zip(*all_generated))
    assert len(data) == len(candidates_texts)
    print(f'Length of data: {len(data)}')

    data_frac = args.data_frac
    os.makedirs(f"ranking/{args.output_dir}", exist_ok=True)

    data_frac, frac_len = args.data_frac, args.frac_len
    prompts_all = split_prompts(prompts_all, frac_len, data_frac)
    candidates_texts = split_prompts(candidates_texts, frac_len, data_frac)

    ranking(args, prompts_all, candidates_texts)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)