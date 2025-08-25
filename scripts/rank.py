from datasets import Dataset
import json
import argparse
import llm_blender
import os
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output_dir", type=str, default="iter1")   # chỉ tên iter
    parser.add_argument("--prompts", type=str, default="/kaggle/working/data/dolly/train.jsonl")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--data_frac", type=int, default=0)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--numgpu", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)# local rank
    

    # teacher LLM options
    parser.add_argument("--use_teacher_llm", action="store_true", help="Use teacher LLM instead of PairRM")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen1.5-1.8B-Chat", help="Teacher model for scoring (HuggingFace model)")
    parser.add_argument("--scoring_prompt_template", type=str, default=None, help="Path to scoring prompt template")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for teacher LLM scoring")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism in vLLM")

    parser.add_argument("--bt_conversion_method", type=str, default="bradley_terry_mle",
                        choices=["bradley_terry_mle", "score_difference", "elo_simulation", "percentile_ranking"],
                        help="Method to convert absolute scores to Bradley-Terry format")
    return parser.parse_args()


def apply_template(text, tokenizer):
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
            tokenize=False, add_generate_prompt=True
        ).split("None")[0]
    return text


def split_prompts(prompts, frac_len, data_frac):
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    return prompts[:]


def simulate_pairwise_from_scores(absolute_scores, method="bradley_terry_mle"):
    import numpy as np
    scores = np.array(absolute_scores, dtype=float)

    if method == "bradley_terry_mle":
        exp_scores = np.exp(scores)
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
                    diff = scores[i] - scores[j]
                    pairwise_probs[i][j] = 1 / (1 + np.exp(-diff))
        avg_win_prob = np.mean(pairwise_probs, axis=1)
        avg_win_prob = np.clip(avg_win_prob, 1e-10, 1 - 1e-10)
        bt_scores = np.log(avg_win_prob / (1 - avg_win_prob))

    elif method == "percentile_ranking":
        from scipy.stats import rankdata
        ranks = rankdata(scores, method='average')
        percentiles = (ranks - 1) / (len(scores) - 1)
        percentiles = np.clip(percentiles, 1e-10, 1 - 1e-10)
        bt_scores = np.log(percentiles / (1 - percentiles))

    else:
        raise ValueError(f"Unknown method: {method}")

    return bt_scores


def ranking(args, prompts, candidates):
    """Rank responses using either PairRM or teacher LLM with vLLM."""
    if args.use_teacher_llm:
        from tqdm import tqdm
        import re

        llm = LLM(
            model=args.teacher_model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=2048
        )

        all_scores = []
        for i, (prompt, responses) in enumerate(tqdm(zip(prompts, candidates), total=len(prompts))):
            scoring_prompts = []
            for response in responses:
                scoring_prompts.append(
                    f"Rate the following response to the prompt on a scale 1–10.\n\nPrompt: {prompt}\n\nResponse: {response}\n\nScore:"
                )

            outputs = llm.generate(scoring_prompts, SamplingParams(temperature=0.1, max_tokens=10))
            scores = []
            for out in outputs:
                text = out.outputs[0].text
                match = re.search(r'(\d+\.?\d*)', text)
                score = float(match.group(1)) if match else 5.0
                scores.append(score)

            bt_scores = simulate_pairwise_from_scores(scores, method=args.bt_conversion_method)
            all_scores.append(bt_scores)

        scores = np.array(all_scores)

    else:
        import llm_blender
        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM")
        scores = blender.rank(prompts, candidates, return_scores=True, batch_size=1)

        # --- FIX: SỬA LẠI LOGIC TẠO ĐƯỜNG DẪN OUTPUT ---
    # Nếu args.output_dir là đường dẫn tuyệt đối (bắt đầu bằng '/'), dùng nó.
    # Nếu không, coi nó là tên thư mục và nối vào /kaggle/working/.
    base_output_dir = args.output_dir if args.output_dir.startswith('/') else os.path.join("/kaggle/working", args.output_dir)
    
    out_dir = os.path.join(base_output_dir, "ranking")
    os.makedirs(out_dir, exist_ok=True)
    
    output_path = os.path.join(out_dir, f"{args.gpu}_{args.data_frac}.npy")
    np.save(output_path, scores)
    print(f"Successfully saved {len(scores)} rankings to {output_path}")

def main(args):
    # load dataset
    data_list = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_list.append(item)
    data = Dataset.from_list(data_list)

    # tokenizer
    if args.use_teacher_llm:
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
            raise ValueError("Unsupported model")

    tokenizer.pad_token = tokenizer.eos_token

    prompts_all = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]
    print("Example prompt:", prompts_all[0])

        # --- FIX: SỬA LẠI LOGIC ĐỌC FILE RESPONSE ---
    # Áp dụng logic tương tự như trên để xác định thư mục cơ sở
    base_output_dir = args.output_dir if args.output_dir.startswith('/') else os.path.join("/kaggle/working", args.output_dir)

    all_generated = []
    for i in range(args.pairs):
        # Nối thư mục 'generated' và tên file vào đường dẫn cơ sở
        file_path = os.path.join(base_output_dir, "generated", f"responses_{i}.json")
        try:
            with open(file_path) as f:
                gen = json.load(f)
                all_generated.append(gen)
        except FileNotFoundError:
            print(f"ERROR: Response file not found at {file_path}. Cannot proceed with ranking.")
            return

    candidates_texts = list(zip(*all_generated))
    assert len(data) == len(candidates_texts)
    print(f"Length of data: {len(data)}")

    # split if needed
    prompts_all = split_prompts(prompts_all, args.frac_len, args.data_frac)
    candidates_texts = split_prompts(candidates_texts, args.frac_len, args.data_frac)

    ranking(args, prompts_all, candidates_texts)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
