# scripts/compute_prob.py
import numpy as np
from datasets import Dataset
import json
import argparse
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--frac_len", type=int, default=6000)
    parser.add_argument("--num_gpu", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    return parser.parse_args()

def from_ranks(args):
    """Gathers raw data and saves an intermediate parquet file."""
    data_list = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    data = Dataset.from_list(data_list)
    n = len(data)
    print(f"Loaded {n} prompts.")

    scores = [None] * n
    gpus = args.gpu_ids.strip("()").split(',')
    ranking_dir = os.path.join(args.output_dir, "ranking")
    
    for data_frac_idx, gpu_id in enumerate(gpus):
        fn = os.path.join(ranking_dir, f"{gpu_id}_{data_frac_idx}.npy")
        if os.path.exists(fn):
            try:
                arr = np.load(fn, allow_pickle=True)
                start_index = data_frac_idx * args.frac_len
                end_index = start_index + len(arr)
                scores[start_index:end_index] = arr.tolist()
                print(f"Loaded {len(arr)} scores from {fn}")
            except Exception as e:
                print(f"Warning: Failed to load {fn}. Error: {e}")
        else:
            print(f"Warning: Ranking file not found: {fn}")

    df = data.to_pandas()
    generated_dir = os.path.join(args.output_dir, "generated")
    
    for p in range(args.pairs):
        resp_file = os.path.join(generated_dir, f"responses_{p}.json")
        if os.path.exists(resp_file):
            with open(resp_file, 'r', encoding='utf-8') as f:
                responses_p = json.load(f)
            convs = [[{"role": "user", "content": df["prompt"][i]}, {"role": "assistant", "content": responses_p[i]}] for i in range(len(df))]
            df[f"generate_{p}"] = convs
        else:
            df[f"generate_{p}"] = [None] * len(df)
            
    df["rm_scores"] = scores

    os.makedirs(generated_dir, exist_ok=True)
    parquet_path = os.path.join(generated_dir, "train.parquet")
    df.to_parquet(parquet_path, index=False)
    print("Saved intermediate parquet to:", parquet_path)
    return parquet_path

def prepare_score_from_parquet(parquet_path, args):
    """Filters data and creates the final training dataset."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Intermediate parquet file not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    original_len = len(df)
    
    df.dropna(subset=['rm_scores'], inplace=True)
    # df = df[df['rm_scores'].apply(lambda s: isinstance(s, list) and len(s) == args.pairs)]
    def is_valid_scores(s):
        # Chấp nhận cả list và mảng NumPy
        is_list_or_array = isinstance(s, (list, np.ndarray))
        if not is_list_or_array:
            return False
        # Kiểm tra độ dài
        return len(s) == args.pairs
    
    df = df[df['rm_scores'].apply(is_valid_scores)]
    
    print(f"Filtered {original_len - len(df)} invalid rows -> {len(df)} remaining.")
    if len(df) == 0:
        raise RuntimeError("No valid rows after filtering.")

    metrics = np.vstack(df['rm_scores'].values)
    chosen_idx = np.argmax(metrics, axis=1)
    rejected_idx = np.argmin(metrics, axis=1)

    # Trích xuất trực tiếp vì bây giờ cột chỉ chứa chuỗi văn bản
    chosen_text = [df.iloc[i][f"generate_{chosen_idx[i]}"] for i in range(len(df))]
    rejected_text = [df.iloc[i][f"generate_{rejected_idx[i]}"] for i in range(len(df))]

    train_new = pd.DataFrame({
        "text_prompt": df["prompt"].tolist(),
        "text_chosen": chosen_text,
        "text_rejected": rejected_text
    })

    base = os.path.basename(os.path.normpath(args.output_dir))
    outdir = f"/kaggle/working/synthetic_data_{base}_score"
    os.makedirs(outdir, exist_ok=True)
    train_new.to_parquet(os.path.join(outdir, "train.parquet"), index=False)
    
    test = train_new.sample(n=min(500, len(train_new)), random_state=42)
    test.to_parquet(os.path.join(outdir, "test.parquet"), index=False)
    
    print("Saved final synthetic dataset to:", outdir)
    return outdir

if __name__ == "__main__":
    args = parse_arguments()
    intermediate_parquet_path = from_ranks(args)
    final_dataset_dir = prepare_score_from_parquet(intermediate_parquet_path, args)
    print("Pipeline complete. Final dataset is at:", final_dataset_dir)
