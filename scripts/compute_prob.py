# save as scripts/prepare_from_ranks.py
import numpy as np
from datasets import Dataset
import json
import argparse
import pandas as pd
import datasets
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    # args.output_dir should be the OUT directory you used for generate/combine (e.g. /kaggle/working/kd-... )
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/kd-gpt2-qwen-dolly-iter1")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--prompts", type=str, default="/kaggle/working/data/dolly/train.jsonl")
    parser.add_argument("--frac_len", type=int, default=6000)
    parser.add_argument("--num_gpu", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--data_frac", type=int, default=0)       # <-- ADDED: data_frac used when filenames include it
    parser.add_argument("--ranking_root", type=str, default=None)  # optional: explicit ranking root
    parser.add_argument("--org", type=str, default="local")
    return parser.parse_args()

def load_prompts(prompts_path):
    data_list = []
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_list.append(item)
    return Dataset.from_list(data_list)

def from_ranks(args):
    # load prompts
    data = load_prompts(args.prompts)
    n = len(data)
    print(f"Loaded {n} prompts from {args.prompts}")

    # determine ranking directory:
    # if user passed --ranking_root, use it; otherwise expect ranking under args.output_dir/ranking
    if args.ranking_root:
        ranking_dir = args.ranking_root
    else:
        ranking_dir = os.path.join(args.output_dir, "ranking")
    print("Looking for ranking files under:", ranking_dir)
    print("Using data_frac:", args.data_frac)

    # determine GPU ids list
    if args.gpu_ids:
        gpus = args.gpu_ids.strip("()").split(',')
        gpus = [g.strip() for g in gpus if g.strip() != ""]
    else:
        gpus = [str(i) for i in range(args.num_gpu)]
    print("GPU ids:", gpus)

    # init scores placeholder (each entry should be a list of length pairs after fill)
    scores = [None] * n

    # read ranking npy for each gpu/data_frac
    for data_frac_idx, gpu_id in enumerate(gpus):
        fn = os.path.join(ranking_dir, f"{gpu_id}_{data_frac_idx}.npy")
        if not os.path.exists(fn):
            print(f"Warning: ranking file not found: {fn} (skipping)")
            continue
        arr = np.load(fn, allow_pickle=True)
        arr = list(arr)
        # fill into scores array using frac_len
        start = data_frac_idx * args.frac_len if args.frac_len > 0 else 0
        for i, val in enumerate(arr):
            idx = start + i
            if idx < n:
                scores[idx] = val
            else:
                print(f"Warning: index {idx} >= n ({n}), skipping extra ranking entry")

    # replace any None with fallback (zeros list)
    for i in range(n):
        if scores[i] is None:
            scores[i] = [0.0] * args.pairs
        else:
            # ensure length pairs
            if len(scores[i]) != args.pairs:
                arr = list(scores[i])
                if len(arr) < args.pairs:
                    arr = arr + [0.0] * (args.pairs - len(arr))
                else:
                    arr = arr[:args.pairs]
                scores[i] = arr

    # compute probabilities (pairwise Bradley-Terry-like) robustly
    probs = []
    for sc in scores:
        sc_arr = np.array(sc, dtype=float)
        prb = np.zeros((args.pairs, args.pairs), dtype=float)
        for i in range(args.pairs):
            for j in range(args.pairs):
                prb[i, j] = 1.0 / (1.0 + np.exp(sc_arr[j] - sc_arr[i]))
        probs.append(prb.tolist())

    # responses (combined) might be in args.output_dir/generated/responses_*.json or args.output_dir/responses_*.json
    responses = []
    for p in range(args.pairs):
        # check typical locations
        candidates = [
            os.path.join(args.output_dir, "generated", f"responses_{p}.json"),
            os.path.join(args.output_dir, f"responses_{p}.json"),
            os.path.join(args.output_dir, "generated", f"responses_{args.data_frac}_{p}.json"),
            os.path.join(args.output_dir, f"responses_{args.data_frac}_{p}.json")
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            print(f"Warning: response file for p={p} not found in expected locations. Will fill with empty strings.")
            responses.append([""] * n)
            continue
        with open(found, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        if len(arr) != n:
            print(f"Warning: {found} length {len(arr)} != n {n}. Padding/truncating.")
            if len(arr) < n:
                arr = arr + [""] * (n - len(arr))
            else:
                arr = arr[:n]
        responses.append(arr)

    # build dataframe like your parquet (generate_0.. generate_{pairs-1}, probability, rm_scores)
    df = pd.DataFrame()
    for p in range(args.pairs):
        convs = []
        for i in range(n):
            prompt_text = data[i]["prompt"] if "prompt" in data[i] else ""
            convs.append([
                {"content": prompt_text, "role": "user"},
                {"content": responses[p][i], "role": "assistant"}
            ])
        df[f"generate_{p}"] = convs

    df["probability"] = probs
    df["rm_scores"] = scores

    # Save parquet under args.output_dir/generated/train.parquet (keeps consistency)
    generated_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    parquet_path = os.path.join(generated_dir, "train.parquet")
    df.to_parquet(parquet_path, index=False)
    print("Saved combined parquet to:", parquet_path)
    return parquet_path

def prepare_score_from_parquet(parquet_path, args):
    print("Loading parquet:", parquet_path)
    df = pd.read_parquet(parquet_path)
    original_len = len(df)

    # validation helpers
    def is_valid_matrix(m):
        if not isinstance(m, list): return False
        if len(m) != args.pairs: return False
        for row in m:
            if not isinstance(row, list) or len(row) != args.pairs:
                return False
            if any(x is None for x in row):
                return False
        return True

    def is_valid_scores(s):
        if not isinstance(s, list): return False
        if len(s) != args.pairs: return False
        if any(x is None for x in s): return False
        return True

    mask = df['probability'].apply(lambda x: is_valid_matrix(x)) & df['rm_scores'].apply(lambda x: is_valid_scores(x))
    df_valid = df[mask].reset_index(drop=True)
    print(f"Filtered {original_len - len(df_valid)} invalid rows -> {len(df_valid)} remaining.")
    if len(df_valid) == 0:
        raise RuntimeError("No valid rows after filtering. Cannot prepare final dataset.")

    # metrics array: shape (N, pairs)
    metrics = np.vstack(df_valid['rm_scores'].apply(lambda x: np.array(x[:args.pairs], dtype=float)).values)
    # chosen index = argmax, rejected index = argmin
    chosen_idx = np.argmax(metrics, axis=1)
    rejected_idx = np.argmin(metrics, axis=1)

    # helper to extract assistant text from generate_* conversation struct
    def extract_assistant(conv):
        if not isinstance(conv, list) or len(conv) == 0:
            return ""
        for item in reversed(conv):
            if isinstance(item, dict) and item.get('role','') == 'assistant':
                return item.get('content','')
        last = conv[-1]
        return last.get('content','') if isinstance(last, dict) else str(last)

    prompts_out, chosen_out, rejected_out = [], [], []
    for i in range(len(df_valid)):
        ci = chosen_idx[i]
        ri = rejected_idx[i]
        gen_c = df_valid.loc[i, f"generate_{ci}"] if f"generate_{ci}" in df_valid.columns else []
        gen_r = df_valid.loc[i, f"generate_{ri}"] if f"generate_{ri}" in df_valid.columns else []
        chosen_txt = extract_assistant(gen_c)
        rejected_txt = extract_assistant(gen_r)
        prompt_txt = ""
        if isinstance(gen_c, list) and len(gen_c) > 0 and isinstance(gen_c[0], dict):
            prompt_txt = gen_c[0].get('content','')
        prompts_out.append(prompt_txt)
        chosen_out.append(chosen_txt)
        rejected_out.append(rejected_txt)

    train_new = pd.DataFrame({
        "text_prompt": prompts_out,
        "text_chosen": chosen_out,
        "text_rejected": rejected_out
    })

    # base from args.output_dir to create synthetic folder name
    base = os.path.basename(os.path.normpath(args.output_dir))
    outdir = f"/kaggle/working/synthetic_data_{base}_score"
    os.makedirs(outdir, exist_ok=True)
    train_new.to_parquet(os.path.join(outdir, "train.parquet"), index=False)
    test = train_new.sample(n=min(500, len(train_new)))
    test.to_parquet(os.path.join(outdir, "test.parquet"), index=False)
    print("Saved synthetic dataset to:", outdir)
    return outdir

if __name__ == "__main__":
    args = parse_arguments()
    parquet_path = from_ranks(args)           # produces OUT/generated/train.parquet
    out_dir = prepare_score_from_parquet(parquet_path, args)
    print("Done. synthetic dataset:", out_dir)
