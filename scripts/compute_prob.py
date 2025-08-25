import numpy as np
from datasets import load_dataset, Dataset
import json
import argparse
import pandas as pd
import datasets
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/generated/iter1")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--prompts", type=str, default="/kaggle/working/data/dolly/train.jsonl")
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--num_gpu", type=int, default=8)
    parser.add_argument("--org", type=str, default="local")
    parser.add_argument("--gpu_ids", type=str, default=None)
    return parser.parse_args()

def from_ranks(args):
    num_gpu = args.num_gpu
    pairs = args.pairs

    data_list = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_list.append(item)
    data = Dataset.from_list(data_list)
    print(f"Length of dataset: {len(data)}")

    scores = [0 for _ in range(len(data))]
    if args.gpu_ids is not None:
        gpus = args.gpu_ids.strip("()").split(',')
    else:
        gpus = range(args.num_gpu)

    for data_frac, idx in enumerate(gpus):
        file_path = os.path.join(args.output_dir, "/ranking/", f"{idx}_{data_frac}.npy")
        locals = np.load(file_path)
        locals = list(locals)
        for lidx, sc in enumerate(locals):
            scores[data_frac * args.frac_len + lidx] = sc

    probs = []
    rm_scores = []
    for idx, score in enumerate(scores):
        prb = np.zeros((pairs, pairs))
        for i in range(pairs):
            for j in range(pairs):
                prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
        prb = prb.tolist()
        probs.append(prb)
        rm_scores.append(score)

    out_dir = os.path.join(args.output_dir,"/generated")
    os.makedirs(out_dir, exist_ok=True)

    print("Saving probabilities...")
    with open(os.path.join(out_dir, "probabilities.json"), "w") as f:
        json.dump(probs, f)

    df = data.to_pandas()
    for i in range(pairs):
        resp_file = os.path.join(out_dir, f"responses_{i}.json")
        with open(resp_file) as f:
            responses = json.load(f)
        fmt = [
            [
                {"content": data[j]["prompt"], "role": "user"},
                {"content": responses[j], "role": "assistant"},
            ]
            for j in range(len(data))
        ]
        df[f"generate_{i}"] = fmt

    df["probability"] = probs
    df["rm_scores"] = rm_scores
    df.to_parquet(os.path.join(out_dir, "train.parquet"))

def prepare_score(args):
    # Load dataset
    train = datasets.load_dataset("parquet", data_files={ "train": os.path.join( args.output_dir, "/generated/", "train.parquet") })
    train = pd.DataFrame(train['train'])

    metrics = train['rm_scores'].apply(lambda x: np.array(x[-5:]))
    metrics_prob = train['probability'].apply(lambda x: np.stack(x).sum(axis=1))
    maxmin = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    train_ordered = train[[f"generate_{i}" for i in range(args.pairs)] + ['probability']]

    chosen = [train_ordered.iloc[i, maxmin[i][0]] for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin[i][1]] for i in range(len(train_ordered))]

    chosen_probs = [train_ordered['probability'].iloc[i][maxmin[i][0]][maxmin[i][1]] for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob[i][maxmin[i][0]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob[i][maxmin[i][1]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]

    train_new = pd.DataFrame({
        'chosen': chosen,
        'rejected': rejected,
        'chosen_probs': chosen_probs,
        'chosen_probs_win': chosen_probs_win,
        'chosen_probs_lose': chosen_probs_lose
    })

    output_dir = '-'.join(args.output_dir.split('-')[1:])
    OUTPATH = f'/kaggle/working/synthetic_data_{output_dir}_score'
    os.makedirs(OUTPATH, exist_ok=True)

    train_new.to_parquet(f'{OUTPATH}/train.parquet', index=False)
    print(f"Saved file to {OUTPATH}/train.parquet")

    test = train_new.sample(n=min(500, len(train_new)))
    test.to_parquet(f'{OUTPATH}/test.parquet', index=False)
    print(f"Saved file to {OUTPATH}/test.parquet")

    return OUTPATH

def push_dataset(file_dir, org):
    data = Dataset.from_parquet(f"{file_dir}/train.parquet")
    try:
        test = Dataset.from_parquet(f"{file_dir}/test.parquet")
    except:
        train = pd.read_parquet(f"{file_dir}/train.parquet")
        test = train.sample(n=min(500, len(train)))
        test.to_parquet(f"{file_dir}/test.parquet", index=False)
        test = Dataset.from_parquet(f"{file_dir}/test.parquet")
    # data.push_to_hub(f"{org}/{file_dir}", split="train", private=True)
    # test.push_to_hub(f"{org}/{file_dir}", split="test", private=True)

if __name__ == "__main__":
    args = parse_arguments()
    from_ranks(args)
    data = Dataset.from_parquet(os.path.join(args.output_dir, "/generated/", "train.parquet"))
    print(f"Generated data saved locally to /kaggle/working/generated/{args.output_dir}/")
    out_path = prepare_score(args)
    push_dataset(out_path, args.org)
