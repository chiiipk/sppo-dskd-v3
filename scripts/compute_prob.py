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
    # Đường dẫn mặc định đã là đường dẫn tuyệt đối, rất tốt.
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
        # SỬA LỖI: Bỏ dấu "/" ở đầu "ranking/"
        file_path = os.path.join(args.output_dir, "ranking", f"{idx}_{data_frac}.npy")
        try:
            locals_scores = np.load(file_path)
            locals_scores = list(locals_scores)
            for lidx, sc in enumerate(locals_scores):
                scores[data_frac * args.frac_len + lidx] = sc
        except FileNotFoundError:
            print(f"Warning: Ranking file not found at {file_path}. Skipping.")
            continue


    probs = []
    rm_scores = []
    for idx, score in enumerate(scores):
        if isinstance(score, int): # Bỏ qua nếu điểm số chưa được cập nhật
            prb = [[0.0] * pairs for _ in range(pairs)]
        else:
            prb = np.zeros((pairs, pairs))
            for i in range(pairs):
                for j in range(pairs):
                    prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
            prb = prb.tolist()
        probs.append(prb)
        rm_scores.append(score)

    # SỬA LỖI: Bỏ dấu "/" ở đầu "generated"
    out_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(out_dir, exist_ok=True)

    print("Saving probabilities...")
    with open(os.path.join(out_dir, "probabilities.json"), "w") as f:
        json.dump(probs, f)

    df = data.to_pandas()
    for i in range(pairs):
        resp_file = os.path.join(out_dir, f"responses_{i}.json")
        try:
            with open(resp_file) as f:
                responses = json.load(f)
            
            if len(responses) != len(data):
                 print(f"Warning: Mismatch length for responses_{i}.json. Padding with empty strings.")
                 responses.extend([""] * (len(data) - len(responses)))

            fmt = [
                [
                    {"content": data[j]["prompt"], "role": "user"},
                    {"content": responses[j], "role": "assistant"},
                ]
                for j in range(len(data))
            ]
            df[f"generate_{i}"] = fmt
        except FileNotFoundError:
            print(f"Warning: Response file not found at {resp_file}. Creating empty column.")
            df[f"generate_{i}"] = [[] for _ in range(len(df))]


    df["probability"] = probs
    df["rm_scores"] = rm_scores
    df.to_parquet(os.path.join(out_dir, "train.parquet"))

def prepare_score(args):
    # SỬA LỖI: Bỏ dấu "/" ở đầu "generated/"
    parquet_path = os.path.join(args.output_dir, "generated", "train.parquet")
    print(f"Loading parquet file from: {parquet_path}")
    train = datasets.load_dataset("parquet", data_files={"train": parquet_path})
    train = pd.DataFrame(train['train'])

    # --- FIX CUỐI CÙNG: Lọc sâu để loại bỏ các giá trị None bên trong list/matrix ---
    original_len = len(train)

    def is_valid_matrix(matrix, pairs):
        """Kiểm tra xem một ma trận có hợp lệ không (không None, đúng shape, và không chứa None bên trong)."""
        if not isinstance(matrix, list) or len(matrix) != pairs:
            return False
        for row in matrix:
            if not isinstance(row, list) or len(row) != pairs:
                return False
            if any(x is None for x in row): # Kiểm tra None bên trong từng hàng
                return False
        return True

    def is_valid_scores(scores, pairs):
        """Kiểm tra xem danh sách điểm số có hợp lệ không."""
        if not isinstance(scores, list) or len(scores) != pairs:
            return False
        if any(x is None for x in scores): # Kiểm tra None bên trong danh sách
            return False
        return True

    # Áp dụng các hàm lọc mạnh mẽ hơn
    train = train[train['probability'].apply(lambda x: is_valid_matrix(x, args.pairs))]
    train = train[train['rm_scores'].apply(lambda x: is_valid_scores(x, args.pairs))]

    print(f"Filtered out {original_len - len(train)} invalid rows. {len(train)} valid rows remaining.")
    
    if len(train) == 0:
        print("Error: No valid data remaining after filtering. Cannot create final dataset.")
        return None

    metrics = train['rm_scores'].apply(lambda x: np.array(x[-args.pairs:]))
    metrics_prob = train['probability'].apply(lambda x: np.stack(x).sum(axis=1))
    maxmin = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    train_ordered = train[[f"generate_{i}" for i in range(args.pairs)] + ['probability']]

    chosen = [train_ordered.iloc[i, maxmin.iloc[i][0]] for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin.iloc[i][1]] for i in range(len(train_ordered))]

    chosen_probs = [train_ordered['probability'].iloc[i][maxmin.iloc[i][0]][maxmin.iloc[i][1]] for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob.iloc[i][maxmin.iloc[i][0]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob.iloc[i][maxmin.iloc[i][1]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]

    train_new = pd.DataFrame({
        'chosen': chosen,
        'rejected': rejected,
        'chosen_probs': chosen_probs,
        'chosen_probs_win': chosen_probs_win,
        'chosen_probs_lose': chosen_probs_lose
    })
    
    # Lấy tên thư mục cuối cùng từ output_dir để đặt tên cho file synthetic
    output_base_name = os.path.basename(os.path.normpath(args.output_dir))
    OUTPATH = f'/kaggle/working/synthetic_data_{output_base_name}_score'
    os.makedirs(OUTPATH, exist_ok=True)

    train_new.to_parquet(f'{OUTPATH}/train.parquet', index=False)
    print(f"Saved file to {OUTPATH}/train.parquet")

    test = train_new.sample(n=min(500, len(train_new)))
    test.to_parquet(f'{OUTPATH}/test.parquet', index=False)
    print(f"Saved file to {OUTPATH}/test.parquet")

    return OUTPATH

def push_dataset(file_dir, org):
    if file_dir is None:
        print("Skipping push_dataset due to previous errors.")
        return
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
    # SỬA LỖI: Bỏ dấu "/" ở đầu "generated/"
    generated_parquet_path = os.path.join(args.output_dir, "generated", "train.parquet")
    if os.path.exists(generated_parquet_path):
        data = Dataset.from_parquet(generated_parquet_path)
        # SỬA LỖI: Cải thiện câu thông báo cho rõ ràng hơn
        print(f"Generated data saved locally to {os.path.dirname(generated_parquet_path)}")
        out_path = prepare_score(args)
        push_dataset(out_path, args.org)
    else:
        print(f"Error: Parquet file not found at {generated_parquet_path}. Cannot proceed to scoring and pushing dataset.")
