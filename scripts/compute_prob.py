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
    """
    Loads intermediate parquet, filters for valid data, and creates the final training dataset
    with the correct flattened string format.
    """
    base_output_dir = args.output_dir if args.output_dir.startswith('/') else os.path.join("/kaggle/working", args.output_dir)
    generated_dir = os.path.join(base_output_dir, "generated")
    parquet_path = os.path.join(generated_dir, "train.parquet")

    if not os.path.exists(parquet_path):
        print(f"Error: Intermediate parquet file not found at {parquet_path}.")
        return None

    train = pd.read_parquet(parquet_path)
    
    # Lọc dữ liệu lỗi
    original_len = len(train)
    train.dropna(subset=['rm_scores', 'probability'], inplace=True)
    train = train[train['rm_scores'].apply(lambda x: isinstance(x, list) and len(x) == args.pairs)]
    
    print(f"Filtered out {original_len - len(train)} invalid rows. {len(train)} valid rows remaining.")
    if len(train) == 0:
        print("Error: No valid data remaining after filtering. Cannot create final dataset.")
        return None

    metrics = train['rm_scores'].apply(lambda x: np.array(x))
    maxmin_indices = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    # --- FIX CHÍNH: "LÀM PHẲNG" DỮ LIỆU TỪ ĐỊNH DẠNG HỘI THOẠI SANG VĂN BẢN ---
    def flatten_conversation(conv_list):
        """Chuyển đổi một danh sách hội thoại thành một chuỗi văn bản duy nhất."""
        if not isinstance(conv_list, list) or len(conv_list) == 0:
            return "" # Trả về chuỗi rỗng nếu dữ liệu không hợp lệ
        # Giả định chúng ta chỉ cần nội dung của tin nhắn cuối cùng (của assistant)
        return conv_list[-1].get('content', '')

    chosen_text, rejected_text = [], []
    for i in range(len(train)):
        idx_pair = maxmin_indices.iloc[i]
        
        # Lấy cột generate_X tương ứng
        chosen_conversation = train[f"generate_{idx_pair[0]}"].iloc[i]
        rejected_conversation = train[f"generate_{idx_pair[1]}"].iloc[i]
        
        # Áp dụng hàm làm phẳng
        chosen_text.append(flatten_conversation(chosen_conversation))
        rejected_text.append(flatten_conversation(rejected_conversation))
        
    # Tạo DataFrame cuối cùng với các cột VĂN BẢN
    train_new = pd.DataFrame({'chosen': chosen_text, 'rejected': rejected_text})
    
    # Các cột khác (nếu cần) có thể được thêm vào đây
    # train_new['prompt'] = train['prompt'].apply(lambda x: x[0]['content']) # Ví dụ

    # Lưu dataset cuối cùng
    output_base_name = os.path.basename(os.path.normpath(base_output_dir))
    outpath = os.path.join('/kaggle/working', f'synthetic_data_{output_base_name}_score')
    os.makedirs(outpath, exist_ok=True)
    
    train_new.to_parquet(os.path.join(outpath, 'train.parquet'), index=False)
    print(f"Saved final training file to {os.path.join(outpath, 'train.parquet')}")
    
    test = train_new.sample(n=min(500, len(train_new)))
    test.to_parquet(os.path.join(outpath, 'test.parquet'), index=False)
    print(f"Saved final test file to {os.path.join(outpath, 'test.parquet')}")
    
    return outpath

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
