import logging
import random
import sys
import yaml
import numpy as np  # Used for tokenization logic

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    SPPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from peft import PeftConfig, PeftModel
from trainer import SPPOTrainer
# Also import the module itself so we can see which trainer file is being used
import trainer as trainer_module
print("[DEBUG] Using trainer from:", trainer_module.__file__)

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def setup_logging(log_level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def load_and_process_datasets(data_args, tokenizer):
    """
    Hàm này tải dataset, trích xuất văn bản thô từ cấu trúc phức tạp,
    và trả về một dataset sẵn sàng cho SPPOTrainer.
    """
    # 1. Tải dữ liệu gốc
    raw_datasets = get_datasets(data_args, splits=["train"])
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    # 2. Định nghĩa hàm để trích xuất văn bản
    def format_row(feature):
        """
        Trích xuất văn bản từ cấu trúc danh sách và trả về một bản ghi mới.

        'chosen' và 'rejected' trong dữ liệu thô là list chứa một dict với khóa
        "content". Hàm này lấy nội dung đầu tiên và đảm bảo luôn trả về chuỗi.

        Chúng ta tránh sửa đổi 'feature' tại chỗ để loại bỏ bất kỳ tham chiếu
        Arrow/pyarrow nào còn sót lại; thay vào đó, trả về một dict mới.
        """
        prompt = feature.get("prompt", "")
        # Extract chosen content
        chosen_field = feature.get("chosen")
        if isinstance(chosen_field, list) and len(chosen_field) > 0:
            chosen = chosen_field[0].get("content", "")
        else:
            chosen = ""
        # Extract rejected content
        rejected_field = feature.get("rejected")
        if isinstance(rejected_field, list) and len(rejected_field) > 0:
            rejected = rejected_field[0].get("content", "")
        else:
            rejected = ""
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    # 3. Áp dụng hàm trích xuất
    raw_datasets = raw_datasets.map(
        format_row,
        num_proc=data_args.preprocessing_num_workers,
        desc="Formatting raw strings from STRUCT"
    )

    return raw_datasets

def setup_model(model_args, training_args):
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    if "gpt2" not in model_args.model_name_or_path.lower():
        model_kwargs["use_flash_attention_2"] = model_args.use_flash_attention_2

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        
        if "gpt2" not in peft_config.base_model_name_or_path.lower():
            model_kwargs["use_flash_attention_2"] = model_args.use_flash_attention_2
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft:
        ref_model = None
        ref_model_kwargs = None

    return model, ref_model, model_kwargs, ref_model_kwargs

def train_and_evaluate(trainer, raw_datasets, training_args):
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def save_model_and_results(trainer, training_args, model_args, data_args):
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    trainer.accelerator.wait_for_everyone()
    logger.info("*** Training complete! ***")

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPPOConfig))
    model_args, data_args, training_args = parser.parse()
    training_args.do_eval = False
    num_iteration = 1

    try:
        for i in range(num_iteration):
            main_inner(model_args, data_args, training_args)
            print(f"-------------------------Finished Iteration {i+1}---------------------------------")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def main_inner(model_args, data_args, training_args):
    setup_logging(training_args.get_process_log_level())

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    # Some models (e.g. GPT-2) do not have a pad token defined.  The
    # DPODataCollatorWithPadding relies on pad_token_id when constructing
    # batches, so we assign the end-of-sequence token as the pad token
    # if none is present.  Without this, collators may treat sequences of
    # different lengths as strings or lists, triggering downstream errors.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    raw_datasets = load_and_process_datasets(data_args, tokenizer)

    # -------------------------------------------------------------------------
    # Tokenize the dataset up front using custom logic adapted from
    # SPPOTrainer.tokenize_row. This avoids relying on internal mapping logic
    # of SPPOTrainer and ensures the training dataset already contains
    # `chosen_input_ids`, `rejected_input_ids`, etc. We handle truncation and
    # label masking similar to the original implementation.

    # Helper to build tokenized answer relative to a prompt.
    def build_tokenized_answer(prompt: str, answer: str):
        # Tokenize combined sequence without adding special tokens to examine merges
        full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
        full_input_ids = full_tokenized["input_ids"]
        full_attention = full_tokenized["attention_mask"]
        prompt_only_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Determine start index of answer tokens, adjusting for tokenizer merges
        resp_start = len(prompt_only_ids)
        # If the prompt tokens differ when tokenized together with answer, step back one token
        if prompt_only_ids != full_input_ids[:resp_start]:
            resp_start -= 1
        # Split into prompt and answer pieces
        prompt_ids = full_input_ids[:resp_start]
        prompt_attn = full_attention[:resp_start]
        ans_ids = full_input_ids[resp_start:]
        ans_attn = full_attention[resp_start:]
        return {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_attn,
            "input_ids": ans_ids,
            "attention_mask": ans_attn,
        }

    # Main tokenization function for each example
    def tokenize_example(example):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        # Tokenize prompt and answers
        _pt = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in _pt.items()}
        chosen_tokens = build_tokenized_answer(prompt, chosen)
        rejected_tokens = build_tokenized_answer(prompt, rejected)
        # Align prompt length across chosen and rejected answers
        chosen_len = len(chosen_tokens["prompt_input_ids"])
        rejected_len = len(rejected_tokens["prompt_input_ids"])
        prompt_len = min(chosen_len, rejected_len)
        for k in prompt_tokens:
            prompt_tokens[k] = prompt_tokens[k][:prompt_len]
        # Ensure only last token differs
        num_diff_tokens = sum(
            a != b
            for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])
        )
        num_diff_len = abs(chosen_len - rejected_len)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
            )
        # Add BOS token
        bos = tokenizer.bos_token_id
        prompt_tokens["prompt_input_ids"] = [bos] + prompt_tokens["prompt_input_ids"]
        chosen_tokens["prompt_input_ids"] = [bos] + chosen_tokens["prompt_input_ids"]
        rejected_tokens["prompt_input_ids"] = [bos] + rejected_tokens["prompt_input_ids"]
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
        # Add EOS token to answers
        eos = tokenizer.eos_token_id
        chosen_tokens["input_ids"].append(eos)
        chosen_tokens["attention_mask"].append(1)
        rejected_tokens["input_ids"].append(eos)
        rejected_tokens["attention_mask"].append(1)
        # Handle truncation if necessary
        max_len = training_args.max_length if training_args.max_length is not None else 512
        max_prompt = training_args.max_prompt_length if training_args.max_prompt_length is not None else 128
        trunc_mode = getattr(training_args, "truncation_mode", "keep_end")
        longer_resp_len = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
        for ans_tok in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(ans_tok["prompt_input_ids"]) + longer_resp_len > max_len:
                if trunc_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        ans_tok[k] = ans_tok[k][:max_prompt]
                elif trunc_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        ans_tok[k] = ans_tok[k][-max_prompt:]
                else:
                    raise ValueError(f"Unknown truncation mode: {trunc_mode}")
        for ans_tok in [chosen_tokens, rejected_tokens]:
            if len(ans_tok["prompt_input_ids"]) + longer_resp_len > max_len:
                for k in ["input_ids", "attention_mask"]:
                    ans_tok[k] = ans_tok[k][: max_len - max_prompt ]
        # Build full sequences and labels
        chosen_seq = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_seq = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        # Labels: mask prompt tokens with -100
        label_pad = -100
        chosen_seq["labels"] = list(chosen_seq["input_ids"])
        rejected_seq["labels"] = list(rejected_seq["input_ids"])
        chosen_prompt_len = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len = len(rejected_tokens["prompt_input_ids"])
        chosen_seq["labels"][:chosen_prompt_len] = [label_pad] * chosen_prompt_len
        rejected_seq["labels"][:rejected_prompt_len] = [label_pad] * rejected_prompt_len
        # Assemble final feature dict
        result = {}
        for prefix, toks in [("chosen_", chosen_seq), ("rejected_", rejected_seq), ("", prompt_tokens)]:
            for key, val in toks.items():
                if key == "token_type_ids":
                    continue
                result[f"{prefix}{key}"] = val
        return result

    # Tokenize entire training set
    tokenized_train = raw_datasets["train"].map(
        tokenize_example,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing dataset for SPPOTrainer"
    )

    # ---------------------------------------------------------------------
    # Ensure each feature is converted to a PyTorch tensor.  HuggingFace
    # datasets by default return Python lists for sequence features which can
    # sometimes be treated as strings by generic data collators.  Setting
    # the format to ``torch`` explicitly here ensures that every item in the
    # dataset yields a dictionary of torch tensors.  This, together with
    # the custom collator below, prevents the infamous "Batch is a string"
    # failure during training.
    tokenized_train = tokenized_train.with_format(
        type="torch",
        columns=[
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_labels",
            "prompt_input_ids",
            "prompt_attention_mask",
        ],
    )

    # Define a strict collator that refuses to operate on non-dictionary batches
    class DictOnlyCollator:
        """Collator that accepts only a list of dictionaries and stacks list values into tensors.

        If the incoming batch is not a list of dicts, this will raise an error to
        surface potential data formatting issues early.  It also converts Python
        lists into ``torch.Tensor`` on the fly when necessary.
        """

        def __call__(self, features):
            # Validate input batch structure
            if not isinstance(features, list) or not features:
                raise ValueError(
                    f"Collator expected a non-empty list of dicts, got {type(features)}"
                )
            if not isinstance(features[0], dict):
                raise ValueError(
                    f"Collator expected list elements to be dicts, got {type(features[0])}"
                )

            batch = {}
            keys = features[0].keys()
            for k in keys:
                # Collect all values for this key across the batch
                vals = [f[k] for f in features]
                # If already tensors, stack them
                if torch.is_tensor(vals[0]):
                    batch[k] = torch.stack(vals)
                # If list of ints, convert to tensor
                elif isinstance(vals[0], list):
                    batch[k] = torch.tensor(vals, dtype=torch.long)
                else:
                    # For any other type, keep as-is in a list
                    batch[k] = vals
            return batch

    data_collator = DictOnlyCollator()

    # Load the actual model and reference model (if any)
    model, ref_model, model_kwargs, ref_model_kwargs = setup_model(model_args, training_args)

    # Ensure we do not drop columns that are not model inputs.
    training_args.remove_unused_columns = False

    # Instantiate the SPPOTrainer with the tokenized dataset. Because the
    # dataset already contains tokenized fields, the internal `_map_dataset`
    # will skip further tokenization.
    trainer = SPPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        data_collator=data_collator,
    )

    # Use the tokenized dataset for training and also track the original raw
    # dataset for logging purposes (e.g., number of samples). We pass the
    # original `raw_datasets` into train_and_evaluate only for logging
    # `train_samples`, but training will use `tokenized_train` internally.
    train_and_evaluate(trainer, raw_datasets, training_args)
    save_model_and_results(trainer, training_args, model_args, data_args)

if __name__ == "__main__":
    main()
