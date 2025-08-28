#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import yaml

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    SPPOConfig,
    H4ArgumentParser,
    ModelArguments,
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

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    - Tải dataset gốc (các cột: prompt, chosen, rejected có thể ở dạng list-of-dict).
    - Chuẩn hóa: đưa 'prompt', 'chosen', 'rejected' về string phẳng.
    """
    raw = get_datasets(data_args, splits=["train"])
    logger.info(
        "Training on splits: %s",
        [f"{k}: {v.num_rows}" for k, v in raw.items()],
    )

    def format_row(example):
        def _extract(x):
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
                return x[0].get("content", "")
            if isinstance(x, str):
                return x
            return ""
        return {
            "prompt": _extract(example.get("prompt", "")),
            "chosen": _extract(example.get("chosen", "")),
            "rejected": _extract(example.get("rejected", "")),
        }

    raw = raw.map(
        format_row,
        num_proc=data_args.preprocessing_num_workers,
        desc="Normalize raw fields",
    )
    return raw


def setup_model(model_args, training_args):
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
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

    # Adapter?
    model_id = model_args.model_name_or_path
    if is_adapter_model(model_id, model_args.model_revision):
        logger.info("Loading base+adapter for %s", model_id)
        peft_conf = PeftConfig.from_pretrained(model_id, revision=model_args.model_revision)
        base_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        if "gpt2" not in peft_conf.base_model_name_or_path.lower():
            base_kwargs["use_flash_attention_2"] = model_args.use_flash_attention_2

        base = AutoModelForCausalLM.from_pretrained(peft_conf.base_model_name_or_path, **base_kwargs)
        model = PeftModel.from_pretrained(base, model_id, revision=model_args.model_revision)
        ref_model = model
        model_kwargs = None
        ref_model_kwargs = None
    else:
        model = model_id  # để SPPOTrainer tự from_pretrained
        ref_model = model
        ref_model_kwargs = model_kwargs

    if model_args.use_peft:
        # Khi train LoRA, ref_model sẽ = None (DPO/ SPPO sẽ dùng implicit ref)
        ref_model = None
        ref_model_kwargs = None

    return model, ref_model, model_kwargs, ref_model_kwargs


def train_and_evaluate(trainer: SPPOTrainer, raw_datasets, training_args):
    checkpoint = None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")


def save_model_and_results(trainer: SPPOTrainer, training_args, model_args, data_args):
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)

    if trainer.accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    trainer.accelerator.wait_for_everyone()
    logger.info("*** Done! ***")


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPPOConfig))
    model_args, data_args, training_args = parser.parse()
    # ta chỉ train
    training_args.do_eval = False

    try:
        main_inner(model_args, data_args, training_args)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        raise


def main_inner(model_args, data_args, training_args):
    setup_logging(training_args.get_process_log_level())

    last_ckpt = get_checkpoint(training_args)
    if last_ckpt and training_args.resume_from_checkpoint is None:
        logger.info("Resuming from detected checkpoint: %s", last_ckpt)

    set_seed(training_args.seed)

    # tokenizer & raw data
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    # đảm bảo có pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw = load_and_process_datasets(data_args, tokenizer)

    # -------- Tokenize upfront (giống SPPOTrainer.tokenize_row) ----------
    def build_tokenized_answer(prompt: str, answer: str):
        full = tokenizer(prompt + answer, add_special_tokens=False)
        full_ids = full["input_ids"]
        full_attn = full["attention_mask"]
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        start = len(p_ids)
        if p_ids != full_ids[:start]:
            start -= 1

        prompt_ids = full_ids[:start]
        prompt_attn = full_attn[:start]
        ans_ids = full_ids[start:]
        ans_attn = full_attn[start:]
        return {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_attn,
            "input_ids": ans_ids,
            "attention_mask": ans_attn,
        }

    def tokenize_example(ex):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        pt = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in pt.items()}

        chosen_tokens = build_tokenized_answer(prompt, chosen)
        rejected_tokens = build_tokenized_answer(prompt, rejected)

        c_len = len(chosen_tokens["prompt_input_ids"])
        r_len = len(rejected_tokens["prompt_input_ids"])
        keep = min(c_len, r_len)
        for k in prompt_tokens:
            prompt_tokens[k] = prompt_tokens[k][:keep]

        # sanity (merge khác nhau chỉ ở cuối)
        diff_tok = sum(
            a != b
            for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])
        )
        if diff_tok > 1 or abs(c_len - r_len) > 1:
            # hiếm khi xảy ra, nhưng ta cứ fail rõ ràng
            raise ValueError(
                "chosen/rejected prompt tokens differ by more than 1 (merge issue)."
            )

        # BOS
        bos = tokenizer.bos_token_id
        for dic in (prompt_tokens, chosen_tokens, rejected_tokens):
            dic["prompt_input_ids"] = [bos] + dic["prompt_input_ids"]
            dic["prompt_attention_mask"] = [1] + dic["prompt_attention_mask"]

        # EOS vào answer
        eos = tokenizer.eos_token_id
        for dic in (chosen_tokens, rejected_tokens):
            dic["input_ids"].append(eos)
            dic["attention_mask"].append(1)

        max_len = training_args.max_length or 512
        max_prompt = training_args.max_prompt_length or 128
        longer = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # truncate prompt nếu cần
        for dic in (prompt_tokens, chosen_tokens, rejected_tokens):
            if len(dic["prompt_input_ids"]) + longer > max_len:
                if getattr(training_args, "truncation_mode", "keep_end") == "keep_start":
                    dic["prompt_input_ids"] = dic["prompt_input_ids"][:max_prompt]
                    dic["prompt_attention_mask"] = dic["prompt_attention_mask"][:max_prompt]
                else:
                    dic["prompt_input_ids"] = dic["prompt_input_ids"][-max_prompt:]
                    dic["prompt_attention_mask"] = dic["prompt_attention_mask"][-max_prompt:]

        # truncate answer nếu vẫn quá dài
        for dic in (chosen_tokens, rejected_tokens):
            if len(dic["prompt_input_ids"]) + longer > max_len:
                cut = max_len - max_prompt
                dic["input_ids"] = dic["input_ids"][:cut]
                dic["attention_mask"] = dic["attention_mask"][:cut]

        # build full + labels (mask prompt bằng -100)
        def _seq_from(dic):
            ids = dic["prompt_input_ids"] + dic["input_ids"]
            attn = dic["prompt_attention_mask"] + dic["attention_mask"]
            labels = ids[:]
            labels[: len(dic["prompt_input_ids"])] = [-100] * len(dic["prompt_input_ids"])
            return {"input_ids": ids, "attention_mask": attn, "labels": labels}

        chosen_seq = _seq_from(chosen_tokens)
        rejected_seq = _seq_from(rejected_tokens)

        out = {}
        for prefix, dic in (("chosen_", chosen_seq), ("rejected_", rejected_seq), ("", prompt_tokens)):
            for k, v in dic.items():
                if k == "token_type_ids":
                    continue
                out[f"{prefix}{k}"] = v
        return out

    tokenized_train = raw["train"].map(
        tokenize_example,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing dataset (upfront)",
    )

    # Quan trọng: luôn trả tensor cho DataLoader
    tokenized_train = tokenized_train.with_format(
        type="torch",
        columns=[
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
            "prompt_input_ids", "prompt_attention_mask",
        ],
    )

    # model & trainer
    model, ref_model, model_kwargs, ref_model_kwargs = setup_model(model_args, training_args)

    # bắt buộc để không drop field
    training_args.remove_unused_columns = False
    # tránh worker phụ pad lẫn lộn
    training_args.dataloader_num_workers = 0

    trainer = SPPOTrainer(
        model=model,
        ref_model=ref_model,
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
        data_collator=None,  # => dùng DPODataCollatorWithPadding mặc định
    )

    train_and_evaluate(trainer, raw, training_args)
    save_model_and_results(trainer, training_args, model_args, data_args)


if __name__ == "__main__":
    main()
