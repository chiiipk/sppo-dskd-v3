#!/usr/bin/env python
#
# Adapted from https://github.com/huggingface/alignment-handbook

import logging
import random
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
def preprocess_function(feature, tokenizer, max_length, max_prompt_length):
    """
    Hàm này áp dụng template và token hóa cho một hàng dữ liệu.
    Nó trả về một dictionary chứa các tensor đã được token hóa.
    """
    # 1. Áp dụng template để tạo chuỗi văn bản hoàn chỉnh
    prompt_messages = [{"role": "user", "content": feature["prompt"]}]
    chosen_messages = feature["chosen"] # Dữ liệu đã là một list of dicts
    rejected_messages = feature["rejected"] # Dữ liệu đã là một list of dicts

    prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    chosen_str = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    rejected_str = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    
    # 2. Token hóa các chuỗi đã định dạng
    tokenized_prompt = tokenizer(prompt_str, truncation=True, max_length=max_prompt_length)
    tokenized_chosen = tokenizer(chosen_str, truncation=True, max_length=max_length)
    tokenized_rejected = tokenizer(rejected_str, truncation=True, max_length=max_length)

    # 3. Tạo dictionary kết quả
    result = {
        "prompt_input_ids": tokenized_prompt["input_ids"],
        "prompt_attention_mask": tokenized_prompt["attention_mask"],
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        "chosen_labels": tokenized_chosen["input_ids"][:], # Sao chép input_ids sang labels
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
        "rejected_labels": tokenized_rejected["input_ids"][:] # Sao chép input_ids sang labels
    }
    return result
    
def load_and_process_datasets(data_args, training_args, tokenizer):
    # 1. Tải dữ liệu gốc
    raw_datasets = get_datasets(data_args, splits=["train"])
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    
    # 2. Lấy tên các cột gốc để xóa chúng đi sau khi xử lý xong
    original_column_names = list(raw_datasets["train"].features)
    
    # 3. Áp dụng hàm preprocess_function lên toàn bộ dataset
    raw_datasets = raw_datasets.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": training_args.max_length,
            "max_prompt_length": training_args.max_prompt_length,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=original_column_names, # Xóa tất cả các cột cũ
        desc="Tokenizing and formatting comparisons",
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
    tokenized_datasets = load_and_process_datasets(data_args, training_args, tokenizer)
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False # Giả sử mô hình của bạn là decoder-only
    )
    model, ref_model, model_kwargs, ref_model_kwargs = setup_model(model_args, training_args)

    trainer = SPPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    train_and_evaluate(trainer, tokenized_datasets, training_args)
    save_model_and_results(trainer, training_args, model_args, data_args)

if __name__ == "__main__":
    main()
