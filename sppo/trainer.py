#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

# TRL
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
)

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


class SPPOTrainer(Trainer):
    """
    A minimal Trainer that:
    - dùng DPODataCollatorWithPadding mặc định (pad & return dict tensor),
    - không override đường lấy mẫu lạ,
    - compute_loss theo SPPO / DPO family.
    """

    _tag_names = ["trl", "sppo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "sppo", "sppo_single", "rpo"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Any] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Any] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: str = None,
        ref_adapter_name: str = None,
    ):
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **(model_init_kwargs or {}))
        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **(ref_model_init_kwargs or {}))

        self._peft_has_been_casted_to_bf16 = False

        if tokenizer is None:
            raise ValueError("tokenizer must be provided")

        if max_length is None:
            warnings.warn("`max_length` defaulting to 512")
            max_length = 512
        if max_prompt_length is None:
            warnings.warn("`max_prompt_length` defaulting to 128")
            max_prompt_length = 128
        if max_target_length is None and (model is not None and model.config.is_encoder_decoder):
            warnings.warn("`max_target_length` defaulting to 128 for encoder-decoder")
            max_target_length = 128

        # default DPO collator (pad everything & return dict of tensors)
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=model.config.is_encoder_decoder if model is not None else bool(is_encoder_decoder),
            )
            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn("Force `remove_unused_columns=False` for DPO collator")

        # tắt dropout nếu muốn
        if disable_dropout and model is not None:
            disable_dropout_in_model(model)

        self.is_encoder_decoder = model.config.is_encoder_decoder if model is not None else bool(is_encoder_decoder)
        self.is_peft_model = _PEFT_AVAILABLE and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        if ref_model is not None:
            self.ref_model = ref_model
        else:
            # nếu không có ref_model thì tạo implicit ref từ model
            self.ref_model = create_reference_model(model) if model is not None else None

        self._tokenizer = tokenizer
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        self.truncation_mode = truncation_mode
        self.generate_during_eval = generate_during_eval
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # đảm bảo dataloader không có worker phụ (tránh pad lẫn)
        if args.dataloader_num_workers != 0:
            args.dataloader_num_workers = 0

        # init Trainer cha
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # DeepSpeed ref_model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    # --------- util ----------

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        if not is_deepspeed_available:
            return model
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        import deepspeed  # lazy
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict batch, got: {type(batch)}")
        if "chosen_input_ids" not in batch or "rejected_input_ids" not in batch:
            raise ValueError(
                f"Missing tokenized fields. Got keys: {list(batch.keys())}"
            )

        out = {}
        if is_encoder_decoder:
            max_len = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_len = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            if k.startswith("chosen"):
                if "labels" in k or is_encoder_decoder:
                    pad_val = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_val = padding_value
                elif k.endswith("_attention_mask"):
                    pad_val = 0
                else:
                    continue
                kk = k.replace("chosen", "concatenated")
                out[kk] = pad_to_length(batch[k], max_len, pad_value=pad_val)

        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            if k.startswith("rejected"):
                if "labels" in k or is_encoder_decoder:
                    pad_val = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_val = padding_value
                elif k.endswith("_attention_mask"):
                    pad_val = 0
                else:
                    continue
                kk = k.replace("rejected", "concatenated")
                out[kk] = torch.cat(
                    (out[kk], pad_to_length(batch[k], max_len, pad_value=pad_val)), dim=0
                ).to(device=device)

        if is_encoder_decoder:
            out["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            out["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device=device)

        return out

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must share (batch, seq) shape")
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        mask = labels != label_pad_token_id
        labels = labels.masked_fill(labels == label_pad_token_id, 0)
        tok_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (tok_logps * mask).sum(-1) / mask.sum(-1)
        return (tok_logps * mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        device = self.accelerator.device
        cat = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=device,
        )
        bs_chosen = batch["chosen_labels"].shape[0] if self.is_encoder_decoder else batch["chosen_input_ids"].shape[0]

        model_kwargs = (
            {"labels": cat["concatenated_labels"], "decoder_input_ids": cat.pop("concatenated_decoder_input_ids", None)}
            if self.is_encoder_decoder
            else {}
        )
        out = model(
            cat["concatenated_input_ids"],
            attention_mask=cat["concatenated_attention_mask"],
            **model_kwargs,
        )
        logits = out.logits
        all_logps = self.get_batch_logps(
            logits,
            cat["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        chosen_logps = all_logps[:bs_chosen]
        rejected_logps = all_logps[bs_chosen:]
        chosen_logits = logits[:bs_chosen]
        rejected_logits = logits[bs_chosen:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    # ---------- loss ----------

    def sppo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_probs: Union[torch.FloatTensor, None] = None,
        chosen_probs_win: Union[torch.FloatTensor, None] = None,
        chosen_probs_lose: Union[torch.FloatTensor, None] = None,
        reference_free: bool = False,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = (pi_logratios - ref_logratios).to(self.accelerator.device)

        logits_w = (policy_chosen_logps - reference_chosen_logps).to(self.accelerator.device)
        logits_l = (policy_rejected_logps - reference_rejected_logps).to(self.accelerator.device)

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(
                -self.beta * logits
            ) * self.label_smoothing
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type in ("sppo", "rpo"):
            if chosen_probs_win is None or chosen_probs_lose is None:
                # fallback an toàn
                chosen_probs_win = torch.full_like(logits_w, 0.5)
                chosen_probs_lose = torch.full_like(logits_l, 0.5)
            loss_w = (logits_w - (1 / self.beta) * (chosen_probs_win - 0.5)) ** 2
            loss_l = (logits_l - (1 / self.beta) * (chosen_probs_lose - 0.5)) ** 2
            losses = 0.5 * (loss_w + loss_l)
        elif self.loss_type == "sppo_single":
            cp = torch.full_like(logits_w, 0.5) if chosen_probs is None else chosen_probs
            loss_w = (logits_w - (1 / self.beta) * (cp - 0.5)) ** 2
            loss_l = (logits_l + (1 / self.beta) * (cp - 0.5)) ** 2
            losses = 0.5 * (loss_w + loss_l)
        elif self.loss_type == "kto_pair":
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = torch.cat(
                (
                    1 - torch.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - torch.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach().to(self.accelerator.device)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach().to(
            self.accelerator.device
        )
        return losses, chosen_rewards, rejected_rewards

    # ---------- train step ----------

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        (p_c_logps, p_r_logps, p_c_logits, p_r_logits) = self.concatenated_forward(model, batch)

        # optional probs
        bs = p_c_logps.shape[0]
        dev = p_c_logps.device
        cp = torch.full((bs,), 0.5, dtype=torch.float, device=dev)
        cpw = torch.full((bs,), 0.5, dtype=torch.float, device=dev)
        cpl = torch.full((bs,), 0.5, dtype=torch.float, device=dev)
        if "chosen_probs" in batch:
            cp = batch["chosen_probs"].to(dev) if isinstance(batch["chosen_probs"], torch.Tensor) else torch.as_tensor(batch["chosen_probs"], device=dev, dtype=torch.float)
        if "chosen_probs_win" in batch:
            cpw = batch["chosen_probs_win"].to(dev) if isinstance(batch["chosen_probs_win"], torch.Tensor) else torch.as_tensor(batch["chosen_probs_win"], device=dev, dtype=torch.float)
        if "chosen_probs_lose" in batch:
            cpl = batch["chosen_probs_lose"].to(dev) if isinstance(batch["chosen_probs_lose"], torch.Tensor) else torch.as_tensor(batch["chosen_probs_lose"], device=dev, dtype=torch.float)

        # ref logps
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            r_c_logps = batch["reference_chosen_logps"].to(dev)
            r_r_logps = batch["reference_rejected_logps"].to(dev)
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    r_c_logps, r_r_logps, _, _ = self.concatenated_forward(self.model, batch)
                else:
                    r_c_logps, r_r_logps, _, _ = self.concatenated_forward(self.ref_model, batch)

        losses, c_rewards, r_rewards = self.sppo_loss(
            p_c_logps, p_r_logps, r_c_logps, r_r_logps, cp, cpw, cpl
        )

        metrics = {
            ("eval_" if train_eval == "eval" else "") + "rewards/chosen": c_rewards.mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "rewards/rejected": r_rewards.mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "rewards/accuracies": (c_rewards > r_rewards).float().mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "rewards/margins": (c_rewards - r_rewards).mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "logps/chosen": p_c_logps.detach().mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "logps/rejected": p_r_logps.detach().mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "logits/chosen": p_c_logits.detach().mean().cpu(),
            ("eval_" if train_eval == "eval" else "") + "logits/rejected": p_r_logits.detach().mean().cpu(),
        }
        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        if not isinstance(inputs, dict):
            raise ValueError("Batch must be a dict (hint: tokenize upfront + DPO collator).")
        if "chosen_input_ids" not in inputs or "rejected_input_ids" not in inputs:
            raise ValueError(
                f"Missing tokenized fields in batch. Got keys: {list(inputs.keys())}"
            )
        with (torch.cuda.amp.autocast() if self._peft_has_been_casted_to_bf16 else torch.no_grad().__class__()):
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        # log ngay
        for k, v in metrics.items():
            self.log({k: v})
        return (loss, metrics) if return_outputs else loss

    # Không override get_train_dataloader / get_batch_samples
    # để tránh mọi đường dẫn có thể cung cấp raw string trước collate.

