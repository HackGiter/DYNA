# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

from torch.nn import CrossEntropyLoss

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

import math
# def _get_cosine_with_warmup_rate(
#     current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, min_rate: float = 0.1
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
#     factor = factor * (1 - min_rate) + min_rate
#     return max(0.1, factor)

# def _get_cosine_with_warmup_rate_with_init(
#     current_step: int, *, num_warmup_steps: int, num_training_steps: int, init: float = 0.5, num_cycles: float = 0.5, min_rate: float = 0.1
# ):

#     if current_step < num_warmup_steps:
#         return init + (1 - init) * float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
#     factor = factor * (1 - min_rate) + min_rate
#     return max(0.1, factor)

# def get_cosin_temperature_rate_with_plateau(
#     current_step: int, *, num_training_steps: int, plateau_steps:int = 0, init: float = 1.2, num_cycles: float = 0.5, min_rate: float = 1.0
# ):
#     if current_step + plateau_steps > num_training_steps:
#         return min_rate
#     progress = float(current_step - plateau_steps) / float(max(1, num_training_steps - plateau_steps))
#     factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
#     return (init - min_rate) * max(0.0, factor) + min_rate
def get_cosin_temperature_rate_with_plateau(
    current_step: int, *, num_training_steps: int, plateau_steps:int = 0, init: float = 1.2, num_cycles: float = 0.5, min_rate: float = 1.0
):
    if current_step + plateau_steps > num_training_steps:
        return min_rate
    progress = float(current_step) / float(max(1, num_training_steps - plateau_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    return (init - min_rate) * max(0.0, factor) + min_rate

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        # v0.1.0 0.2
        # v0.1.1
        # self.t_scale = kwargs.pop("t_scale", 0.1)
        # v0.2.0
        self.t_scale = kwargs.pop("t_scale", 0.1)
        self.p_tempt = kwargs.pop("p_tempt", 1.2)

        # v0.2.2
        # self.t_scale = kwargs.pop("t_scale", 0.1)
        # self.p_tempt = kwargs.pop("p_tempt", 0.9)
        # self.p_scale = kwargs.pop("p_scale", 2.5)

        # v0.2.4
        # self.t_scale = kwargs.pop("t_scale", 0.1)
        # self.p_tempt = kwargs.pop("p_tempt", 1.2)
        # self.p_scale = kwargs.pop("p_scale", 2.5)

        # self.p_scale = kwargs.pop("p_scale", 0.05)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels:torch.LongTensor = inputs.pop("labels")

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            logits:torch.Tensor = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            # v0.1.0
            # _, _, vocab_size = logits.size()
            # shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            # shift_labels = labels[..., 1:].contiguous().view(-1, 1).to(logits.device)
            # with torch.no_grad():
            #     index = shift_labels.detach().clamp(0)
            #     probs = shift_logits.detach().softmax(dim=-1)
            #     entpy = (probs * shift_logits.detach().log_softmax(dim=-1)).sum(dim=-1, keepdim=True)
            #     # detach_shift_logits = logits.detach()[..., :-1, :].contiguous().view(-1, vocab_size)
            #     # probs = detach_shift_logits.softmax(dim=-1)
            #     # entpy = (probs * detach_shift_logits.log_softmax(dim=-1)).sum(dim=-1, keepdim=True)
            #     nbase = torch.log(torch.tensor(1 / vocab_size).to(entpy.device))

            #     # denoising: gather small probabilities to concentrate
            #     # Accumulate the small probalities for next token prediction according to target domain
            #     p_thres = probs.max(dim=-1, keepdim=True).values * self.t_scale
            #     # probs[probs < p_thres] = 0
            #     probs = torch.where(probs < p_thres, torch.tensor(0.0, dtype=probs.dtype, device=probs.device), probs)
            #     probs.scatter_(dim=-1, index=index, value=0)

            #     # scaling: scale the rest probabilities according to next tokens entropy
            #     # high entropy means uncertainty which we intend to concentrate the path;
            #     # low entropy represent the experience learning from their instruction datasets which we want to maintain
            #     # probs = probs * (1.0 - entpy / nbase - p_scale )
            #     # v0.1.0
            #     # probs = probs * (1.0 - entpy / nbase - p_thres )
            #     probs = probs * (1.0 - entpy / nbase)
            #     p_next = 1.0 - probs.sum(dim=-1, keepdim=True)
            #     probs.scatter_(dim=-1, index=index, src=p_next)

            # v0.2.0
            _, _, vocab_size = logits.size()
            shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1, 1).to(shift_logits.device)
            with torch.no_grad():
                index = shift_labels.detach().clamp(0)
                probs:torch.Tensor = (shift_logits.detach() / self.p_tempt).softmax(dim=-1)
                entpy = (probs * (shift_logits.detach() / self.p_tempt).log_softmax(dim=-1)).sum(dim=-1, keepdim=True)
                nbase = torch.log(torch.tensor(1 / vocab_size).to(entpy.device))

                # denoising: gather small probabilities to concentrate
                # Accumulate the small probalities for next token prediction according to target domain
                p_thres = probs.max(dim=-1, keepdim=True).values * self.t_scale
                probs = torch.where(probs < p_thres, torch.tensor(0.0, dtype=probs.dtype, device=probs.device), probs)
                p_cur = probs.gather(dim=-1, index=index)

                # V20 EXTRA
                shift_labels = torch.where(
                    p_cur > 0.9,
                    IGNORE_INDEX,
                    shift_labels.detach()
                )

                # scaling: scale the rest probabilities according to next tokens entropy
                # high entropy means uncertainty which we intend to concentrate the path;
                # low entropy represent the experience learning from their instruction datasets which we want to maintain
                probs = probs * (1.0 - entpy / nbase)
                probs.scatter_(dim=-1, index=index, src=p_cur + 0.9)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # v0.2.3
            # _, _, vocab_size = logits.size()
            # shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            # shift_labels = labels[..., 1:].contiguous().view(-1, 1).to(shift_logits.device)
            # # ratio = _get_cosine_with_warmup_rate(
            # #     current_step=self.state.global_step,
            # #     num_warmup_steps=self.args.get_warmup_steps(self.state.max_steps) * 0.5,
            # #     num_training_steps=self.state.max_steps,
            # #     )
            # ratio = _get_cosine_with_warmup_rate_with_init(
            #     current_step=self.state.global_step,
            #     num_warmup_steps=self.args.get_warmup_steps(self.state.max_steps) * 2.0,
            #     num_training_steps=self.state.max_steps,
            #     init=0.5,
            #     min_rate=0.2,
            #     )
            # with torch.no_grad():
            #     index = shift_labels.detach().clamp(0)
            #     probs:torch.Tensor = (shift_logits.detach()).softmax(dim=-1)
            #     entpy = (probs * (shift_logits.detach()).log_softmax(dim=-1)).sum(dim=-1, keepdim=True)
            #     nbase = torch.log(torch.tensor(1 / vocab_size).to(entpy.device))
            #     p_scale = entpy / nbase
            #     p_tempt = torch.pow(torch.tensor(self.p_tempt).to(entpy.device), p_scale)

            #     probs = (shift_logits.detach() / p_tempt).softmax(dim=-1)

            #     # denoising: gather small probabilities to concentrate
            #     # Accumulate the small probalities for next token prediction according to target domain
            #     p_thres = probs.max(dim=-1, keepdim=True).values * self.t_scale
            #     probs = torch.where(probs < p_thres,
            #                         torch.tensor(0.0, dtype=probs.dtype, device=probs.device),
            #                         probs)
            #     probs = probs / probs.sum(dim=-1, keepdim=True)

            #     # filter
            #     p_cur = probs.gather(dim=-1, index=index)
            #     shift_labels = torch.where(
            #         p_cur > 0.8,
            #         torch.tensor(IGNORE_INDEX, dtype=shift_labels.dtype, device=shift_labels.device),
            #         shift_labels.detach()
            #     )

            #     # scaling: scale the rest probabilities according to next tokens entropy
            #     # high entropy means uncertainty which we intend to concentrate the path;
            #     # low entropy represent the experience learning from their instruction datasets which we want to maintain
            #     p_next = (1 - ratio) * p_cur + ratio * torch.sigmoid((1 - p_scale) * self.p_scale)
            #     probs.scatter_(dim=-1, index=index, src=p_next)
            #     probs = probs / probs.sum(dim=-1, keepdim=True)

            # v0.2.4
            # _, _, vocab_size = logits.size()
            # shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            # shift_labels = labels[..., 1:].contiguous().view(-1, 1).to(shift_logits.device)
            # ratio = _get_cosine_with_warmup_rate(
            #     current_step=self.state.global_step,
            #     num_warmup_steps=self.args.get_warmup_steps(self.state.max_steps) * 0.5,
            #     num_training_steps=self.state.max_steps,
            #     )
            # ratio = _get_cosine_with_warmup_rate_with_init(
            #     current_step=self.state.global_step,
            #     num_warmup_steps=self.args.get_warmup_steps(self.state.max_steps) * 1.5,
            #     num_training_steps=self.state.max_steps,
            #     init=0.5,
            #     min_rate=0.2,
            #     )
            # p_tempt = get_cosin_temperature_rate_with_plateau(
            #     current_step=self.state.global_step,
            #     num_training_steps=self.state.max_steps,
            #     plateau_steps=int(self.state.max_steps * 0.3),
            #     init=self.p_tempt,
            #     min_rate=0.95,
            # )
            # with torch.no_grad():
            #     index = shift_labels.detach().clamp(0)
            #     probs:torch.Tensor = (shift_logits.detach()).softmax(dim=-1)
            #     p_cur = probs.gather(dim=-1, index=index)
            #     probs.scatter_(-1, index, 0.0)
            #     probs = probs / probs.sum(dim=-1, keepdim=True)

            #     entpy = (probs * (shift_logits.detach()).log_softmax(dim=-1)).sum(dim=-1, keepdim=True)
            #     nbase = torch.log(torch.tensor(1 / math.sqrt(vocab_size)).to(entpy.device))
            #     p_scale = entpy / nbase
            #     p_tempt = torch.pow(torch.tensor(p_tempt).to(entpy.device), (p_scale if p_tempt > 1.0 else 1.0 - p_scale))

            #     probs = (shift_logits.detach() / p_tempt).softmax(dim=-1)

            #     # denoising: gather small probabilities to concentrate
            #     # Accumulate the small probalities for next token prediction according to target domain
            #     p_thres = probs.max(dim=-1, keepdim=True).values * self.t_scale
            #     probs = torch.where(probs < p_thres,
            #                         torch.tensor(0.0, dtype=probs.dtype, device=probs.device),
            #                         probs)
            #     probs = probs / probs.sum(dim=-1, keepdim=True)

            #     # filter
            #     shift_labels = torch.where(
            #         p_cur > 0.85,
            #         torch.tensor(IGNORE_INDEX, dtype=shift_labels.dtype, device=shift_labels.device),
            #         shift_labels.detach()
            #     )

            #     # scaling: scale the rest probabilities according to next tokens entropy
            #     # high entropy means uncertainty which we intend to concentrate the path;
            #     # low entropy represent the experience learning from their instruction datasets which we want to maintain
            #     p_next = (1 - ratio) * p_cur + ratio * torch.sigmoid((0.5 - p_scale) * self.p_scale * 2)
            #     probs.scatter_(dim=-1, index=index, src=p_next)
            #     probs = probs / probs.sum(dim=-1, keepdim=True)

            # Enable model parallelism
            shift_masks = shift_labels.view(-1).eq(IGNORE_INDEX).to(shift_logits.device)
            shift_probs = probs.contiguous().to(shift_logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss:torch.Tensor = loss_fct(shift_logits, shift_probs)
            loss.masked_fill_(shift_masks, 0.0)
            loss = loss.sum() / (shift_masks.numel() - shift_masks.long().sum())
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
