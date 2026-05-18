import math
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.utils import add_start_docstrings, ModelOutput, logging
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ...model.loader import load_model

logger = logging.get_logger(__name__)

@dataclass
class rWCLSOutput(ModelOutput):
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments

rWCLS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`rWCLSConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
import os
class rWCLSConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLSModel`]. It is used to instantiate an CLS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLS-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the CLS model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CLSModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    ```python
    >>> from transformers import CLSModel, CLSConfig

    >>> # Initializing a CLS CLS-7b style configuration
    >>> configuration = rWCLSConfig()

    >>> # Initializing a model from the CLS-7b style configuration
    >>> model = rWCLSModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rwcls"


    def __init__(
        self,
        hidden_size:int=4096,
        num_labels:int=3,
        num_tasks:int=8,
        label_block:int=768,
        task_block:int=384,
        classify_path:str=None,
        initializer_range:float=0.02,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_labels=num_labels
        self.classify_path = classify_path
        self.num_tasks = num_tasks
        self.label_block = label_block
        self.task_block = task_block
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

@add_start_docstrings(
    "The bare rWCLS Model outputting raw hidden-states without any specific head on top.",
    rWCLS_START_DOCSTRING,
)
class rWCLSPreTrainedModel(PreTrainedModel):
    config_class = rWCLSConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class rWCLS(rWCLSPreTrainedModel):
    def __init__(
            self,
            config: rWCLSConfig,
            tokenizer: "PreTrainedTokenizer",
            model_args: "ModelArguments",
            finetuning_args: "FinetuningArguments",
            add_valuehead: bool = False,
            **kwargs
            ):
        super().__init__(config)
        self.model = load_model(tokenizer, model_args, finetuning_args, True, add_valuehead)
        self.model.train()
        self.model.requires_grad_(True)
        # self.model.return_hidden_states = True
        # self.task_score = nn.Linear(config.hidden_size,
        #                             config.num_tasks * config.task_block, bias=False,
        #                             device="cpu",
        #                             dtype=self.model.dtype)
        # self.label_score = nn.Linear(config.hidden_size,
        #                              config.num_labels * config.label_block, bias=False,
        #                              device="cpu",
        #                              dtype=self.model.dtype,)

        # self.post_init()

        # from safetensors.torch import load_file
        # safetensors = sorted([item for item in os.listdir(self.config.classify_path) if "safetensors" in item])
        # safetensors = load_file(os.path.join(self.config.classify_path, safetensors[1]))

        # if hasattr(config, "num_block"):
        #     self.label_block = config.num_block
        #     self.task_block = config.num_block
        # else:
        #     self.label_block = config.label_block
        #     self.task_block = config.task_block
        # self.num_labels = config.num_labels
        # self.num_tasks = config.num_tasks
        # self.task_score.load_state_dict(
        #     {'weight':safetensors['task_score.weight']}
        # )
        # self.label_score.load_state_dict(
        #     {'weight':safetensors['label_score.weight']}
        # )

        # self.task_score.to(self.model.device)
        # self.label_score.to(self.model.device)

        # self.task_score.requires_grad_(False)
        # self.label_score.requires_grad_(False)
        import numpy as np
        import scipy.stats as st
        self.window_size = 5
        self.mean = 0
        self.std = 1.0
        self.conv_kernel = st.norm.pdf(np.linspace(-(self.window_size // 2), self.window_size // 2, self.window_size), self.mean, self.std)
        self.conv_kernel = nn.Parameter(
            torch.tensor(self.conv_kernel / np.sum(self.conv_kernel)).unsqueeze_(0).unsqueeze_(0),
            requires_grad=False)
        self.ignore_index = -100
        self.cnt = 0
        self.global_numel = 0
        self.global_tokens = []

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        # type_labels: Optional[torch.Tensor] = None,
        # task_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Dict, Tuple, torch.Tensor, rWCLSOutput]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # def norm(x:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        #     dtype = x.dtype
        #     x = x.float()
        #     x = x - x.mean(-1, keepdim=True)
        #     x = x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        #     return x.to(dtype)

        # def proc(x:torch.Tensor, m:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        #     dtype = x.dtype
        #     x = x.float()
        #     x = x.sum(dim=1) / (m.sum(dim=1, keepdim=True) + eps)
        #     return x.to(dtype)

        shift_logits:torch.Tensor = outputs[0][..., :-1, :].contiguous()
        shift_labels:torch.Tensor = labels[..., 1:].contiguous()
        # shift_hidden_states:torch.Tensor = outputs[1][..., :-1, :]
        # shift_label_masks:torch.Tensor = labels[..., 1:].not_equal(self.ignore_index)

        # shift_hidden_states = norm(shift_hidden_states)
        # shift_hidden_states = shift_hidden_states * shift_label_masks.unsqueeze(-1)
        # shift_hidden_states = proc(shift_hidden_states, shift_label_masks)

        # label_scores = self.label_score(shift_hidden_states).view(-1, self.num_labels).abs().float()
        # task_scores = self.task_score(shift_hidden_states).view(-1, self.num_tasks).abs().float()

        # with torch.no_grad():
        #     label_masks = type_labels.not_equal(self.num_labels - 1).unsqueeze_(1).repeat(1, shift_label_masks.shape[1]).logical_and_(
        #         task_labels.not_equal(self.num_tasks - 1).unsqueeze_(1).repeat(1, shift_label_masks.shape[1])
        #     )
        #     shift_labels = shift_labels.masked_fill_(label_masks, self.ignore_index).contiguous()
        #     shift_labels = shift_labels.masked_fill_(
        #         shift_logits.detach().argmax(dim=-1) == shift_labels, self.ignore_index
        #     ).contiguous()

        #     shift_label_weights = shift_labels.not_equal(
        #         self.ignore_index).sum(dim=-1, keepdim=True).repeat(1, shift_labels.shape[1])
        #     shift_label_weights = shift_label_weights.masked_fill_(
        #         shift_labels.eq(self.ignore_index), 0).contiguous().view(-1)
        #     shift_label_weights = shift_label_weights.float() / (shift_label_weights.sum() + 1e-8)

        #     type_labels = type_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, self.ignore_index
        #                                            ).repeat_interleave(self.label_block, 0)
        #     task_labels = task_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, self.ignore_index
        #                                            ).repeat_interleave(self.task_block, 0)

        #     type_labels = type_labels.masked_fill_(label_scores.detach().argmax(dim=-1) == type_labels,
        #                                            self.ignore_index)
        #     task_labels = task_labels.masked_fill_(task_scores.detach().argmax(dim=-1) == task_labels,
        #                                            self.ignore_index)

        # loss_fct_n = nn.CrossEntropyLoss()
        loss_fct_m = nn.CrossEntropyLoss(reduction="none")

        # label_loss = loss_fct_n(label_scores, type_labels) if type_labels.not_equal(self.ignore_index).sum() > 0 else torch.tensor(0.0, dtype=shift_logits.dtype, device=shift_logits.device)
        # task_loss = loss_fct_n(task_scores, task_labels) if task_labels.not_equal(self.ignore_index).sum() > 0 else torch.tensor(0.0, dtype=shift_logits.dtype, device=shift_logits.device)
        loss:torch.Tensor = loss_fct_m(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        # loss = (loss * shift_label_weights).sum()

        with torch.no_grad():
            numel = shift_labels.not_equal(self.ignore_index).sum()
            self.global_numel = (numel + self.global_numel * self.cnt) / (self.cnt + 1)
            self.cnt += 1
            numel = int(numel * 0.05)
            # masked_loss = loss.detach().masked_fill(shift_labels.view(-1).eq(self.ignore_index), 1)
            masked_loss = F.conv1d(loss.detach().unsqueeze(0), self.conv_kernel, padding='same', groups=1).squeeze(0)
            _, top_indices = torch.topk(masked_loss, numel)
            global_top_values, _ = torch.topk(torch.concat(self.global_tokens + [masked_loss], dim=-1), self.global_numel * 0.1)
            self.global_tokens = [global_top_values]
            # _, bottom_indices = torch.topk(-masked_loss, numel // 2)
            token_masks = torch.ones_like(loss, requires_grad=False)
            # token_masks.scatter_(-1, top_indices, top_values)
            # token_masks.scatter_(-1, top_indices[:numel//2], loss.detach()[top_indices[:numel//2]])
            # token_masks.scatter_(-1, top_indices[numel//2:], top_values[numel//2:])
            token_masks.scatter_(-1, top_indices, loss.detach()[top_indices])
            token_masks.masked_fill_(masked_loss < self.global_tokens[-1], 1.0)
            token_masks.masked_fill_(loss.detach() < 1.0, 1.0)
            # token_masks.scatter_(-1, bottom_indices, 1)
            # shift_labels = shift_labels.view(-1).masked_fill(token_masks, self.ignore_index)

        # loss.masked_fill_(token_masks, 0.0)
        loss /= token_masks
        loss = loss.sum() / (shift_labels.numel() - shift_labels.eq(self.ignore_index).sum())
        # loss = 0.95 * loss + 0.05 * ( label_loss + task_loss )

        return (loss, )
