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
        # num_labels:int=3,
        # num_tasks:int=8,
        # label_block:int=768,
        # task_block:int=384,
        # classify_path:str=None,
        local_window_size:int=5,
        initializer_range:float=0.02,
        mean:float=0.0,
        std:float=1.0,
        epoch: int=1737,
        local_numel_ratio:float=0.05,
        global_numel_ratio:float=0.25,
        eos: int=128009,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # self.num_labels=num_labels
        # self.classify_path = classify_path
        # self.num_tasks = num_tasks
        # self.label_block = label_block
        # self.task_block = task_block
        self.local_window_size = local_window_size
        self.mean = mean
        self.std = std
        self.epoch = epoch
        self.local_numel_ratio = local_numel_ratio
        self.global_numel_ratio = global_numel_ratio
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.eos = eos

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
        import numpy as np
        import scipy.stats as st
        # self.window_size = 5
        # self.mean = 0
        # self.std = 1.0
        self.config = config
        self.window_size = config.local_window_size
        self.mean = config.mean
        self.std = config.std
        self.conv_kernel = st.norm.pdf(np.linspace(-(self.window_size // 2), self.window_size // 2, self.window_size), self.mean, self.std)
        self.conv_kernel = nn.Parameter(
            torch.tensor(self.conv_kernel / np.sum(self.conv_kernel)).unsqueeze_(0).unsqueeze_(0),
            requires_grad=False)
        self.ignore_index = -100
        self.cnt = 0
        self.global_numel = 0
        self.global_top_tokens = torch.empty(0)
        # self.epoch = 1237
        self.epoch = config.epoch
        self.local_numel_ratio = config.local_numel_ratio
        self.global_numel_ratio = config.global_numel_ratio

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
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

        shift_logits:torch.Tensor = outputs[0][..., :-1, :].contiguous()
        shift_labels:torch.Tensor = labels[..., 1:].contiguous()

        loss_fct_m = nn.CrossEntropyLoss(reduction="none")

        loss:torch.Tensor = loss_fct_m(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        # 0.05 0.05
        # 0.08 1.0
        # 0.08 0.5
        # 0.08 0.25
        with torch.no_grad():
            numel = shift_labels.not_equal(self.ignore_index).sum()
            # self.global_numel = (numel + self.global_numel * self.cnt) / (self.cnt + 1)
            # self.cnt += 1

            # numel = int(numel * 0.08)
            numel = int(numel * self.local_numel_ratio)
            local_conv_loss:torch.Tensor = F.conv1d(loss.detach().unsqueeze(0),
                                   self.conv_kernel, padding='same', groups=1).squeeze(0)

            local_conv_loss.masked_fill_(shift_labels.view(-1).eq(self.config.eos), -1)
            local_top_values, local_top_indices = torch.topk(local_conv_loss, numel)
            # self.global_top_tokens, _ = torch.topk(torch.concat([self.global_top_tokens.to(local_conv_loss.device),
            #                                                      local_conv_loss], dim=-1),
            #                                                      int(self.global_numel * 0.25))

            # self.global_top_tokens, _ = torch.topk(torch.concat([self.global_top_tokens.to(local_conv_loss.device),
            #                                                      local_conv_loss], dim=-1),
            #                                                      int(self.global_numel * self.global_numel_ratio))
            token_masks = torch.ones_like(loss, requires_grad=False)

            local_detach_loss = loss.detach()[local_top_indices]
            # token_masks.scatter_(-1, local_top_indices, local_detach_loss)
            # token_masks.masked_fill_(local_conv_loss < self.global_top_tokens[-1], 1.0)

            # token_masks.scatter_(-1,
            #                      local_top_indices[local_top_values < self.global_top_tokens[-1]],
            #                      local_top_values[local_top_values < self.global_top_tokens[-1]])
            # local_thres_indices = local_top_values < self.global_top_tokens[-1]
            # token_masks.scatter_(-1,
            #                     local_top_indices[local_thres_indices],
            #                     local_detach_loss[local_thres_indices] / local_top_values[local_thres_indices],
            #                     )
            token_masks.scatter_(-1,
                                local_top_indices,
                                local_detach_loss / local_top_values,
                                )
            # token_masks.masked_fill_(loss.detach() < 1.0, 1.0)
            token_masks.masked_fill_(token_masks < 1.0, 1.0)
            # self.cnt = self.cnt % self.epoch
            # self.global_top_tokens = (self.cnt > 0) * self.global_top_tokens

        loss /= token_masks
        loss = loss.sum() / (shift_labels.numel() - shift_labels.eq(self.ignore_index).sum())

        return (loss, )
