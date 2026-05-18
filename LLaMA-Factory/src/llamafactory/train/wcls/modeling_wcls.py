import math
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.utils import add_start_docstrings, ModelOutput, logging
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ...model.loader import (
    register_autoclass,
    init_adapter,
    load_model,
    patch_model,
)

logger = logging.get_logger(__name__)

@dataclass
class WCLSOutput(ModelOutput):
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments

WCLS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WCLSConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
import os
class WCLSConfig(PretrainedConfig):
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
    >>> configuration = WCLSConfig()

    >>> # Initializing a model from the CLS-7b style configuration
    >>> model = WCLSModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "wcls"

    def __init__(
        self,
        hidden_size:int=4096,
        num_labels:int=3,
        num_tasks:int=8,
        # num_block:int=4,
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
        # self.num_block = num_block
        self.label_block = label_block
        self.task_block = task_block
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

@add_start_docstrings(
    "The bare WCLS Model outputting raw hidden-states without any specific head on top.",
    WCLS_START_DOCSTRING,
)
class WCLSPreTrainedModel(PreTrainedModel):
    config_class = WCLSConfig
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

class WCLS(WCLSPreTrainedModel):
    def __init__(
            self,
            config: WCLSConfig,
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
        self.model.return_hidden_states = True
        self.task_score = nn.Linear(config.hidden_size,
                                    config.num_tasks * config.task_block, bias=False,
                                    device="cpu",
                                    dtype=self.model.dtype)
        self.label_score = nn.Linear(config.hidden_size,
                                     config.num_labels * config.label_block, bias=False,
                                     device="cpu",
                                     dtype=self.model.dtype,)

        self.post_init()

        from safetensors.torch import load_file
        safetensors = sorted([item for item in os.listdir(self.config.classify_path) if "safetensors" in item])
        safetensors = load_file(os.path.join(self.config.classify_path, safetensors[1]))

        # self.num_block = config.num_block
        if hasattr(config, "num_block"):
            self.label_block = config.num_block
            self.task_block = config.num_block
        else:
            self.label_block = config.label_block
            self.task_block = config.task_block
        self.num_labels = config.num_labels
        self.num_tasks = config.num_tasks
        self.task_score.load_state_dict(
            {'weight':safetensors['task_score.weight']}
        )
        self.label_score.load_state_dict(
            {'weight':safetensors['label_score.weight']}
        )

        self.task_score.to(self.model.device)
        self.label_score.to(self.model.device)

        self.task_score.requires_grad_(False)
        self.label_score.requires_grad_(False)

        self.ignore_index = -100

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
    ) -> Union[Dict, Tuple, torch.Tensor, WCLSOutput]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[1]

        def norm(x:torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            x = x.float()
            x = x - x.mean(-1, keepdim=True)
            x = x * (x.pow(2).mean(-1, keepdim=True) + 1e-8).rsqrt()
            return x.to(dtype)

        def proc(x:torch.Tensor, m:torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            x = x.float()
            x = x.sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-8)
            return x.to(dtype)

        shift_hidden_states = hidden_states[..., :-1, :]
        shift_label_masks = labels[..., 1:].not_equal(self.ignore_index)

        shift_hidden_states = norm(shift_hidden_states)
        shift_hidden_states = shift_hidden_states * shift_label_masks.unsqueeze(-1)
        shift_hidden_states = proc(shift_hidden_states, shift_label_masks)

        label_scores = self.label_score(shift_hidden_states).view(-1, self.num_labels).abs().float()
        task_scores = self.task_score(shift_hidden_states).view(-1, self.num_tasks).abs().float()

        with torch.no_grad():
            type_labels = torch.full([shift_label_masks.shape[0]],
                                     fill_value=self.num_labels - 1,
                                     dtype=labels.dtype,
                                     device=labels.device)
            task_labels = torch.full([shift_label_masks.shape[0]],
                                     fill_value=self.num_tasks - 1,
                                     dtype=labels.dtype,
                                     device=labels.device)

            type_labels = type_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, self.ignore_index
                                                   ).repeat_interleave(self.label_block, 0)
            task_labels = task_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, self.ignore_index
                                                   ).repeat_interleave(self.task_block, 0)

            type_labels = type_labels.masked_fill_(label_scores.detach().argmax(dim=-1) == type_labels,
                                                   self.ignore_index)
            task_labels = task_labels.masked_fill_(task_scores.detach().argmax(dim=-1) == task_labels,
                                                   self.ignore_index)

        loss_fct = nn.CrossEntropyLoss()
        label_loss = loss_fct(label_scores, type_labels) if type_labels.not_equal(self.ignore_index).sum() > 0 else 0
        task_loss = loss_fct(task_scores, task_labels) if task_labels.not_equal(self.ignore_index).sum() > 0 else 0
        loss = 0.91 * outputs[0] + 0.09 * ( label_loss + task_loss )

        return (loss, )