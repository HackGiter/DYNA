import math
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.utils import add_start_docstrings, ModelOutput, logging
from transformers.modeling_utils import PreTrainedModel
from transformers import LlamaModel
from transformers.configuration_utils import PretrainedConfig

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ...model import load_config
from ...model.loader import (
    _get_init_kwargs,
    patch_config,
    register_autoclass,
    init_adapter,
)

logger = logging.get_logger(__name__)

@dataclass
class CLSOutput(ModelOutput):
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None
    activations: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments

CLS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CLSConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

class CLSConfig(PretrainedConfig):
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
    >>> configuration = CLSConfig()

    >>> # Initializing a model from the CLS-7b style configuration
    >>> model = CLSModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cls"

    def __init__(
        self,
        hidden_size:int=4096,
        num_labels:int=3,
        num_tasks:int=8,
        # num_block:int=4,
        block_orth:float=0.01,
        label_block:int=768,
        task_block:int=384,
        label_smooth:float=0.85,
        task_smooth:float=0.95,
        initializer_range:float=0.02,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_labels=num_labels
        self.num_tasks = num_tasks
        # self.num_block = num_block
        self.block_orth = block_orth
        self.label_block = label_block
        self.task_block = task_block
        self.label_smooth = label_smooth
        self.task_smooth = task_smooth
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

@add_start_docstrings(
    "The bare CLS Model outputting raw hidden-states without any specific head on top.",
    CLS_START_DOCSTRING,
)
class CLSPreTrainedModel(PreTrainedModel):
    config_class = CLSConfig
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

def orthogonal(x:torch.Tensor, y:torch.Tensor):
    self_mm = torch.matmul(x, x.transpose(-1, -2))
    amplitude = x.float().pow(2).sum(-1).rsqrt()
    self_mm = self_mm.float() * amplitude.unsqueeze(1) * amplitude.unsqueeze(0)
    self_mm = self_mm * (1.0 - y)
    return self_mm.pow(2).sum()

class CLS(CLSPreTrainedModel):
    def __init__(
            self,
            config: CLSConfig,
            tokenizer: "PreTrainedTokenizer",
            model_args: "ModelArguments",
            finetuning_args: "FinetuningArguments",
            add_valuehead: bool = False,
            **kwargs
            ):
        super().__init__(config)
        init_kwargs = _get_init_kwargs(model_args)
        self.model_config = load_config(model_args)
        patch_config(self.model_config, tokenizer, model_args, init_kwargs, False)
        init_kwargs['config'] = self.model_config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        # self.model = load_model(tokenizer, model_args, finetuning_args, False, add_valuehead)
        self.model = LlamaModel.from_pretrained(**init_kwargs)
        # patch_model(self.model, tokenizer, model_args, False, add_valuehead)
        register_autoclass(self.model_config, self.model, tokenizer)
        self.model = init_adapter(self.model_config, self.model, model_args, finetuning_args, False)
        self.model.requires_grad_(False)
        self.model.eval()
        self.num_tasks = config.num_tasks
        self.num_labels = config.num_labels
        # self.num_block = config.num_block
        if hasattr(config, "num_block"):
            self.label_block = config.num_block
            self.task_block = config.num_block
        else:
            self.label_block = config.label_block
            self.task_block = config.task_block
        self.block_orth = config.block_orth
        self.task_score = nn.Linear(config.hidden_size,
                                    config.num_tasks * self.task_block,
                                    bias=False,
                                    device=self.model.device,
                                    dtype=self.model.dtype)
        self.label_score = nn.Linear(config.hidden_size,
                                     config.num_labels * self.label_block,
                                     bias=False,
                                     device=self.model.device,
                                     dtype=self.model.dtype)
        self.ignore_index = -100
        # 0.75 0.75
        self.label_smooth = config.label_smooth
        self.task_smooth = config.task_smooth
        self.post_init()
        self.label_mask = nn.Parameter(
            torch.kron(
                torch.ones([self.label_block, self.label_block]),
                torch.eye(self.num_labels) * self.label_smooth) + torch.eye(config.num_labels * self.label_block) * (1 - self.label_smooth))
        self.task_mask = nn.Parameter(
            torch.kron(
                torch.ones([self.task_block, self.task_block]),
                torch.eye(self.num_tasks) * self.task_smooth) + torch.eye(config.num_tasks * self.task_block) * (1 - self.task_smooth))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        type_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Dict, Tuple, torch.Tensor, CLSOutput]:
        with torch.no_grad():
            # if input_ids is not None:
            #     batch_size = input_ids.shape[0]
            # else:
            #     batch_size = inputs_embeds.shape[0]
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            hidden_states:torch.Tensor = outputs[0]

        # if self.model_config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
        #         sequence_lengths = torch.eq(input_ids, self.model_config.pad_token_id).int().argmax(-1) - 1
        #         sequence_lengths = sequence_lengths % input_ids.shape[-1]
        #         sequence_lengths = sequence_lengths.to(hidden_states.device)
        #     else:
        #         sequence_lengths = -1

        # eos_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        # label_scores = self.label_score(eos_hidden_states).float()
        # task_scores = self.task_score(eos_hidden_states).float()

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
        shift_label_masks = labels[..., 1:].not_equal(-100)

        # hidden_states.masked_fill_(shift_label_masks.unsqueeze(-1).expand_as(shift_hidden_states), 0.0)
        shift_hidden_states = norm(shift_hidden_states)
        shift_hidden_states = shift_hidden_states * shift_label_masks.unsqueeze(-1)
        shift_hidden_states = proc(shift_hidden_states, shift_label_masks)

        label_scores = self.label_score(shift_hidden_states).abs().float()
        task_scores = self.task_score(shift_hidden_states).abs().float()

        label_diversity_loss = orthogonal(self.label_score.weight, self.label_mask)
        task_diversity_loss = orthogonal(self.task_score.weight, self.task_mask)

        label_scores = label_scores.view(-1, self.num_labels)
        task_scores = task_scores.view(-1, self.num_tasks)

        with torch.no_grad():
            type_labels = type_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, -100
                                                   ).repeat_interleave(self.label_block, 0)
            task_labels = task_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, -100
                                                   ).repeat_interleave(self.task_block, 0)

        if type_labels is not None and task_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(label_scores, type_labels) * self.label_ratio + loss_fct(task_scores, task_labels) * self.task_ratio
            loss = loss_fct(label_scores, type_labels) \
                + loss_fct(task_scores, task_labels) \
                    + self.block_orth * (label_diversity_loss + task_diversity_loss)
            return (loss, )
        else:
            label_logits = torch.softmax(label_scores, dim=-1)
            task_scores = torch.softmax(task_scores, dim=-1)
        return (label_logits, task_scores, )
