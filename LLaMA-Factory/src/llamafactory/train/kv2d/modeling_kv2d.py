import math
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
import torch.nn as nn
import torch.functional as F

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
class KV2dOutput(ModelOutput):
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None
    activations: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments

KV2d_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`KV2dConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

class KV2dConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`KV2dModel`]. It is used to instantiate an KV2d
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the KV2d-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the KV2d model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`KV2dModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    ```python
    >>> from transformers import KV2dModel, KV2dConfig

    >>> # Initializing a KV2d KV2d-7b style configuration
    >>> configuration = KV2dConfig()

    >>> # Initializing a model from the KV2d-7b style configuration
    >>> model = KV2dModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cls"

    def __init__(
        self,
        hidden_size:int=4096,
        expand_head_size:int=2,
        initializer_range:float=0.02,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.expand_head_size = expand_head_size
        self.initializer_range = initializer_range

@add_start_docstrings(
    "The bare KV2d Model outputting raw hidden-states without any specific head on top.",
    KV2d_START_DOCSTRING,
)
class KV2dPreTrainedModel(PreTrainedModel):
    config_class = KV2dConfig
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

from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

class KV2dAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.expand_head_dim = self.head_dim * self.config.expand_head_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.k2d_proj = nn.Linear(self.head_dim, self.expand_head_dim, bias=config.attention_bias)
        self.v2d_proj = nn.Linear(self.head_dim, self.expand_head_dim, bias=config.attention_bias)
        self.q2d_proj = nn.Linear(self.head_dim, self.expand_head_dim, bias=config.attention_bias)

        self.k2d_gate = nn.Linear(self.expand_head_dim, self.num_key_value_heads * 2, bias=config.attention_bias)
        self.v2d_gate = nn.Linear(self.expand_head_dim, self.num_key_value_heads * 2, bias=config.attention_bias)
        self.q2d_proj = nn.Linear(self.expand_head_dim, self.num_heads * 2, bias=config.attention_bias)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        sparse_entropy_loss = None
        if kwargs.get("prev_qkv_states", None) is not None:
            prev_qkv_states = kwargs.pop("prev_qkv_states")
            expand_key_states = torch.cat([key_states, prev_qkv_states[0]], dim=2)
            expand_value_states = torch.cat([value_states, prev_qkv_states[1]], dim=2)
            expand_query_states = torch.cat([query_states, prev_qkv_states[2]], dim=2)
            k_states_logits = self.k2d_gate(self.k2d_proj(key_states))
            v_states_logits = self.v2d_gate(self.v2d_proj(value_states))
            q_states_logits = self.q2d_proj(self.q2d_proj(query_states))
            k_states_activation = k_states_logits.softmax(dim=-1, dtype=torch.float32)
            v_states_activation = v_states_logits.softmax(dim=-1, dtype=torch.float32)
            q_states_activation = q_states_logits.softmax(dim=-1, dtype=torch.float32)
            key_states = torch.matmul(k_states_activation.to(key_states.dtype), expand_key_states)
            value_states = torch.matmul(v_states_activation.to(value_states.dtype), expand_value_states)
            query_states = torch.matmul(q_states_activation.to(query_states.dtype), expand_query_states)

            k_states_act_acc = k_states_activation.sum(dim=2)
            v_states_act_acc = v_states_activation.sum(dim=2)
            sparse_entropy_loss = - (k_states_activation * k_states_logits.float().log_softmax(dim=-1, dtype=torch.float32)).sum(-1) \
                                - (v_states_activation * v_states_logits.float().log_softmax(dim=-1, dtype=torch.float32)).sum(-1) \
                                - (k_states_act_acc.softmax(dim=-1, dtype=torch.float32) * k_states_act_acc.log_softmax(dim=-1, dtype=torch.float32)).sum(-1) \
                                - (v_states_act_acc.softmax(dim=-2, dtype=torch.float32) * v_states_act_acc.log_softmax(dim=-2, dtype=torch.float32)).sum(-1)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if sparse_entropy_loss is not None:
            return attn_output, attn_weights, past_key_value, sparse_entropy_loss

        return attn_output, attn_weights, past_key_value

class KV2d(KV2dPreTrainedModel):
    def __init__(
            self, 
            config: KV2dConfig,
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
        self.model = LlamaModel.from_pretrained(**init_kwargs)
        register_autoclass(self.model_config, self.model, tokenizer)
        self.model = init_adapter(self.model_config, self.model, model_args, finetuning_args, False)
        self.model.requires_grad_(False)
        self.model.eval()
        self.task_score = nn.Linear(config.hidden_size, config.num_tasks, bias=False, device=self.model.device, dtype=self.model.dtype)
        self.label_score = nn.Linear(config.hidden_size, config.num_labels, bias=False, device=self.model.device, dtype=self.model.dtype)
        self.ignore_index = -100
        self.post_init()

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
    ) -> Union[Dict, Tuple, torch.Tensor, KV2dOutput]:
        with torch.no_grad():
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]
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

        def norm(x:torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            x = x.float()
            x = x - x.mean(-1, keepdim=True)
            x = x * x.pow(2).mean(-1, keepdim=True).rsqrt()
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
        
        label_scores = self.label_score(shift_hidden_states).float()
        task_scores = self.task_score(shift_hidden_states).float()

        with torch.no_grad():
            type_labels = type_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, -100)
            task_labels = task_labels.masked_fill_(shift_label_masks.sum(dim=-1) < 1, -100)

        if type_labels is not None and task_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(label_scores, type_labels) * self.label_ratio + loss_fct(task_scores, task_labels) * self.task_ratio
            loss = loss_fct(label_scores, type_labels) + loss_fct(task_scores, task_labels)
            return (loss, )
        else:
            label_logits = torch.softmax(label_scores, dim=-1)
            task_scores = torch.softmax(task_scores, dim=-1)
        return (label_logits, task_scores, )
        
