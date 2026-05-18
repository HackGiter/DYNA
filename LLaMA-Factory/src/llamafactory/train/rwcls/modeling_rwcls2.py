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
        intermediate_size:int=8192,
        num_tasks:int=8,
        head_dim:int=128,
        num_key_value_heads:int=8,
        initializer_range:float=0.02,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
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
        self.ref = load_model(tokenizer, model_args, finetuning_args, False, add_valuehead)
        self.ref.eval()
        self.ref.requires_grad_(False)
        self.models:Dict[torch.nn.Parameter] = {}
        for name, param in self.ref.named_parameters():
            if "norm" not in name and "lm_head" not in name and "embed_tokens" not in name:
                self.models[name.replace('.', '-')] = nn.Parameter(
                    # torch.empty_like(param.data).normal_(0.0, self.config.initializer_range),
                    torch.empty([param.data.shape[0] // 4, param.data.shape[1]]).normal_(0.0, self.config.initializer_range),
                    requires_grad=True)
        self.models = torch.nn.ParameterDict(self.models)
        self.post_init()

        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.ignore_index = -100

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        tasks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Dict, Tuple, torch.Tensor, rWCLSOutput]:

        with torch.no_grad():
            if (tasks != 1).sum() > 0:
                extr_input_ids = input_ids[tasks != 1]
                task_labels = labels[tasks == 1]
                extra_attention_mask = attention_mask[tasks != 1]
                logits = self.ref(
                    input_ids=extr_input_ids,
                    attention_mask=extra_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )[0]

            update_h2h_params = torch.randn([self.hidden_size, self.hidden_size],
                                            dtype=torch.bfloat16,
                                            device=self.ref.device,
                                            requires_grad=False) * 4
            update_h2kv_params = torch.randn([self.head_dim * self.num_key_value_heads, self.hidden_size],
                                            dtype=torch.bfloat16,
                                            device=self.ref.device,
                                            requires_grad=False) * 4
            update_h2i_params = torch.randn([self.hidden_size, self.intermediate_size],
                                            dtype=torch.bfloat16,
                                            device=self.ref.device,
                                            requires_grad=False) * 4
            update_i2h_params = torch.randn([self.intermediate_size, self.hidden_size],
                                            dtype=torch.bfloat16,
                                            device=self.ref.device,
                                            requires_grad=False) * 4

        for name, param in self.ref.named_parameters():
            name = name.replace('.', '-')
            if name in self.models.keys():
                # detached_param = param.detach()
                if param.numel() == self.hidden_size * self.hidden_size:
                    # param.add_(torch.matmul(
                    #     F.linear(update_h2h_params, self.models[name]),
                    #     self.models[name]))
                    # param.data = param.data + torch.matmul(
                    #     F.linear(update_h2h_params, self.models[name]),
                    #     self.models[name])
                    param = param.detach() + torch.matmul(
                        F.linear(update_h2h_params, self.models[name]),
                        self.models[name])
                elif param.shape[0] == self.head_dim * self.num_key_value_heads:
                    # param.add_(torch.matmul(
                    #     F.linear(update_h2kv_params, self.models[name]),
                    #     self.models[name]))
                    # param.data = param.data + torch.matmul(
                    #     F.linear(update_h2kv_params, self.models[name]),
                    #     self.models[name])
                    param = param.detach() + torch.matmul(
                        F.linear(update_h2kv_params, self.models[name]),
                        self.models[name])
                elif param.shape[0] == self.hidden_size:
                    # param.add_(torch.matmul(
                    #     F.linear(update_h2i_params, self.models[name]),
                    #     self.models[name]))
                    # param.data = param.data + torch.matmul(
                    #     F.linear(update_h2i_params, self.models[name]),
                    #     self.models[name])
                    param = param.detach() + torch.matmul(
                        F.linear(update_h2i_params, self.models[name]),
                        self.models[name])
                else:
                    # param.add_(torch.matmul(
                    #     F.linear(update_i2h_params, self.models[name]),
                    #     self.models[name]))
                    # param.data = param.data + torch.matmul(
                    #     F.linear(update_i2h_params, self.models[name]),
                    #     self.models[name])
                    param = param.detach() + torch.matmul(
                        F.linear(update_i2h_params, self.models[name]),
                        self.models[name])
                # update_params[name] = torch.matmul(self.models[name].T,
                #                                 torch.matmul(self.models[name],
                #                                                 torch.randn_like(param, dtype=torch.float32, requires_grad=False)))


        outputs = self.ref(
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
        )[0]

        # with torch.no_grad():
        #     for name, param in self.ref.named_parameters():
        #         name = name.replace('.', '-')
        #         if name in self.models.keys():
        #             if param.numel() == self.hidden_size * self.hidden_size:
        #                 param.detach_().sub_(torch.matmul(
        #                     F.linear(update_h2h_params, self.models[name].detach()),
        #                     self.models[name]))
        #             elif param.shape[0] == self.head_dim * self.num_key_value_heads:
        #                 param.detach_().sub_(torch.matmul(
        #                     F.linear(update_h2kv_params, self.models[name].detach()),
        #                     self.models[name]))
        #             elif param.shape[0] == self.hidden_size:
        #                 param.detach_().sub_(torch.matmul(
        #                     F.linear(update_h2i_params, self.models[name].detach()),
        #                     self.models[name]))
        #             else:
        #                 param.detach_().sub_(torch.matmul(
        #                     F.linear(update_i2h_params, self.models[name].detach()),
        #                     self.models[name]))

        loss = 0
        print(tasks)
        if (tasks == 1).sum() >= 1:
            loss_fct = nn.CrossEntropyLoss()
            loss += loss_fct(outputs[tasks == 1][:, :-1, :].contiguous().view(-1, self.ref.config.vocab_size),
                    task_labels[:, 1:].contiguous().view(-1))
            print(f"1:{loss}")
        if (tasks != 1).sum() >= 1:
            loss += (outputs[tasks != 1].softmax(-1) - logits.softmax(-1)).abs().float().sum(-1).mean()
            print(f"2:{loss}")
        print(f"3:{loss}")
        return (loss, )
