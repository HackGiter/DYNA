import math
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
import torch.nn as nn
from torch.nn import MSELoss

from transformers.cache_utils import Cache
from transformers.utils import add_start_docstrings, ModelOutput, logging
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ...model import load_model, load_config
from ...model.loader import _get_init_kwargs, patch_config

logger = logging.get_logger(__name__)

def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

@dataclass
class SAEOutput(ModelOutput):
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None
    activations: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[str, torch.FloatTensor]]] = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments

SAE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SAEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

class SAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SAEModel`]. It is used to instantiate an SAE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SAE-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the SAE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SAEModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    ```python
    >>> from transformers import SAEModel, SAEConfig

    >>> # Initializing a SAE SAE-7b style configuration
    >>> configuration = SAEConfig()

    >>> # Initializing a model from the SAE-7b style configuration
    >>> model = SAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sae"

    def __init__(
        self,
        k:Union[int, List[int]]=2,
        hidden_size:int=4096,
        base_latent_size:int=8,
        num_hidden_layers:int=5,
        initializer_range:float=0.02,
        hidden_act:str="relu",
        topk:bool=True,
        eps:float=1e-5,
        **kwargs,
    ):
        self.k = k
        self.hidden_size = hidden_size
        self.base_latent_size = base_latent_size
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.topk = topk
        self.eps = eps

        super().__init__(
            **kwargs,
        )

from functools import partial
class Hooker:
    def __init__(self, model:Union[nn.Module, PreTrainedModel]):
        self.modules = {}
        self.hooks = {}
        self.hook_names = []
        self.hook_outputs = {}
        for name, module in model.named_modules():
            self.modules[name] = module

    def register_hooks(
        self,
        names:Union[str, List[str]],
    ):
        self.hook_names = [names] if isinstance(names, str) else names
        for name in self.hook_names:
            def hook_fn(
                module: torch.nn.Module,
                module_inputs: Any,
                module_outputs: Any,
                cls: Hooker = None,
                name: str = None,
            ):
                cls.hook_outputs[name] = module_inputs[0].detach()
            handle = self.modules[name].register_forward_hook(
                partial(hook_fn, cls=self, name=name)
            )
            self.hooks[name] = [handle]

@add_start_docstrings(
    "The bare SAE Model outputting raw hidden-states without any specific head on top.",
    SAE_START_DOCSTRING,
)
class SAEPreTrainedModel(PreTrainedModel):
    config_class = SAEConfig
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

class SparseTreeEncoder(nn.Module):
    def __init__(self, config:SAEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            nn.Linear(config.hidden_size, int(math.pow(config.base_latent_size, i+1)), dtype=config.torch_dtype)
            for i in range(config.num_hidden_layers)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        bsz, _ = hidden_states.size()
        if self.config.topk:
            topk = [
                self.config.k ** (i + 1)
                for i in range(self.config.num_hidden_layers)
            ]
            act = self.act_fn(self.layers[0](hidden_states)).float()
            activations = [act * torch.zeros_like(act).scatter_(-1, index=torch.topk(act, k=topk[0], dim=-1).indices, value=1.0)]

            for i, encoder in enumerate(self.layers[1:]):
                act = self.act_fn(encoder(hidden_states)).view(bsz, -1, self.config.base_latent_size).float()
                prev_act = activations[-1]
                act = (prev_act.view(bsz, -1, 1) * act).view(bsz, -1)
                activations.append(act * torch.zeros_like(act).scatter_(-1, index=torch.topk(act, k=topk[i+1], dim=-1).indices, value=1.0))
        else:
            activations = [self.act_fn(self.layers[0](hidden_states)).float()]
            for _, encoder in enumerate(self.layers[1:]):
                act = self.act_fn(encoder(hidden_states)).float().view(bsz, -1, self.config.base_latent_size)
                prev_act = activations[-1]
                act = (prev_act.view(bsz, -1, 1) * act).view(bsz, -1)
                activations.append(act)
        activations = torch.cat(activations, dim=-1)
        return activations

class SAE(SAEPreTrainedModel):
    def __init__(
            self,
            config: SAEConfig,
            tokenizer: "PreTrainedTokenizer",
            model_args: "ModelArguments",
            finetuning_args: "FinetuningArguments",
            add_valuehead: bool = False,
            **kwargs
            ):
        super().__init__(config)
        self.latent_size = sum([
            int(math.pow(config.base_latent_size, i + 1))
            for i in range(config.num_hidden_layers)
        ])
        init_kwargs = _get_init_kwargs(model_args)
        model_config = load_config(model_args)
        patch_config(model_config, tokenizer, model_args, init_kwargs, False)
        self.model = load_model(tokenizer, model_args, finetuning_args, False, add_valuehead)
        self.encoder = SparseTreeEncoder(config)
        self.decoder = nn.Linear(self.latent_size, config.hidden_size, dtype=config.torch_dtype)
        # self.module_names = module_names if isinstance(module_names, List) else [module_names]
        self.ignore_index = -100
        self.hook = Hooker(self.model)
        self.setup_hooks("model.layers.22")

    def setup_hooks(
        self,
        module_names: Union[str, List[str]],
    ):
        self.module_names = module_names if isinstance(module_names, List) else [module_names]
        # self.hook = Hooker(self.model)
        self.hook.register_hooks(self.module_names)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Dict, Tuple, torch.Tensor, SAEOutput]:
        with torch.no_grad():
            labels = labels.view(-1).eq(self.ignore_index)
            self.model(
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
            residual_hidden_states = self.hook.hook_outputs
            bsz, seq_len, _ = residual_hidden_states[self.module_names[0]].size()
            residual_hidden_states = { k:LN(v.view(bsz * seq_len, -1))[0] for k, v in residual_hidden_states.items() }

        activations, outputs = {}, {}
        for name, hidden_states in residual_hidden_states.items():
            activations[name] = self.encoder(hidden_states)
            outputs[name] = self.decoder(activations[name])
        if self.training:
            loss = 0
            loss_fn = MSELoss(reduction="none")
            active = (labels.numel() - labels.long().sum())
            for k, v in outputs.items():
                loss += torch.mean(loss_fn(v, residual_hidden_states[k]), dim=-1).masked_fill_(labels, 0.0).sum() / active
            return (loss, activations)
        return (outputs, activations)
