from dataclasses import dataclass
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import ModelOutput

class TransformerConfig(PretrainedConfig):
    model_type = "transformer-impl"
    def __init__(
            self,
            num_hidden_layers = 4,
            d_model = 128,
            vocab_size = 30522,
            **kwargs,
    ):
        super().__init__(self, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.d_model = d_model
        self.vocab_size = vocab_size

class AttentionBlock(GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = x
        return out

class Transformer(PreTrainedModel):
    config_class = TransformerConfig
    # shared weight between LM head and embedding layers
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [AttentionBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        pass

@dataclass
class TransformerOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
