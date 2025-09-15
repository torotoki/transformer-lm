from math import sqrt
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
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
            context_size = 512,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_size = context_size

@dataclass
class TransformerOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None

class AttentionBlock(GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(config.d_model, config.d_model * 3)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, D = x.shape
        assert((B, T) == attention_mask.shape)
        # 1. Apply the attention mask to x
        # 2. Compute Q, K, V from the linear projection
        # 3. Compute the self attention over Q, K, V
        
        # Q, K, V: (B, T, D)
        Q, K, V = self.in_proj(x).chunk(3, dim=-1)

        hidden_states = Q @ K.transpose(1, 2)  # (B, T, T)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]
        
        hidden_states = F.softmax(hidden_states / sqrt(D), dim=-1)  # (B, T, T)
        hidden_states = hidden_states @ V  # (B, T, D)

        return hidden_states

class MLPBlock(GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(config.d_model, config.d_model * 4)
        self.gate = nn.GELU()
        self.out_proj = nn.Linear(config.d_model * 4, config.d_model)
        self.dropout = nn.Dropout()
    
    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        x = self.in_proj(x)
        x = self.gate(x)
        x = self.out_proj(x)
        x = self.dropout(x)

        return x

class TransformerBlock(GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm_fn1 = nn.RMSNorm((config.d_model))
        self.attention_block = AttentionBlock(config, layer_idx)  # d_model -> d_model
        self.norm_fn2 = nn.RMSNorm((config.d_model))
        self.mlp_block = MLPBlock(config, layer_idx)  # d_model -> d_model
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # x: (B, T, D)
        # Normalize the hidden states before applying attention and MLP layers
        x = self.norm_fn1(x)
        x = self.attention_block(x, attention_mask)
        x = self.norm_fn2(x)
        x = self.mlp_block(x)
        return x

class Transformer(PreTrainedModel):
    config_class = TransformerConfig
    # shared weight between LM head and embedding layers
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()  # Init weight (HF recommended)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, labels: torch.Tensor = None) -> TransformerOutput:
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is None:
            labels = input_ids
        
        # Remove the last token
        shift_logits = logits[..., :-1, :].contiguous()
        # Remove the first token
        shift_labels = labels[..., 1:].contiguous()

        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)
        
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),  # (B * T, vocab_size)
            shift_labels.view(-1)  # (B * T,)
        )

        return TransformerOutput(
            loss = loss,
            logits = logits,
        )

