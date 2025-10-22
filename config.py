@dataclass
class TrainConfig:
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    

def small_model():
    config = TransformerConfig(
        vocab_size=tok.vocab_size,
        d_model=256,
        num_hidden_layers=8,
    )
    target_tokens = 20_000_000 * 20
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 1
    token_per_step = \
        config.context_size * \
        per_device_train_batch_size * \
        gradient_accumulation_steps * \
        num_gpus
    
