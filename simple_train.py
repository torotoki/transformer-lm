from transformers import AutoTokenizer, DataCollatorWithPadding, GenerationConfig, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from model import TransformerConfig, Transformer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "This message is hello world",
    "Goodbye world until the sun rises",
    "Smashing the mics in the bar",
    "custom transformer model works",
    "trainer rocks",
    "pytorch is fun"
]

def encode(ex):
    out = tok(ex["text"], truncation=True, padding=False, max_length=64)
    return out

ds = Dataset.from_dict({"text": texts}).map(encode)
collator = DataCollatorWithPadding(tokenizer=tok)

config = TransformerConfig(vocab_size=tok.vocab_size, d_model=128, num_hidden_layers=2)
model = Transformer(config)

args = TrainingArguments(
    output_dir="out-custom",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=100,
    eval_strategy="epoch",
    save_strategy="no",  #TODO: epoch
    logging_steps=1,
    fp16=False,
    report_to="wandb"  # or "none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds.select(range(4)),
    eval_dataset=ds.select(range(4, len(ds))),
    processing_class=tok,
    data_collator=collator,
)

trainer.train()

folder ="out-custom-simple"
print("Save folder:", folder)
# Add `generation_config.json``
gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.eos_token_id,
    use_cache=False,  # KV-cache is not supported yet
)
# Add model code in the saved directory
config.register_for_auto_class()
model.register_for_auto_class("AutoModelForCausalLM")
model.save_pretrained(folder) 
config.save_pretrained(folder)
gen_config.save_pretrained(folder)