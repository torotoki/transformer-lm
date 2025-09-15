import multiprocessing as mp
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from model import TransformerConfig, Transformer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

num_proc = max(1, mp.cpu_count() - 1)

def encode(ex):
    out = tok(ex["text"], truncation=True, padding=False, max_length=64)
    return out

ds = load_dataset("roneneldan/TinyStories").map(
    encode, batched=1000, num_proc=num_proc,
)
collator = DataCollatorWithPadding(tokenizer=tok)

config = TransformerConfig(vocab_size=tok.vocab_size, d_model=128, num_hidden_layers=2)
model = Transformer(config)

args = TrainingArguments(
    output_dir="out-custom",
    torch_compile=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="steps",
    save_strategy="epoch",  #TODO: epoch
    logging_steps=50,
    fp16=False,
    report_to="wandb"  # or "none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation[:10%]'],
    tokenizer=tok,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("out-custom") 
