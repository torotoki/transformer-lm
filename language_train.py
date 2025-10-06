import multiprocessing as mp
from transformers import AutoTokenizer, DataCollatorWithPadding, GenerationConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from model import TransformerConfig, Transformer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

num_proc = max(1, mp.cpu_count() - 1)

def encode(ex):
    out = tok(ex["text"], truncation=True, padding=False, max_length=64)
    return out

# We only use 10% of the validation dataset
ds = load_dataset(
    "roneneldan/TinyStories",
    split={"train": "train[:100%]",
           "validation": "validation[:10%]"}
).map(
    encode, batched=1000, num_proc=num_proc,
)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

config = TransformerConfig(vocab_size=tok.vocab_size, d_model=128, num_hidden_layers=2)
model = Transformer(config)

args = TrainingArguments(
    output_dir="out-custom",
    torch_compile=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    eval_strategy="steps",
    save_strategy="epoch",  #TODO: epoch
    logging_steps=50,
    eval_steps=2000,
    fp16=False,
    report_to="wandb",  # or "none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    processing_class=tok,
    data_collator=collator,
)

trainer.train()

metrics = trainer.evaluate()
print("eval_loss:", metrics)

folder ="out-custom-language"
# Add `generation_config.json`
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