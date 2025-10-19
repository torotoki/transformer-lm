"""
Usage:
```py
python language_train.py
```

Running with multiple GPUs:
```py
accelerate launch language_train.py
"""
import os
import multiprocessing as mp
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import wandb
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset, load_from_disk
from model import TransformerConfig, Transformer


def main():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    accelerator = Accelerator()

    #dataset_path = "datasets/fineweb-edu-sample-10BT-tokenized/"
    dataset_path = "datasets/tiny-stories-tokenized/"
    print("Preprocessed dataset path:", dataset_path)
    if accelerator.is_main_process and not os.path.exists(dataset_path):
        print("Please run the preprocessing script before training the model.")
        exit(-1)
    ds = load_from_disk(dataset_path)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    config = TransformerConfig(
        vocab_size=tok.vocab_size,
        d_model=256,
        num_hidden_layers=8,
    )
    model = Transformer(config)
    folder = None
    if accelerator.is_main_process:
        print(config)
        print("#Model parameters:", model.num_parameters())
        num_tokens = sum(ds["train"]["num_tokens"])
        print("#Total training tokens", num_tokens)
        print("#Avg training tokens", num_tokens / len(ds["train"]))

        run = wandb.init(
            project="transformer-lm",
        )
        folder = f"outputs/{run.name}-{run.id}"
        print("Output folder:", folder)
    args = TrainingArguments(
        output_dir=folder,
        torch_compile=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        eval_strategy="no",
        save_strategy="steps",  #or "epoch"
        save_steps=10000,
        eval_steps=2000,
        logging_steps=50,
        fp16=False,
        report_to="wandb",  # or "none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        #eval_dataset=ds['validation'],
        processing_class=tok,
        data_collator=collator,
    )

    trainer.train()

    ## Save model files
    
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
    if accelerator.is_main_process:
        # Add model code in the saved directory
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModelForCausalLM")
        model.save_pretrained(folder)
        config.save_pretrained(folder)
        tok.save_pretrained(folder)  # tokenizer is also saved
        gen_config.save_pretrained(folder)

if __name__ == '__main__':
    main()
