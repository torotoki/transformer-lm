"""
Preprocess the training dataset tokenization and
save it to the specified directory.
"""

import multiprocessing as mp
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset


def main():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    num_procs = max(1, mp.cpu_count() - 1)

    def encode(ex):
        # context size: 512
        out = tok(ex["text"], truncation=True, padding=False, max_length=512)
        out["num_tokens"] = list(map(lambda ex: len(ex), out["input_ids"]))
        return out

    # We only use 10% of the validation dataset
    out_dir = "datasets/fineweb-edu-sample-10BT-tokenized/"
    ds = load_dataset(
        #"roneneldan/TinyStories",
        # split={"train": "train[:100%]",
        #     "validation": "validation[:10%]"}
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split={"train": "train[:100%]"},
    ).map(
        encode, batched=1000, num_proc=num_procs,
    )

    num_tokens = sum(ds["train"]["num_tokens"])
    print("Output directory:", out_dir)
    print("#Total training tokens:", num_tokens)
    print("#Avg training tokens:", num_tokens / len(ds["train"]))
    ds.save_to_disk(out_dir)


if __name__ == '__main__':
    main()
