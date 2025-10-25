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


def _load_fineweb():
    return load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split={"train": "train[:100%]"},
    )

def _load_tinystories():
    return load_dataset(
        "roneneldan/TinyStories",
        split={"train": "train[:100%]",
            "validation": "validation[:10%]"}
    )

def main():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    num_procs = max(1, mp.cpu_count() - 1)

    def encode(ex):
        # context size: 512
        out = tok(ex["text"], truncation=True, padding=False, max_length=512)
        out["num_tokens"] = list(map(lambda ex: len(ex), out["input_ids"]))
        return out

    # We only use 10% of the validation dataset for tiny-stories
    use_dataset = "fineweb-edu"
    out_dir_aliases = {
        "fineweb-edu": "datasets/fineweb-edu-sample-10BT-tokenized/",        "tiny-stories": "datasets/tiny-stories-tokenized/",
    }
    dataset_aliases = {
        "fineweb-edu": _load_fineweb,
        "tiny-stories": _load_tinystories,
    }
    out_dir = out_dir_aliases[use_dataset]
    ds = dataset_aliases[use_dataset]().map(
        encode, batched=1000, num_proc=num_procs,
    )

    num_tokens = sum(ds["train"]["num_tokens"])
    print("Output directory:", out_dir)
    print("#Total training tokens:", num_tokens)
    print("#Avg training tokens:", num_tokens / len(ds["train"]))
    ds.save_to_disk(out_dir)


if __name__ == '__main__':
    main()
