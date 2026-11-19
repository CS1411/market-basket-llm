"""Lesson 7: verbalize a basket. Text a language model can read, with the item order shuffled."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson6"))
from dataset import load_dataset  # noqa: E402
from verbalize import PREFIX, parse_basket, verbalize_all, verbalize_basket, verbalize_prefix  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"

if __name__ == "__main__":
    data = load_dataset(Path(__file__).resolve().parents[2] / "data" / "dataset")  # written by Lesson 6
    basket = data.baskets[0]
    print("a basket as the till saw it:", basket[:5], "...")
    for seed in (0, 1, 2):
        print(f"seed {seed}:", verbalize_basket(basket, seed=seed)[:110], "...")
    assert verbalize_basket(basket, seed=0) == verbalize_basket(basket, seed=0)       # repeatable
    assert verbalize_basket(basket, seed=0) != verbalize_basket(basket, seed=1)       # a fresh order each time
    assert sorted(parse_basket(verbalize_basket(basket, seed=5))) == sorted(set(basket))  # nothing lost, quantity dropped
    print("prefix:", repr(verbalize_prefix(sorted(set(basket))[:2], seed=0)), "| empty:", repr(verbalize_prefix([])))

    # Why shuffle? If every basket were written in a fixed (alphabetical) order, "which item comes next" would be partly
    # decided by that order, which means nothing. Check that the shuffle leaves no first-item bias:
    firsts = [parse_basket(t)[0] for t in verbalize_all(data.baskets[:5000], seed=0)]
    alphabetical = [sorted(set(b))[0] for b in data.baskets[:5000]]
    print(f"distinct first items over 5,000 baskets: shuffled {len(set(firsts))}, alphabetical {len(set(alphabetical))}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = verbalize_all(data.baskets[:2000], seed=0)
    lengths = np.array([len(tok.encode(t)) + 1 for t in texts])  # plus the end-of-text token
    print(f"\ntokens per basket with the {MODEL.split('/')[1]} tokenizer: mean {lengths.mean():.0f}, "
          f"median {int(np.median(lengths))}, max {lengths.max()}")
    name = data.catalog[0].name
    print(f"one item name is {len(tok.encode(' ' + name))} tokens: {tok.convert_ids_to_tokens(tok.encode(' ' + name))}")
    print(f"vocabulary {len(tok):,} tokens; end-of-text id {tok.eos_token_id}")
