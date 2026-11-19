"""Lesson 10: probe the trained model. What does it believe goes together?

A language model has no table of itemsets inside it, only a distribution over baskets: it can WRITE one, item after item.
So we ask it to write many, count what turns up together with the same exact miner that counted the real baskets, and
compare. Where the model's frequent itemsets match the truth, it learned them; where they differ, it did not (or made some up).
"""
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for lesson in ("lesson6", "lesson7"):
    sys.path.insert(0, str(HERE.parent / lesson))
from mine import frequent_itemsets  # noqa: E402
from verbalize import PREFIX, SEPARATOR  # noqa: E402


@torch.no_grad()
def sample_baskets(model, tokenizer, names: Sequence[str], n: int, batch: int = 100, max_tokens: int = 240,
                   temperature: float = 1.0, seed: int = 0) -> Tuple[List[List[str]], Dict[str, float]]:
    """Have the model write n baskets from the bare prompt "Basket:". Returns (baskets, stats).

    A batch is written together, one token per step, sharing one key/value cache; a basket is finished when the model
    writes the end-of-text token. Names not in the catalogue are dropped, a repeated item is kept once, and a basket
    cut off at max_tokens loses its last (possibly half-written) item. stats says how often each of those happened.
    """
    known = set(names)
    prompt = list(tokenizer.encode(PREFIX.rstrip()))
    eos = tokenizer.eos_token_id
    generator = torch.Generator().manual_seed(seed)
    baskets, unknown, repeats, cut, written_items = [], 0, 0, 0, 0
    for start in range(0, n, batch):
        size = min(batch, n - start)
        out = model(torch.tensor([prompt] * size), use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        written = [[] for _ in range(size)]
        done = np.zeros(size, dtype=bool)
        for _ in range(max_tokens):
            token = torch.multinomial(F.softmax(logits.float() / temperature, dim=-1), 1, generator=generator)
            chosen = token[:, 0].numpy()
            for row in np.flatnonzero(~done):
                if chosen[row] == eos:
                    done[row] = True
                else:
                    written[row].append(int(chosen[row]))
            if done.all():
                break
            out = model(token, past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1, :]
        for row, ids in enumerate(written):
            pieces = [p.strip() for p in tokenizer.decode(ids).split(SEPARATOR.strip()) if p.strip()]
            if not done[row] and pieces:
                pieces = pieces[:-1]
                cut += 1
            written_items += len(pieces)
            unknown += sum(p not in known for p in pieces)
            pieces = [p for p in pieces if p in known]
            repeats += len(pieces) - len(set(pieces))
            baskets.append(list(dict.fromkeys(pieces)))
    stats = {"baskets": len(baskets), "items written": written_items,
             "share not in catalogue": unknown / max(1, written_items), "share repeated": repeats / max(1, written_items),
             "baskets cut off": cut / max(1, len(baskets)),
             "mean items per basket": float(np.mean([len(b) for b in baskets]))}
    return baskets, stats


def recover_itemsets(sampled, min_support: float, max_size: int = 5) -> Dict[frozenset, float]:
    "The itemsets frequent in the model's own baskets, with their support there: the model's estimate."
    return frequent_itemsets(list(sampled), min_support, max_size=max_size)


def recovery_by_support(recovered, truth, bands: Sequence[float], group=None) -> List[dict]:
    """How many of the true itemsets did the model recover, per band of true support?

    bands are ascending support edges, e.g. [0.005, 0.01, 0.05, 1]: the first band is [0.005, 0.01). group optionally
    labels each itemset to split every band further. Each row: band, label, true (how many), found, recall.
    """
    rows = []
    for low, high in zip(bands[:-1], bands[1:]):
        in_band = [s for s, v in truth.items() if low <= v < high]
        for label in (sorted({group(s) for s in in_band}) if group else [None]):
            chosen = [s for s in in_band if group is None or group(s) == label]
            found = sum(s in recovered for s in chosen)
            rows.append({"band": f"{low:.2%} to {min(high, 1):.2%}", "label": label, "true": len(chosen),
                         "found": found, "recall": found / len(chosen) if chosen else float("nan")})
    return rows


def precision_of(recovered, truth, min_support: float) -> float:
    "Of the itemsets the model calls frequent, the share that really are (true support at least min_support)."
    if not recovered:
        return float("nan")
    return float(np.mean([truth.get(s, 0.0) >= min_support for s in recovered]))


def support_error(recovered, truth) -> np.ndarray:
    "For itemsets in both, the model's support minus the true support."
    return np.array([recovered[s] - truth[s] for s in recovered if s in truth])
