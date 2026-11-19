"""Lesson 8: next-item scoring. Hide one item of a basket, show the model the rest as text, and ask how likely every
catalogue item is as the next one. The rank of the hidden item gives the reciprocal rank (MRR) and hits@K.

PyTorch version (the course's own runs use MLX on an Apple-silicon Mac; this runs on any CPU).
"""
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson7"))
from verbalize import verbalize_prefix  # noqa: E402


def candidate_token_ids(tokenizer, names: Sequence[str]) -> List[List[int]]:
    'The tokens of each item name as it follows "...;" in a basket: a space, then the name.'
    return [list(tokenizer.encode(" " + name)) for name in names]


@torch.no_grad()
def score_candidates(model, prefix_ids: Sequence[int], candidate_ids: Sequence[Sequence[int]], batch: int = 250) -> np.ndarray:
    """log P(candidate | prefix) for every candidate, summed over the candidate's tokens.

    The prefix runs through the model once and its key/value cache is copied for a batch of candidates, so a thousand
    candidates cost a thousand short sequences, not a thousand whole baskets. Candidates are sorted by length and padded
    within each batch to waste little work.
    """
    out = model(torch.tensor([list(prefix_ids)]), use_cache=True)
    first = F.log_softmax(out.logits[0, -1].float(), dim=-1).numpy()  # how likely each token is to start the candidate
    cache, prefix_len = out.past_key_values, len(prefix_ids)

    scores = np.zeros(len(candidate_ids))
    order = sorted(range(len(candidate_ids)), key=lambda i: len(candidate_ids[i]))
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        width = max(len(candidate_ids[i]) for i in chunk)
        rows = len(chunk)
        tokens = torch.tensor([list(candidate_ids[i]) + [0] * (width - len(candidate_ids[i])) for i in chunk])
        chosen = None
        if width > 1:
            shared = copy.deepcopy(cache)
            shared.batch_repeat_interleave(rows)
            mask = torch.ones(rows, prefix_len + width - 1, dtype=torch.long)
            logits = model(tokens[:, :-1], past_key_values=shared, attention_mask=mask).logits.float()
            chosen = F.log_softmax(logits, dim=-1).gather(-1, tokens[:, 1:, None])[..., 0].numpy()
        for row, i in enumerate(chunk):
            length = len(candidate_ids[i])
            scores[i] = first[candidate_ids[i][0]] + (chosen[row, :length - 1].sum() if length > 1 else 0.0)
    return scores


def rank_of(scores: np.ndarray, target: int, exclude: Iterable[int] = ()) -> int:
    "The rank (1 is best) of target among the candidates not in exclude. Ties count against the target."
    keep = np.ones(len(scores), dtype=bool)
    keep[list(exclude)] = False
    keep[target] = False  # the target is compared with the others, never with itself
    return 1 + int((scores[keep] >= scores[target]).sum())


def make_cases(baskets, names: Sequence[str], n_cases: int, seed: int = 0) -> List[dict]:
    "Hide one random item of each of n_cases random baskets: {visible: [...], target: item name}."
    rng = np.random.default_rng(seed)
    cases = []
    for number in rng.choice(len(baskets), size=min(n_cases, len(baskets)), replace=False):
        different = sorted(set(baskets[number]))
        hidden = different[int(rng.integers(len(different)))]
        cases.append({"visible": [n for n in different if n != hidden], "target": hidden})
    return cases


def summarise(ranks: Sequence[int], ks=(1, 10, 50)) -> Dict[str, float]:
    ranks = np.asarray(ranks)
    out = {"mrr": float(np.mean(1.0 / ranks)), "median_rank": float(np.median(ranks))}
    out.update({f"hits@{k}": float(np.mean(ranks <= k)) for k in ks})
    return out


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    "A bootstrap range for the mean of values: resample the cases with replacement and look at the spread."
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2]))


@torch.no_grad()
def score_matrix(model, tokenizer, names: Sequence[str], cases: Sequence[dict], seed: int = 0, log=None) -> np.ndarray:
    "One row per case: the model's log-probability for every catalogue item as the next item (cases x items)."
    candidates = candidate_token_ids(tokenizer, names)
    rows = []
    for k, case in enumerate(cases):
        prefix = verbalize_prefix(case["visible"], seed=seed + k)
        rows.append(score_candidates(model, tokenizer.encode(prefix), candidates))
        if log and (k + 1) % 10 == 0:
            log(f"  scored {k + 1}/{len(cases)} cases")
    return np.array(rows)


def ranks_from_scores(scores: np.ndarray, names: Sequence[str], cases: Sequence[dict]) -> List[int]:
    "The rank of each case's hidden item, among the items not already visible."
    position = {name: i for i, name in enumerate(names)}
    return [rank_of(row, position[c["target"]], [position[n] for n in c["visible"]]) for row, c in zip(scores, cases)]


def popularity_scores(train_baskets, names: Sequence[str]) -> np.ndarray:
    "A baseline that never reads the basket: log of how many training baskets hold each item."
    position = {name: i for i, name in enumerate(names)}
    counts = np.ones(len(names))  # one extra everywhere, so an unseen item scores low rather than minus infinity
    for basket in train_baskets:
        for name in set(basket):
            counts[position[name]] += 1
    return np.log(counts)


def popularity_ranks(train_baskets, names, cases) -> List[int]:
    scores = popularity_scores(train_baskets, names)
    position = {name: i for i, name in enumerate(names)}
    return [rank_of(scores, position[c["target"]], [position[n] for n in c["visible"]]) for c in cases]
