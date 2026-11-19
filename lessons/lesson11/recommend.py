"""Lesson 11: from scores to suggestions. Cross-sell lists, a counting baseline, and the choice among variants.

The model gives, for a basket, a score for every catalogue item (Lesson 8). A suggestion is a choice made from those
scores. This module holds the choices that need no model: the lift adjustment, the top-k list, an item-to-item counting
recommender to compare against, and ranking among the variants (brands and pack sizes) of one product.
"""
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson8"))
from evaluate import rank_of  # noqa: E402


def lift_adjusted(scores, blank, alpha: float = 1.0):
    """scores - alpha * blank, both log-probabilities: what the basket adds to an item's general popularity.

    blank is the model's score for each item with no basket shown. With alpha 1 this is the log of
    P(item | basket) / P(item), a lift; with alpha 0 it is the raw score; in between is a dial.
    """
    return np.asarray(scores) - alpha * np.asarray(blank)


def top_k(scores, k: int, exclude: Iterable[int] = ()) -> List[int]:
    "Positions of the k highest scores, best first, never an excluded one (the items already in the basket)."
    scores = np.array(scores, dtype=float)
    scores[list(exclude)] = -np.inf
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


class ItemToItem:
    """Recommend by counting: for each item in the basket, how often did each other item share a basket with it?

    score(visible) averages P(c | v) = (baskets with v and c) / (baskets with v) over the visible items v. With lift=True
    each is divided by P(c), so common items do not win just by being common. No training beyond a count, no notion of
    order: the fair baseline for anything fancier.
    """

    def __init__(self, train_baskets: Sequence[Sequence[str]], names: Sequence[str]):
        self.position = {name: i for i, name in enumerate(names)}
        together = np.zeros((len(names), len(names)), dtype=np.float64)
        for basket in train_baskets:
            held = np.array(sorted({self.position[x] for x in basket}), dtype=int)
            together[np.ix_(held, held)] += 1
        self.alone = np.diag(together).copy()
        self.together = together
        self.n_baskets = len(train_baskets)

    def score(self, visible: Sequence[str], lift: bool = False):
        held = [self.position[x] for x in visible]
        confidence = (self.together[held] / np.maximum(self.alone[held, None], 1)).mean(axis=0)
        return confidence / ((self.alone + 1) / self.n_baskets) if lift else confidence


def variants_of(catalog) -> Dict[str, List[int]]:
    'Catalogue positions of every SKU of each product (same base: the brands and pack sizes of "Toor Dal").'
    groups: Dict[str, List[int]] = {}
    for position, item in enumerate(catalog):
        groups.setdefault(item.base, []).append(position)
    return groups


def rank_among(scores, target: int, group: Sequence[int]) -> int:
    "The rank of target among only the items in group (ties count against it, as in rank_of)."
    inside = set(group)
    return rank_of(np.asarray(scores), target, [i for i in range(len(scores)) if i not in inside])


def repeat_share(suggested: Sequence[int], visible: Sequence[int], catalog) -> float:
    """The share of suggestions that are another SKU of a product already in the basket. In our data a shopper buys one
    SKU of a product, so such a suggestion is never taken."""
    bases = {catalog[i].base for i in visible}
    return float(np.mean([catalog[i].base in bases for i in suggested])) if len(suggested) else 0.0
