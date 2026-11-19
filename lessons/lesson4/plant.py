"""Lesson 4: plant one itemset at an exact support, so we know the right answer."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Planted:
    "An itemset put into the data on purpose, so we know the answer."
    items: tuple          # 2 to 5 item names
    support: float        # target share of baskets holding every one of them
    kind: str = ""        # "natural" or "surprising", for analysis later


def find_items(catalog, base, brand=None, size=None):
    "The SKUs of a product that this catalogue carries, optionally narrowed to a brand and/or size."
    return [i for i in catalog if i.base == base and (brand is None or i.brand == brand) and (size is None or i.size == size)]


def support_of(baskets, items):
    "The share of baskets holding every one of the items (quantity does not matter)."
    wanted = set(items)
    return sum(wanted <= set(b) for b in baskets) / len(baskets)


def plant_itemset(baskets, planted, catalog, seed=0, quantity_probs=(0.80, 0.14, 0.04, 0.02), allow_excess=False):
    """A copy of baskets where exactly round(support x n) baskets hold the whole planted itemset.

    Some baskets already hold it by chance and they count. Just enough others are picked at random and given the
    items; a different brand or size of a planted product is swapped out so a basket never holds two SKUs of one
    product. Too common by chance to plant is an error (or left alone with allow_excess).
    """
    items = list(planted.items)
    product_of = {i.name: (i.category, i.base) for i in catalog}
    unknown = [n for n in items if n not in product_of]
    if unknown:
        raise ValueError(f"not in the catalogue: {unknown}")
    if len(set(items)) != len(items) or len({product_of[n] for n in items}) != len(items):
        raise ValueError("a planted itemset needs different products: two of these are the same item or product")
    if not 0 < planted.support < 1:
        raise ValueError(f"support must be between 0 and 1, got {planted.support}")

    rng = np.random.default_rng(seed)
    wanted, planted_products = set(items), {product_of[n] for n in items}
    target = round(planted.support * len(baskets))
    holds = np.array([wanted <= set(b) for b in baskets])
    already = int(holds.sum())
    if already > target and allow_excess:
        return list(baskets)
    if already > target:
        raise ValueError(f"{already} baskets already hold this itemset by chance, more than the {target} asked for: "
                         f"it is too common to plant at support {planted.support}")
    picked = rng.choice(np.flatnonzero(~holds), size=target - already, replace=False)

    out = list(baskets)
    for i in picked:
        kept = [n for n in out[i] if n in wanted or product_of[n] not in planted_products]
        have = set(kept)
        added = [n for n in items if n not in have for _ in range(1 + rng.choice(len(quantity_probs), p=quantity_probs))]
        basket = kept + added
        out[i] = [basket[j] for j in rng.permutation(len(basket))]
    return out
