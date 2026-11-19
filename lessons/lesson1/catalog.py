"""Lesson 1: a shop's catalogue. Products x brands x pack sizes become SKUs, and a catalogue picks 1,000 of them."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from catalog_data import CATEGORIES


@dataclass(frozen=True)
class Item:
    "One SKU: a product in one brand and one pack size."
    name: str                # the full name a model will read, e.g. "Surya Grains Toor Dal 1 kg"
    base: str                # the product without brand or size, e.g. "Toor Dal"
    category: str            # e.g. "Pulses"
    brand: Optional[str]     # None for loose goods such as fresh produce
    size: str                # e.g. "1 kg", "Rs 2 sachet"


def split_product(product):
    "A product is (name, weight) or (name, weight, {size: weight}) when it has its own pack sizes."
    return product if len(product) == 3 else (product[0], product[1], None)


def all_skus(brands_per_product, rng) -> List[Item]:
    """Every SKU the shop could stock: each product in each pack size under a few brands.

    A shop carries a few brands of each product, and a big brand is more likely to be stocked than a small one.
    """
    skus = []
    for category, spec in CATEGORIES.items():
        brand_names = list(spec["brands"])
        brand_weights = np.array(list(spec["brands"].values()), dtype=float)
        for product in spec["products"]:
            base, _, own_sizes = split_product(product)
            sizes = own_sizes or spec["sizes"]
            if brand_names:
                k = min(brands_per_product, len(brand_names))
                picked = rng.choice(len(brand_names), size=k, replace=False, p=brand_weights / brand_weights.sum())
                brands = [brand_names[i] for i in sorted(picked)]
            else:
                brands = [None]  # sold loose
            for brand in brands:
                for size in sizes:
                    skus.append(Item(" ".join(p for p in (brand, base, size) if p), base, category, brand, size))
    return skus


def make_catalog(n_items=1000, brands_per_product=4, seed=0) -> List[Item]:
    """n_items SKUs, the same every time for the same seed, with every product stocked at least once."""
    rng = np.random.default_rng(seed)
    pool = all_skus(brands_per_product, rng)
    positions = {}  # (category, product) -> where its SKUs sit in the pool
    for i, sku in enumerate(pool):
        positions.setdefault((sku.category, sku.base), []).append(i)
    if n_items < len(positions):
        raise ValueError(f"need at least {len(positions)} items to cover every product, got {n_items}")
    if n_items > len(pool):
        raise ValueError(f"only {len(pool)} SKUs are possible with brands_per_product={brands_per_product}; "
                         f"ask for fewer items or raise brands_per_product")
    chosen = {int(rng.choice(where)) for where in positions.values()}  # one SKU of every product first
    rest = [i for i in range(len(pool)) if i not in chosen]
    chosen |= {int(i) for i in rng.choice(rest, size=n_items - len(chosen), replace=False)}
    return [pool[i] for i in sorted(chosen)]
