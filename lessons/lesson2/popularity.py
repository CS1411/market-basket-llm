"""Lesson 2: how much each item sells. One share per SKU, built from four normalised levels."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson1"))
from catalog import split_product  # noqa: E402
from catalog_data import CATEGORIES  # noqa: E402


def fraction(weights, chosen, stocked):
    "The share of `chosen` among the stocked options: its weight over the total weight of what is on the shelf."
    return weights[chosen] / sum(weights[option] for option in stocked)


def level_shares(item, catalog, categories=CATEGORIES):
    "The four fractions for one SKU: category, product, brand, size. Each is a share of its own level."
    spec = categories[item.category]
    here = [it for it in catalog if it.category == item.category]
    product_weight, size_weights = {}, {}
    for product in spec["products"]:
        name, weight, own_sizes = split_product(product)
        product_weight[name] = weight
        size_weights[name] = own_sizes or spec["sizes"]

    category = spec["weight"] / sum(c["weight"] for c in categories.values())
    product = fraction(product_weight, item.base, {it.base for it in here})
    if item.brand is None:  # sold loose: no brand to choose
        brand = 1.0
    else:
        brand = fraction(spec["brands"], item.brand, {it.brand for it in here if it.base == item.base})
    size = fraction(size_weights[item.base], item.size,
                    {it.size for it in here if it.base == item.base and it.brand == item.brand})
    return category, product, brand, size


def popularity(catalog, categories=CATEGORIES):
    "One share per SKU, in catalogue order, adding up to 1: category x product x brand x size."
    shares = np.array([np.prod(level_shares(item, catalog, categories)) for item in catalog])
    return shares / shares.sum()


def head_share(shares, fraction_of_items=0.2):
    "The share of all sales made by the best-selling fraction of items."
    ordered = np.sort(np.asarray(shares))[::-1]
    return float(ordered[: max(1, int(np.ceil(fraction_of_items * len(ordered))))].sum())
