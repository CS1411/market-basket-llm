"""Lesson 2: how much each item sells. A long tail of sales, built level by level."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson1"))
from catalog import make_catalog  # noqa: E402
from catalog_data import CATEGORIES  # noqa: E402
from popularity import head_share, level_shares, popularity  # noqa: E402

if __name__ == "__main__":
    catalog = make_catalog()
    shares = popularity(catalog)
    print(f"shares add up to {shares.sum():.6f}\n")

    sku = next(i for i in catalog if i.name == "Kaveri Valley Toor Dal 1 kg")
    c, p, b, s = level_shares(sku, catalog)
    print(sku.name)
    print(f"  category share {c:.4f}   Pulses is {c:.1%} of the shop's sales")
    print(f"  product share  {p:.4f}   Toor Dal is {p:.1%} of Pulses")
    print(f"  brand share    {b:.4f}   Kaveri Valley is {b:.1%} of the Toor Dal that is stocked")
    print(f"  size share     {s:.4f}   1 kg is {s:.1%} of Kaveri Valley Toor Dal")
    print(f"  SKU share      {c * p * b * s:.5f} = {c * p * b * s:.2%} of all sales\n")

    order = np.argsort(-shares)
    print("the ten best sellers:")
    for i in order[:10]:
        print(f"  {catalog[i].name:36s}{shares[i]:.2%}")
    print(f"best seller / slowest seller: {shares[order[0]] / shares[order[-1]]:.0f} to 1")
    ranked = shares[order]
    for top in (0.01, 0.05, 0.10, 0.20, 0.50):
        print(f"top {top:>4.0%} of SKUs ({int(top * len(ranked)):>3} items) make {head_share(shares, top):.0%} of sales")
    cumulative = np.cumsum(ranked)
    print(f"{int((cumulative < 0.5).sum() + 1)} SKUs make half the sales, {int((cumulative < 0.8).sum() + 1)} make 80%")

    sachet = np.array([("sachet" in i.size or i.size.startswith("Rs ")) for i in catalog])
    print(f"\n{sachet.sum()} sachet and rupee-pack SKUs ({sachet.mean():.1%} of the catalogue) make {shares[sachet].sum():.1%} of sales")

    edited = copy.deepcopy(CATEGORIES)
    edited["Pulses"]["brands"]["Surya Grains"] = 16  # a leading brand doubles its weight
    after = popularity(catalog, edited)
    surya = np.array([i.brand == "Surya Grains" and i.category == "Pulses" for i in catalog])
    toor = np.array([i.base == "Toor Dal" for i in catalog])
    print(f"Surya Grains pulses {shares[surya].sum():.2%} -> {after[surya].sum():.2%}; "
          f"Toor Dal as a whole {shares[toor].sum():.2%} -> {after[toor].sum():.2%} (unchanged: levels are normalised separately)")
    assert abs(shares.sum() - 1) < 1e-9 and abs(shares[toor].sum() - after[toor].sum()) < 1e-9
