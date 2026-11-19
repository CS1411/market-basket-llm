"""Lesson 1: a shop's catalogue. Build the 1,000-SKU catalogue and look at what it holds."""
from collections import Counter

import numpy as np

from catalog import all_skus, make_catalog
from catalog_data import CATEGORIES

if __name__ == "__main__":
    n_products = sum(len(spec["products"]) for spec in CATEGORIES.values())
    print(f"{n_products} products in {len(CATEGORIES)} categories")

    rng = np.random.default_rng(0)
    for k in (1, 2, 4, 6):
        print(f"brands_per_product={k}: {len(all_skus(k, np.random.default_rng(0))):>5,} possible SKUs")

    catalog = make_catalog()
    print(f"\n{len(catalog):,} items in the catalogue; every name unique: {len({i.name for i in catalog}) == len(catalog)}")
    for item in catalog[:3]:
        print(" ", item)
    per_category = Counter(i.category for i in catalog)
    print("\nSKUs per category:")
    for category, n in per_category.most_common():
        print(f"  {category:26s}{n:4d}")
    per_product = Counter((i.category, i.base) for i in catalog)
    print(f"\nSKUs per product: fewest {min(per_product.values())}, median {int(np.median(list(per_product.values())))}, "
          f"most {max(per_product.values())}")
    print(f"loose goods with no brand: {sum(i.brand is None for i in catalog)} SKUs")
    print(f"sachets and small rupee packs: {sum(('sachet' in i.size or i.size.startswith('Rs ')) for i in catalog)} SKUs")

    for bad in (5000, 100):  # it refuses to build a broken shop
        try:
            make_catalog(n_items=bad)
        except ValueError as error:
            print(f"n_items={bad}: ValueError: {error}")
    assert len(catalog) == 1000 and n_products == 127
