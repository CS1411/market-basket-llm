"""Lesson 3: background baskets. Ordinary shopping trips, and how much 'together' you get by chance alone."""
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2"):
    sys.path.insert(0, str(HERE.parent / lesson))
from baskets import background_baskets  # noqa: E402
from catalog import make_catalog  # noqa: E402
from popularity import popularity  # noqa: E402

if __name__ == "__main__":
    catalog = make_catalog()
    shares = popularity(catalog)
    baskets = background_baskets(100_000, catalog, shares, seed=0)
    distinct = np.array([len(set(b)) for b in baskets])
    lines = np.array([len(b) for b in baskets])
    print(f"{len(baskets):,} baskets; different items per basket: mean {distinct.mean():.2f}, "
          f"median {int(np.median(distinct))}, max {distinct.max()}; lines per basket: mean {lines.mean():.2f}")
    print("example basket:", baskets[0][:6], "...")
    quantities = Counter(q for b in baskets[:20000] for q in Counter(b).values())
    total = sum(quantities.values())
    print("quantity per item:", {q: f"{quantities[q] / total:.1%}" for q in sorted(quantities)})

    item_baskets = {}  # for each item, the set of baskets that hold it
    for number, basket in enumerate(baskets):
        for name in set(basket):
            item_baskets.setdefault(name, set()).add(number)
    top = sorted(item_baskets, key=lambda n: -len(item_baskets[n]))[:30]
    best = max((len(item_baskets[a] & item_baskets[b]) / len(baskets), a, b) for a, b in combinations(top, 2))
    print(f"\nmost common item: {top[0]} in {len(item_baskets[top[0]]) / len(baskets):.1%} of baskets")
    print(f"the most common pair among the top 30 meets by chance in {best[0]:.2%} of baskets: {best[1]} + {best[2]}")
    same = background_baskets(2000, catalog, shares, seed=3) == background_baskets(2000, catalog, shares, seed=3)
    print("same seed, same baskets:", same)
    assert same and 7 < distinct.mean() < 10
