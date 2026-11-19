"""Lesson 5: plant many itemsets. 32 patterns that vary in size, support and kind, and overlap on purpose."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson4"))
from plant import Planted, plant_itemset  # noqa: E402

# Products that really go together in a kitchen or a bathroom: four sets of each size (product names; the SKU of
# each is picked when the itemsets are built).
NATURAL = {
    2: [("Tea Powder", "Sugar"), ("Bread", "Butter"), ("Toothpaste", "Toothbrush"), ("Detergent Powder", "Dishwash Bar")],
    3: [("Toor Dal", "Sona Masoori Rice", "Sunflower Oil"), ("Turmeric Powder", "Red Chilli Powder", "Coriander Powder"),
        ("Milk", "Bread", "Eggs"), ("Shampoo Sachet", "Soap Bar", "Hair Oil")],
    4: [("Wheat Flour (Atta)", "Sugar", "Salt", "Sunflower Oil"), ("Tea Powder", "Milk", "Glucose Biscuits", "Rusk"),
        ("Instant Noodles", "Potato Chips", "Soft Drink", "Namkeen Mixture"),
        ("Dosa Batter", "Coconut Oil", "Curry Leaves", "Mustard Seeds")],
    5: [("Onion", "Tomato", "Green Chilli", "Ginger", "Garlic"),
        ("Basmati Rice", "Biryani Masala", "Ghee", "Onion", "Curd"),
        ("Cornflakes", "Milk", "Banana", "Jam", "Bread"),
        ("Detergent Bar", "Phenyl", "Floor Cleaner", "Toilet Cleaner", "Garbage Bags")],
}
SUPPORTS = (0.05, 0.01, 0.002, 0.0005)  # from common to nearly invisible: the rarest is 50 baskets in 100,000


def item_index(baskets):
    "For each item, the set of basket numbers that hold it, so we can count how often items meet by chance."
    index = {}
    for number, basket in enumerate(baskets):
        for name in set(basket):
            index.setdefault(name, set()).add(number)
    return index


def chance_support(index, items, n_baskets):
    held = [index.get(name, set()) for name in items]
    return len(set.intersection(*sorted(held, key=len))) / n_baskets


def default_planted(catalog, shares, background, seed=0, supports=SUPPORTS):
    """32 itemsets: for each of four supports, a natural and a surprising one of each size 2 to 5.

    `background` is a sample of ordinary baskets used to check no itemset turns up by chance often enough to spoil
    its target. Rules: a product keeps ONE SKU across every itemset (overlaps are deliberate and one itemset can
    never swap away another's item); no itemset contains another; chance alone gives at most 30% of the target; and
    the natural sets that are commoner by chance get the higher supports.
    """
    rng = np.random.default_rng(seed)
    shares = np.asarray(shares)
    by_base, category_of = {}, {}
    for position, item in enumerate(catalog):
        by_base.setdefault(item.base, []).append(position)
        category_of[item.base] = item.category
    index, n = item_index(background), len(background)
    chosen = {}  # product -> the SKU it uses in every itemset

    def sku_for(base):
        if base not in chosen:
            positions = np.array(by_base[base])
            chosen[base] = catalog[int(rng.choice(positions, p=shares[positions] / shares[positions].sum()))].name
        return chosen[base]

    def fits(items, support, existing):
        if chance_support(index, items, n) > 0.3 * support:
            return False
        return not any(set(items) <= set(o.items) or set(o.items) <= set(items) for o in existing)

    planted, descending = [], sorted(supports, reverse=True)
    for size in sorted(NATURAL, reverse=True):
        groups = [tuple(sku_for(base) for base in group) for group in NATURAL[size]]
        for support, items in zip(descending, sorted(groups, key=lambda g: -chance_support(index, g, n))):
            if not fits(items, support, planted):
                raise ValueError(f"the natural set {items} is too common by chance, or overlaps another, at support {support}")
            planted.append(Planted(items, support, "natural"))
        for support in descending:
            for _ in range(500):
                categories = list(rng.choice(sorted(set(category_of.values())), size=size, replace=False))
                bases = [str(rng.choice([b for b in by_base if category_of[b] == c])) for c in categories]
                items = tuple(sku_for(base) for base in bases)
                if fits(items, support, planted):
                    planted.append(Planted(items, support, "surprising"))
                    break
            else:
                raise ValueError(f"no surprising set of size {size} fits at support {support}")
    return sorted(planted, key=lambda p: (-p.support, len(p.items), p.kind))


def plant_all(baskets, planted, catalog, seed=0):
    """Plant every itemset in turn, larger ones first. Itemsets may share items, but a product must use the same SKU
    everywhere and no itemset may contain another. Planting one can complete another by accident, so the realised
    supports are close to the targets rather than exact (see planted_report)."""
    product_of = {i.name: i.base for i in catalog}
    used = {}
    for itemset in planted:
        for name in itemset.items:
            if name not in product_of:
                raise ValueError(f"not in the catalogue: {name}")
            if used.setdefault(product_of[name], name) != name:
                raise ValueError(f"{product_of[name]} appears as both {used[product_of[name]]} and {name}")
    sets = [set(p.items) for p in planted]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j and a <= b:
                raise ValueError(f"{planted[i].items} is contained in {planted[j].items}")
    out = list(baskets)
    # Larger first: planting a small itemset afterwards rarely completes a big one by accident; the other way
    # round it often does.
    order = sorted(range(len(planted)), key=lambda i: (-len(planted[i].items), -planted[i].support))
    for step, i in enumerate(order):
        out = plant_itemset(out, planted[i], catalog, seed=seed + step, allow_excess=True)
    return out


def planted_report(baskets, planted):
    "For each planted itemset: its kind, target support, and the support the baskets really have."
    index, n = item_index(baskets), len(baskets)
    return [{"items": p.items, "kind": p.kind, "size": len(p.items), "target": p.support,
             "realised": chance_support(index, p.items, n)} for p in planted]
