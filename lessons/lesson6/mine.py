"""Lesson 6: the exact answer. Frequent itemsets by Apriori (Agrawal and Srikant, 1994), written twice.

An itemset can only be frequent if every part of it is frequent, so count single items first, build pairs only from
frequent items, triples only from frequent pairs, and so on. The plain version counts with sets; the fast version
gives every item a bitset (one bit per basket) so the number of baskets holding two items is one AND and a bit count.
"""
import math
from itertools import combinations

import numpy as np


def apriori_plain(baskets, min_support, max_size=5):
    "Level by level with Python sets. Clear, and fine for a few thousand baskets."
    n = len(baskets)
    min_count = max(1, math.ceil(min_support * n - 1e-9))
    sets = [set(b) for b in baskets]
    counts = {}
    for basket in sets:
        for name in basket:
            counts[(name,)] = counts.get((name,), 0) + 1
    level = {items: c for items, c in counts.items() if c >= min_count}
    found = {frozenset(i): c / n for i, c in level.items()}
    size = 1
    while level and size < max_size:
        keys = sorted(level)
        candidates = set()
        for a, b in combinations(keys, 2):
            if a[:-1] == b[:-1]:                                   # they differ only in the last item
                candidate = a + (b[-1],)
                if all(candidate[:i] + candidate[i + 1:] in level for i in range(len(candidate))):
                    candidates.add(candidate)                      # every part is frequent
        level = {}
        for candidate in candidates:
            count = sum(set(candidate) <= basket for basket in sets)
            if count >= min_count:
                level[candidate] = count
        found.update({frozenset(i): c / n for i, c in level.items()})
        size += 1
    return found


def bit_count(value):
    return value.bit_count()


def bitsets(baskets):
    "For each item, an integer whose bit i is 1 when basket i holds it (quantity does not matter)."
    where = {}
    for number, basket in enumerate(baskets):
        for name in set(basket):
            where.setdefault(name, []).append(number)
    bits = {}
    for name, numbers in where.items():
        held = np.zeros(len(baskets), dtype=bool)
        held[numbers] = True
        bits[name] = int.from_bytes(np.packbits(held).tobytes(), "big")
    return bits


def frequent_itemsets(baskets, min_support, max_size=5):
    """Every itemset of at most max_size items held by at least min_support of the baskets, with its exact support.

    Quantities are ignored: two packs of milk count as milk.
    """
    n = len(baskets)
    min_count = max(1, math.ceil(min_support * n - 1e-9))  # the allowance guards against 0.0005 * 100000 rounding up
    bits = bitsets(baskets)
    level = {(name,): value for name, value in bits.items() if bit_count(value) >= min_count}
    found = {frozenset(i): bit_count(v) / n for i, v in level.items()}
    size = 1
    while level and size < max_size:
        by_prefix = {}  # itemsets that agree on all but the last item can be joined into a bigger one
        for items in sorted(level):
            by_prefix.setdefault(items[:-1], []).append(items[-1])
        next_level = {}
        for prefix, lasts in by_prefix.items():
            for a, b in combinations(lasts, 2):  # lasts is sorted, so a < b
                candidate = prefix + (a, b)
                if any(candidate[:i] + candidate[i + 1:] not in level for i in range(len(candidate) - 2)):
                    continue
                held = level[prefix + (a,)] & bits[b]
                if bit_count(held) >= min_count:
                    next_level[candidate] = held
        level = next_level
        found.update({frozenset(i): bit_count(v) / n for i, v in level.items()})
        size += 1
    return found


def rules(itemsets, min_confidence=0.3):
    """Association rules: (antecedent, consequent, support, confidence, lift). Lift says how many times more often
    the two sides meet than if they were unrelated (1 means unrelated)."""
    out = []
    for items, support in itemsets.items():
        if len(items) < 2:
            continue
        for k in range(1, len(items)):
            for antecedent in map(frozenset, combinations(sorted(items), k)):
                consequent = items - antecedent
                confidence = support / itemsets[antecedent]
                if confidence >= min_confidence:
                    out.append((antecedent, consequent, support, confidence, confidence / itemsets[consequent]))
    return sorted(out, key=lambda r: (-r[4], -r[2]))
