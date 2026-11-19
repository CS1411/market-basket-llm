"""Lesson 6: the exact answer. Apriori twice (plain and bitset), and the truth every model is scored against."""
import sys
import time
from collections import Counter
from pathlib import Path

from dataset import generate_dataset, save_dataset
from mine import apriori_plain, frequent_itemsets, rules

if __name__ == "__main__":
    six = [["tea", "sugar", "milk"], ["tea", "sugar"], ["bread", "butter", "milk"], ["tea", "milk"],
           ["bread", "butter"], ["tea", "sugar", "bread"]]
    plain = apriori_plain(six, 0.5)
    print("six baskets, min support 0.5:", {" + ".join(sorted(k)): round(v, 2) for k, v in sorted(plain.items(), key=lambda kv: sorted(kv[0]))})
    assert plain == frequent_itemsets(six, 0.5)

    start = time.time()
    data = generate_dataset()
    print(f"\ngenerated the dataset and mined the exact truth in {time.time() - start:.0f} s")
    small = data.baskets[:3000]  # the plain miner agrees with the bitset miner (it is far too slow for 100,000 baskets)
    t = time.time()
    a = apriori_plain(small, 0.01)
    plain_time = time.time() - t
    t = time.time()
    b = frequent_itemsets(small, 0.01)
    print(f"on 3,000 baskets at 1%: plain {plain_time:.2f} s, bitsets {time.time() - t:.3f} s, identical: {a == b}")
    assert a == b

    truth = data.labelled_truth()
    print(f"\n{len(data.baskets):,} baskets, {len(data.planted)} planted itemsets, min support {data.min_support:.2%}")
    print(f"exact truth: {len(truth):,} frequent itemsets")
    by_label = Counter(t["label"] for t in truth)
    by_size = Counter(t["size"] for t in truth)
    print("  by label:", dict(by_label))
    print("  by size: ", dict(sorted(by_size.items())))
    planted_found = sum(frozenset(p.items) in data.truth for p in data.planted)
    print(f"  all {planted_found} of {len(data.planted)} planted itemsets are in the truth")
    assert planted_found == len(data.planted)

    print("\nthe strongest rules (by lift):")
    for ante, cons, support, confidence, lift in rules(data.truth, 0.3)[:3]:
        print(f"  {' + '.join(sorted(ante))[:44]} -> {' + '.join(sorted(cons))[:44]}  conf {confidence:.0%} lift {lift:.0f}")

    out = save_dataset(data, Path(__file__).resolve().parents[2] / "data" / "dataset")
    print(f"\nsaved to {out.parent.name}/{out.name}/ (git-ignored)")
