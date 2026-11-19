"""The finished dataset: baskets, the planted itemsets, and the exact truth, labelled by where each itemset comes from."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2", "lesson3", "lesson4", "lesson5"):
    sys.path.insert(0, str(HERE.parent / lesson))
from baskets import background_baskets  # noqa: E402
from catalog import Item, make_catalog  # noqa: E402
from many import default_planted, plant_all, planted_report  # noqa: E402
from mine import frequent_itemsets  # noqa: E402
from plant import Planted  # noqa: E402
from popularity import popularity  # noqa: E402

LABELS = ("single", "planted", "part", "extension", "chance")


def classify_itemset(itemset, planted):
    """Why is this itemset frequent?

    single: one item. planted: one we planted. part: a smaller piece of a planted itemset (every part is at least as
    common as the whole). extension: two or more items of a planted itemset plus others (the planted baskets also
    hold whatever else the shopper bought). chance: common items that meet by chance, no planted pair inside.
    """
    itemset = frozenset(itemset)
    if len(itemset) == 1:
        return "single"
    sets = [frozenset(p.items) for p in planted]
    if itemset in sets:
        return "planted"
    if any(itemset < s for s in sets):
        return "part"
    if any(len(itemset & s) >= 2 for s in sets):
        return "extension"
    return "chance"


@dataclass
class Dataset:
    catalog: List[Item]
    shares: np.ndarray
    planted: List[Planted]
    baskets: List[List[str]]
    truth: Dict[frozenset, float]  # every frequent itemset in the baskets, exact, with its support
    min_support: float
    seed: int

    def label(self, itemset):
        return classify_itemset(itemset, self.planted)

    def labelled_truth(self):
        return [{"items": sorted(i), "support": s, "size": len(i), "label": self.label(i)}
                for i, s in sorted(self.truth.items(), key=lambda kv: (-kv[1], sorted(kv[0])))]


def generate_dataset(n_baskets=100_000, seed=0, min_support=0.0005, max_size=5):
    "Baskets with 32 planted itemsets, and the exact truth. min_support cannot be above the rarest planted (0.05%)."
    if n_baskets < 10_000:
        raise ValueError("need at least 10,000 baskets: the rarest itemset is planted in 0.05% of them")
    catalog = make_catalog(seed=seed)
    shares = popularity(catalog)
    sample = background_baskets(200_000, catalog, shares, seed=seed + 10_000)  # to check chance co-occurrence
    planted = default_planted(catalog, shares, sample, seed=seed)
    if min_support > min(p.support for p in planted):
        raise ValueError(f"min_support {min_support} is above the rarest planted support")
    baskets = plant_all(background_baskets(n_baskets, catalog, shares, seed=seed), planted, catalog, seed=seed)
    return Dataset(catalog, shares, planted, baskets, frequent_itemsets(baskets, min_support, max_size), min_support, seed)


def save_dataset(data, directory):
    "Plain files (JSON, and one basket per line) so any tool can read them."
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    realised = {r["items"]: r["realised"] for r in planted_report(data.baskets, data.planted)}
    (directory / "catalog.json").write_text(json.dumps(
        [{"name": i.name, "base": i.base, "category": i.category, "brand": i.brand, "size": i.size, "share": float(s)}
         for i, s in zip(data.catalog, data.shares)], indent=1))
    (directory / "planted.json").write_text(json.dumps(
        [{"items": list(p.items), "support": p.support, "kind": p.kind, "realised": realised[p.items]}
         for p in data.planted], indent=1))
    truth = data.labelled_truth()
    (directory / "truth.json").write_text(json.dumps(truth))
    (directory / "baskets.jsonl").write_text("\n".join(json.dumps(b) for b in data.baskets) + "\n")
    (directory / "meta.json").write_text(json.dumps({
        "seed": data.seed, "n_baskets": len(data.baskets), "min_support": data.min_support,
        "n_items": len(data.catalog), "n_planted": len(data.planted), "n_truth": len(truth),
        "truth_by_label": {l: sum(1 for t in truth if t["label"] == l) for l in LABELS}}, indent=1))
    return directory


def load_dataset(directory):
    "Read back what save_dataset wrote."
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    rows = json.loads((directory / "catalog.json").read_text())
    catalog = [Item(r["name"], r["base"], r["category"], r["brand"], r["size"]) for r in rows]
    planted = [Planted(tuple(r["items"]), r["support"], r["kind"]) for r in json.loads((directory / "planted.json").read_text())]
    baskets = [json.loads(line) for line in (directory / "baskets.jsonl").read_text().splitlines()]
    truth = {frozenset(r["items"]): r["support"] for r in json.loads((directory / "truth.json").read_text())}
    return Dataset(catalog, np.array([r["share"] for r in rows]), planted, baskets, truth, meta["min_support"], meta["seed"])
