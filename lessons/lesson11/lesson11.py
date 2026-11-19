"""Lesson 11: from itemsets to cross-sell. Turn the model's scores into suggestions and test them against simple counting.

Reuses the 60 cases and score matrices saved by Lessons 8 and 9, and scores one blank basket with the trained model.
Needs data/pretrained from Lesson 9.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2", "lesson3", "lesson4", "lesson5", "lesson6", "lesson7", "lesson8"):
    sys.path.insert(0, str(HERE.parent / lesson))
from dataset import load_dataset  # noqa: E402
from evaluate import (bootstrap_ci, candidate_token_ids, popularity_scores, rank_of, ranks_from_scores,  # noqa: E402
                      score_candidates, summarise)
from recommend import ItemToItem, lift_adjusted, rank_among, repeat_share, top_k, variants_of  # noqa: E402
from verbalize import verbalize_prefix  # noqa: E402

DATA = HERE.parents[1] / "data" / "dataset"
MODEL_DIR = HERE.parents[1] / "data" / "pretrained"


def row(label, ranks, out):
    s = summarise(ranks)
    lo, hi = bootstrap_ci(1.0 / np.asarray(ranks))
    print(f"{label:38s}{s['mrr']:8.3f} ({lo:.3f} to {hi:.3f}){s['hits@10']:9.0%}{s['hits@50']:9.0%}")
    out[label] = {**s, "mrr_ci": [lo, hi]}


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_dataset(DATA)
    names = [i.name for i in data.catalog]
    position = {n: i for i, n in enumerate(names)}
    cases = json.loads((HERE.parent / "lesson8" / "cases.json").read_text())
    model_scores = np.load(HERE.parent / "lesson9" / "scores.npy")
    train = data.baskets[:90_000]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    blank = score_candidates(model, tok.encode(verbalize_prefix([])), candidate_token_ids(tok, names))
    counting = ItemToItem(train, names)

    results = {}
    print(f"{'method':38s}{'MRR (95% range)':>24s}{'hits@10':>9s}{'hits@50':>9s}")
    pop = np.tile(popularity_scores(train, names), (len(cases), 1))
    row("popularity (ignores the basket)", ranks_from_scores(pop, names, cases), results)
    for lift in (False, True):
        scores = np.array([counting.score(c["visible"], lift=lift) for c in cases])
        row(f"counting item-to-item{' with lift' if lift else ''}", ranks_from_scores(scores, names, cases), results)
    for alpha in (0.0, 0.5, 1.0):
        scores = np.array([lift_adjusted(r, blank, alpha) for r in model_scores])
        row(f"trained model, lift alpha {alpha}", ranks_from_scores(scores, names, cases), results)

    print("\nwhat each method suggests for one basket (hidden item in brackets):")
    case = cases[0]
    visible = [position[n] for n in case["visible"]]
    print("  basket:", ", ".join(case["visible"][:5]), "..." if len(visible) > 5 else "")
    print("  hidden:", case["target"])
    suggestions = {"counting with lift": counting.score(case["visible"], lift=True),
                   "trained model, lift alpha 1": lift_adjusted(model_scores[0], blank, 1.0)}
    for label, scores in suggestions.items():
        chosen = top_k(scores, 3, visible)
        print(f"  {label:28s}", "; ".join(names[i] for i in chosen))
        print(f"  {'':28s} repeats a product already in the basket: {repeat_share(chosen, visible, data.catalog):.0%}")

    groups = variants_of(data.catalog)
    hidden = [position[c["target"]] for c in cases]
    multi = [(k, c) for k, c in enumerate(cases) if len(groups[data.catalog[position[c["target"]]].base]) > 1]
    among = [rank_among(model_scores[k], position[c["target"]], groups[data.catalog[position[c["target"]]].base]) for k, c in multi]
    chance = np.mean([1 / len(groups[data.catalog[position[c["target"]]].base]) for k, c in multi])
    print(f"\nchoosing among the brands and pack sizes of the right product ({len(multi)} cases with several variants):")
    print(f"  the model ranks the right variant first in {np.mean(np.array(among) == 1):.0%} of cases "
          f"(guessing at random: {chance:.0%}); median rank {np.median(among):.0f}")

    (HERE / "results.json").write_text(json.dumps({"n_cases": len(cases), "methods": results,
                                                    "variant_top1": float(np.mean(np.array(among) == 1))}, indent=1))
    assert set(results) >= {"popularity (ignores the basket)", "trained model, lift alpha 1.0"}
