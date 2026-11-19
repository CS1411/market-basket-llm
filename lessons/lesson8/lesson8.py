"""Lesson 8: the off-the-shelf baseline. What does SmolLM2-135M already know about our baskets, before it reads any?

Hide one item in each of 60 baskets, show the model the rest, and rank all 1,000 catalogue items. Scored with MRR and
hits@K, each with a bootstrap range, and set against a baseline that ignores the basket and just ranks by popularity.
About 9 minutes on a laptop CPU. Writes results.json and scores.npy (reused by Lessons 9 and 11).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2", "lesson3", "lesson4", "lesson5", "lesson6", "lesson7"):
    sys.path.insert(0, str(HERE.parent / lesson))
from dataset import load_dataset  # noqa: E402
from evaluate import (bootstrap_ci, make_cases, popularity_scores, ranks_from_scores, score_matrix,  # noqa: E402
                      summarise)

MODEL = "HuggingFaceTB/SmolLM2-135M"
N_CASES = 60
DATA = HERE.parents[1] / "data" / "dataset"


def report(label, ranks):
    s = summarise(ranks)
    lo, hi = bootstrap_ci(1.0 / np.asarray(ranks))
    print(f"{label:34s} MRR {s['mrr']:.3f} ({lo:.3f} to {hi:.3f})  median rank {s['median_rank']:.0f}  "
          f"hits@1 {s['hits@1']:.0%}  hits@10 {s['hits@10']:.0%}  hits@50 {s['hits@50']:.0%}")
    return {**s, "mrr_ci": [lo, hi]}


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_dataset(DATA)
    names = [i.name for i in data.catalog]
    cases = make_cases(data.baskets, names, N_CASES, seed=1)
    train = data.baskets[:90_000]  # the popularity baseline only counts these
    print(f"{N_CASES} cases, {len(names)} candidates each; random guessing would give MRR about "
          f"{np.mean(1 / np.arange(1, len(names) + 1)):.4f}\n")

    pop = np.tile(popularity_scores(train, names), (len(cases), 1))
    results = {"popularity": report("popularity (ignores the basket)", ranks_from_scores(pop, names, cases))}

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    start = time.time()
    scores = score_matrix(model, tok, names, cases, log=print)
    print(f"scored in {time.time() - start:.0f} s\n")
    results["off_the_shelf"] = report("SmolLM2-135M, off the shelf", ranks_from_scores(scores, names, cases))

    np.save(HERE / "scores.npy", scores)
    (HERE / "cases.json").write_text(json.dumps(cases))
    (HERE / "results.json").write_text(json.dumps({"model": MODEL, "n_cases": N_CASES, **results}, indent=1))
    assert scores.shape == (N_CASES, len(names))
