"""Lesson 9: continued pretraining. Phase 1 in miniature: read our baskets, then re-take the Lesson 8 test.

300 steps of batch 16 (4,800 baskets, about 5% of the data seen once) on a laptop CPU: roughly 50 minutes. The course's
own run is longer and on an Apple GPU, so treat these as small-scale measurements of this recipe, not of the idea.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2", "lesson3", "lesson4", "lesson5", "lesson6", "lesson7", "lesson8"):
    sys.path.insert(0, str(HERE.parent / lesson))
from dataset import load_dataset  # noqa: E402
from evaluate import bootstrap_ci, ranks_from_scores, score_matrix, summarise  # noqa: E402
from pretrain import pretrain  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
DATA = HERE.parents[1] / "data" / "dataset"
OUT = HERE.parents[1] / "data" / "pretrained"  # git-ignored (about 540 MB)

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_dataset(DATA)
    names = [i.name for i in data.catalog]
    train, val = data.baskets[:90_000], data.baskets[90_000:90_200]
    history = pretrain(MODEL, train, OUT, val_baskets=val, steps=300, batch_size=16, learning_rate=2e-4, warmup=20,
                       eval_every=50)
    first, last = history[0]["val_loss"], history[-1]["val_loss"]
    print(f"\nvalidation loss {first:.3f} -> {last:.3f} (perplexity {np.exp(first):.1f} -> {np.exp(last):.1f}); "
          f"{history[-1]['tokens']:,} tokens in {history[-1]['seconds'] / 60:.0f} min")

    cases = json.loads((HERE.parent / "lesson8" / "cases.json").read_text())
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.float32).eval()
    start = time.time()
    scores = score_matrix(model, tok, names, cases, log=print)
    print(f"scored the same {len(cases)} cases in {time.time() - start:.0f} s")
    ranks = ranks_from_scores(scores, names, cases)
    s = summarise(ranks)
    lo, hi = bootstrap_ci(1.0 / np.asarray(ranks))
    before = json.loads((HERE.parent / "lesson8" / "results.json").read_text())["off_the_shelf"]
    print(f"\n{'':26s}{'MRR':>8s}{'hits@10':>9s}{'hits@50':>9s}")
    print(f"{'off the shelf (Lesson 8)':26s}{before['mrr']:8.3f}{before['hits@10']:9.0%}{before['hits@50']:9.0%}")
    print(f"{'after pretraining':26s}{s['mrr']:8.3f}{s['hits@10']:9.0%}{s['hits@50']:9.0%}   MRR range {lo:.3f} to {hi:.3f}")

    np.save(HERE / "scores.npy", scores)
    (HERE / "results.json").write_text(json.dumps({"model": MODEL, "steps": 300, "batch_size": 16, "history": history,
                                                    "after": {**s, "mrr_ci": [lo, hi]}}, indent=1))
    assert last < first
