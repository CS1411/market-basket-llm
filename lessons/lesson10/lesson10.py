"""Lesson 10: probe the model. Have it write baskets, mine them, and score recovery against the exact truth.

The model writes 2,000 baskets (about 10 minutes on a laptop CPU). A ceiling shows what ANY 2,000 baskets could recover:
2,000 real baskets, mined the same way. Needs data/pretrained from Lesson 9.
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
from mine import frequent_itemsets  # noqa: E402
from probe import precision_of, recover_itemsets, recovery_by_support, sample_baskets, support_error  # noqa: E402

N_SAMPLES = 2000
MIN_SUPPORT = 0.005  # 10 of 2,000 baskets: the lowest support we can judge with this many samples
BANDS = [0.005, 0.01, 0.02, 0.05, 1.0]
DATA = HERE.parents[1] / "data" / "dataset"
MODEL_DIR = HERE.parents[1] / "data" / "pretrained"


def table(rows, title):
    print(f"\n{title}\n{'true support':>20s} {'true':>6s} {'found':>6s} {'recall':>7s}")
    for r in rows:
        print(f"{r['band']:>20s} {r['true']:6d} {r['found']:6d} {r['recall']:7.0%}")


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_dataset(DATA)
    names = [i.name for i in data.catalog]
    truth = {s: v for s, v in data.truth.items() if v >= MIN_SUPPORT}
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()

    start = time.time()
    sampled, stats = sample_baskets(model, tok, names, N_SAMPLES, seed=0)
    print(f"wrote {N_SAMPLES} baskets in {time.time() - start:.0f} s")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    recovered = recover_itemsets(sampled, MIN_SUPPORT)
    ceiling_baskets = [data.baskets[i] for i in np.random.default_rng(7).choice(len(data.baskets), N_SAMPLES, replace=False)]
    ceiling = frequent_itemsets(ceiling_baskets, MIN_SUPPORT)
    print(f"\ntrue itemsets with support >= {MIN_SUPPORT:.1%}: {len(truth):,}")
    print(f"the model's baskets: {len(recovered):,} frequent itemsets; {N_SAMPLES} real baskets: {len(ceiling):,}")

    model_rows = recovery_by_support(recovered, truth, BANDS)
    ceiling_rows = recovery_by_support(ceiling, truth, BANDS)
    table(model_rows, "recovered from the model's baskets, by true support")
    table(ceiling_rows, f"ceiling: recovered from {N_SAMPLES} real baskets")
    print(f"\nprecision (of what the model calls frequent, the share that truly is): {precision_of(recovered, truth, MIN_SUPPORT):.0%}"
          f"  (ceiling {precision_of(ceiling, truth, MIN_SUPPORT):.0%})")
    err = support_error(recovered, truth)
    print(f"support error on shared itemsets: mean {err.mean():+.4f}, mean absolute {np.abs(err).mean():.4f}")

    print("\nthe 32 planted itemsets, by target support and kind (found in the model's baskets / in the ceiling):")
    for support in sorted({p.support for p in data.planted}, reverse=True):
        for kind in ("natural", "surprising"):
            group = [p for p in data.planted if p.support == support and p.kind == kind]
            m = sum(frozenset(p.items) in recovered for p in group)
            c = sum(frozenset(p.items) in ceiling for p in group)
            print(f"  support {support:6.2%} {kind:10s} {m}/{len(group)} (ceiling {c}/{len(group)})")

    (HERE / "results.json").write_text(json.dumps({"n_samples": N_SAMPLES, "min_support": MIN_SUPPORT, "stats": stats,
                                                    "model_recovery": model_rows, "ceiling_recovery": ceiling_rows}, indent=1))
    assert stats["baskets"] == N_SAMPLES
