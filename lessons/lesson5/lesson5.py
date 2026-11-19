"""Lesson 5: plant many. 32 itemsets (4 sizes x 4 supports x 2 kinds) in 100,000 baskets, and what really got planted."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2", "lesson3", "lesson4"):
    sys.path.insert(0, str(HERE.parent / lesson))
from baskets import background_baskets  # noqa: E402
from catalog import make_catalog  # noqa: E402
from many import default_planted, plant_all, planted_report  # noqa: E402
from popularity import popularity  # noqa: E402

if __name__ == "__main__":
    catalog = make_catalog()
    shares = popularity(catalog)
    sample = background_baskets(200_000, catalog, shares, seed=10_000)  # only to check chance co-occurrence
    planted = default_planted(catalog, shares, sample, seed=0)
    print(f"{len(planted)} itemsets: sizes {sorted({len(p.items) for p in planted})}, "
          f"supports {sorted({p.support for p in planted}, reverse=True)}, kinds {sorted({p.kind for p in planted})}")

    baskets = plant_all(background_baskets(100_000, catalog, shares, seed=0), planted, catalog, seed=0)
    report = planted_report(baskets, planted)
    print(f"\n{'size':>4} {'kind':10} {'target':>8} {'realised':>9}  items")
    for row in report:
        print(f"{row['size']:>4} {row['kind']:10} {row['target']:8.2%} {row['realised']:9.3%}  "
              f"{' + '.join(n.split()[0] + '..' for n in row['items'])}")
    worst = max(abs(r["realised"] - r["target"]) / r["target"] for r in report)
    print(f"\nlargest relative gap between realised and target support: {worst:.0%}")
    assert len(planted) == 32 and len({(len(p.items), p.support, p.kind) for p in planted}) == 32
