"""Lesson 4: plant one itemset. Make a pattern true on purpose, at an exact support."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for lesson in ("lesson1", "lesson2", "lesson3"):
    sys.path.insert(0, str(HERE.parent / lesson))
from baskets import background_baskets  # noqa: E402
from catalog import make_catalog  # noqa: E402
from plant import Planted, find_items, plant_itemset, support_of  # noqa: E402
from popularity import popularity  # noqa: E402

if __name__ == "__main__":
    catalog = make_catalog()
    shares = popularity(catalog)
    baskets = background_baskets(100_000, catalog, shares, seed=0)

    toor = find_items(catalog, "Toor Dal", brand="Kaveri Valley", size="1 kg")[0].name
    rice = find_items(catalog, "Sona Masoori Rice", size="5 kg")[0].name
    oil = find_items(catalog, "Sunflower Oil", size="1 L")[0].name
    items = (toor, rice, oil)
    print("planting:", items)
    print(f"support by chance before planting: {support_of(baskets, items):.4%}")

    new = plant_itemset(baskets, Planted(items, 0.01, "natural"), catalog, seed=1)
    print(f"support after planting at 1%:      {support_of(new, items):.4%}  ({round(0.01 * len(baskets))} baskets, exactly)")
    print("the original baskets are untouched; a copy is returned:", baskets is not new)
    for pair in ((toor, rice), (toor, oil), (rice, oil)):
        print(f"  a part of {len(pair)} items has support {support_of(new, pair):.3%} (at least the whole's)")

    cases = (("too common to plant", Planted((find_items(catalog, "Onion", size="1 kg")[0].name, find_items(catalog, "Potato", size="1 kg")[0].name), 0.0001)),
             ("not in the catalogue", Planted(("nope", toor), 0.01)),
             ("two SKUs of one product", Planted((toor, find_items(catalog, "Toor Dal")[1].name), 0.01)))
    for label, planted in cases:
        try:
            plant_itemset(baskets, planted, catalog)
        except ValueError as error:
            print(f"{label}: ValueError: {str(error)[:90]}")
    assert round(support_of(new, items) * len(new)) == round(0.01 * len(new))
