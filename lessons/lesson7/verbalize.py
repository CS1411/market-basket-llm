"""Lesson 7: turn a basket into text a language model can read.

A basket is a SET: nothing in it comes first. A language model reads left to right, so an order must be chosen, and any
fixed order (alphabetical, the order the till scanned) is something the model can learn that means nothing. So the order
is shuffled afresh every time a basket is shown. Quantities are dropped, as the exact miner ignores them too.
"""
import numpy as np

PREFIX = "Basket: "
SEPARATOR = "; "


def verbalize_basket(items, seed=None, max_items=30):
    """One basket as a string, e.g. "Basket: Onion 1 kg; Gokul Fresh Milk 500 ml; Potato 1 kg".

    Each different item appears once, in a random order (seed makes it repeatable). A basket with more than max_items
    different items is cut to a random max_items.
    """
    different = sorted(set(items))  # sorted first, so the same seed always gives the same order
    order = np.random.default_rng(seed).permutation(len(different))[:max_items]
    return PREFIX + SEPARATOR.join(different[i] for i in order)


def parse_basket(text):
    "The items in a verbalized basket, in the order written. The reverse of verbalize_basket."
    if not text.startswith(PREFIX):
        raise ValueError(f"not a verbalized basket: {text[:40]!r}")
    body = text[len(PREFIX):]
    return body.split(SEPARATOR) if body else []


def verbalize_all(baskets, epochs=1, seed=0, max_items=30):
    "Every basket as text, `epochs` times over, each time in a fresh random order."
    rng = np.random.default_rng(seed)
    return [verbalize_basket(b, seed=int(rng.integers(2**31)), max_items=max_items) for _ in range(epochs) for b in baskets]


def verbalize_prefix(items, seed=None, max_items=30):
    'A basket written up to where the next item goes, e.g. "Basket: Onion 1 kg; Potato 1 kg;" (just "Basket:" if empty).'
    text = verbalize_basket(items, seed=seed, max_items=max_items)
    return PREFIX.rstrip() if text == PREFIX else text + ";"
