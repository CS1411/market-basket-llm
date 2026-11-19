"""Lesson 3: background baskets. Ordinary shopping trips, nothing planted. A basket is a list of item names."""
import numpy as np


def basket_sizes(n_baskets, rng, mean=8.0, dispersion=4.0, minimum=2):
    "How many different products each basket holds: most trips are small, a few are big (a negative binomial tail)."
    extra_mean = mean - minimum
    p = dispersion / (dispersion + extra_mean)
    return minimum + rng.negative_binomial(dispersion, p, size=n_baskets)


def draw_items(sizes, shares, rng, chunk=2000):
    """Pick sizes[i] different things for basket i, favouring the best sellers, without repeats.

    The Gumbel trick: score every candidate with log(share) plus Gumbel noise and keep the highest scores.
    That equals drawing one at a time and removing it, but handles a whole chunk of baskets at once.
    """
    log_share = np.log(np.asarray(shares))
    out = []
    for start in range(0, len(sizes), chunk):
        sizes_here = np.asarray(sizes[start:start + chunk])
        top = int(sizes_here.max())
        scores = log_share + rng.gumbel(size=(len(sizes_here), len(shares)))
        best = np.argpartition(-scores, top - 1, axis=1)[:, :top]          # the `top` best per basket
        order = np.argsort(-np.take_along_axis(scores, best, axis=1), axis=1)
        best = np.take_along_axis(best, order, axis=1)
        out += [row[:k] for row, k in zip(best, sizes_here)]
    return out


class Products:
    """The catalogue grouped by product, so a shopper picks a product first and a brand and size second."""

    def __init__(self, catalog, shares):
        shares = np.asarray(shares)
        groups = {}
        for i, item in enumerate(catalog):
            groups.setdefault((item.category, item.base), []).append(i)
        groups = [np.array(g) for g in groups.values()]
        self.share = np.array([shares[g].sum() for g in groups])      # a product's share is its SKUs added up
        self._sku = np.concatenate(groups)                            # SKU positions, product by product
        within = [shares[g] / shares[g].sum() for g in groups]        # odds of each SKU inside its product
        # A running total of those odds with the product number added gives every product its own 0-to-1 stretch,
        # so one sorted lookup finds the SKU for any (product, random number) pair.
        self._cdf = np.concatenate([np.cumsum(w) + k for k, w in enumerate(within)])
        self._cdf[np.cumsum([len(g) for g in groups]) - 1] = np.arange(1, len(groups) + 1)

    def pick_sku(self, products, rng):
        "One SKU (a catalogue position) for each product number, chosen by its odds within the product."
        products = np.asarray(products)
        return self._sku[np.searchsorted(self._cdf, products + rng.random(len(products)))]


def background_baskets(n_baskets, catalog, shares, seed=0, mean_size=8.0, quantity_probs=(0.80, 0.14, 0.04, 0.02)):
    """n_baskets ordinary baskets: which products (by popularity), which brand and size of each, how many of each.

    A basket never holds two brands or sizes of one product. Quantities of 1 to 4 repeat the name, so
    set(basket) is what was bought and Counter(basket) is how many. Items come out in random order.
    """
    rng = np.random.default_rng(seed)
    names = [item.name for item in catalog]
    products = Products(catalog, shares)
    sizes = basket_sizes(n_baskets, rng, mean=mean_size)
    chosen = draw_items(sizes, products.share, rng)
    skus = products.pick_sku(np.concatenate(chosen), rng)
    counts = rng.choice(np.arange(1, len(quantity_probs) + 1), size=len(skus), p=quantity_probs)
    baskets, at = [], 0
    for k in sizes:
        lines = [names[i] for i, q in zip(skus[at:at + k], counts[at:at + k]) for _ in range(q)]
        baskets.append([lines[j] for j in rng.permutation(len(lines))])
        at += k
    return baskets
