# Lesson 4 - Plant One Itemset

Planting forces an itemset into exactly round(support x n) baskets by swapping items in, keeping one SKU per product, and returns a copy (the originals are untouched).

- Demo: Kaveri Valley Toor Dal 1 kg + Surya Grains Sona Masoori Rice 5 kg + Dhara Fresh Sunflower Oil 1 L went from 0.0030% by chance to exactly 1.0000% (1,000 baskets).
- Every part of the planted set has support at least as high as the whole (monotone property).
- Refuses bad requests: an itemset already present by chance in too many baskets (Onion + Potato: 1,309), items not in the catalogue, two SKUs of one product.

Takeaway: planting gives a known right answer to score against.
