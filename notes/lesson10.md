# Lesson 10 - Probe the Model

The model writes 2,000 baskets from the bare prompt `Basket:` (1,591 s on CPU); the exact miner counts what appears together; recovery is scored against the true itemsets with support of at least 0.5%. A ceiling is 2,000 real baskets mined the same way.

- 5.5% of written items are not in the catalogue, 1.0% repeated, 3.4% of baskets cut off; mean 9.1 items (real: about 8).
- Recall by true support: 0.5-1%: 34% (ceiling 81%); 1-2%: 72% (100%); 2-5%: 100% (100%); 5%+: 61% (100%).
- Precision 69% (ceiling 65%); mean support error -0.0021.
- Of 32 planted itemsets the model's baskets contain 3 (all at 5% or 1%); the ceiling finds all 16 at 5% and 1%. At 0.2% and 0.05% nothing can be found from 2,000 baskets, even the ceiling.

Takeaway: the model learned common structure, and under-represents the rare tail. The ceiling matters: some misses are the sample size, not the model.
