# Lesson 11 - From Itemsets to Cross-sell and Upsell

Scores become suggestions: lift adjustment (subtract alpha times the no-basket score), top-k excluding items in the basket, an item-to-item counting recommender as the baseline, ranking among the variants of one product.

Same 60 cases:

| method | MRR (95% range) | hits@10 |
|---|---|---|
| popularity | 0.124 (0.066 to 0.196) | 25% |
| counting item-to-item | 0.225 (0.139 to 0.332) | 33% |
| counting with lift | 0.128 (0.058 to 0.218) | 13% |
| trained model, alpha 0 | 0.163 (0.089 to 0.251) | 25% |
| trained model, alpha 0.5 | 0.164 (0.085 to 0.259) | 18% |
| trained model, alpha 1 | 0.046 (0.008 to 0.104) | 7% |

- The simple counting recommender is best, nominally above the trained model; ranges overlap, so the honest claim is "the model did not beat counting at this scale".
- Full lift (alpha 1) hurts both methods: dividing by popularity promotes rare items that rarely get bought.
- Choosing the right brand/pack size among the variants of the right product: first in 46% of 59 cases versus 22% by chance, median rank 2.

Takeaway: build the counting baseline first; a small briefly-trained model has to earn its place against it.
