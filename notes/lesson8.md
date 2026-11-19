# Lesson 8 - The Off-the-Shelf Baseline

Task: hide one item of a basket, rank all 1,000 candidates by the model's log-probability of the item name, report MRR and hits@K with a bootstrap range. A shared key/value cache for the basket prefix makes scoring 1,000 candidates affordable.

Results (60 cases, my CPU run; random guessing is MRR about 0.0075):

| method | MRR (95% range) | hits@10 | hits@50 |
|---|---|---|---|
| popularity | 0.124 (0.066 to 0.196) | 25% | 45% |
| SmolLM2-135M off the shelf | 0.045 (0.017 to 0.087) | 8% | 32% |

The untrained model is worse than just ranking by popularity. With only 60 cases the ranges are wide, so I only claim the direction. Scoring took about 14 minutes on CPU.
