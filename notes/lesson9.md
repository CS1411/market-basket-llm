# Lesson 9 - Continued Pretraining

Hand-written loop: masked next-token cross-entropy, AdamW, 20-step warmup then cosine decay, gradient clipping, float32 on CPU. 300 steps of batch 16 (4,800 baskets, about 5% of the data seen once), learning rate 2e-4, 523,222 tokens, 40 minutes. The course trains longer on an Apple GPU, so these are small-scale numbers from this recipe.

- Validation loss 3.479 to 0.568 (perplexity 32.4 to 1.8).
- Same 60 cases: MRR 0.045 to 0.163 (range 0.089 to 0.251), hits@10 8% to 25%, hits@50 32% to 43%.
- 0.163 is above popularity (0.124) but the ranges overlap, so this is a clear gain over the untrained model, not a proven win over popularity.

Takeaway: a few hundred steps teach the format and the item vocabulary; learning real co-purchase structure takes more.
