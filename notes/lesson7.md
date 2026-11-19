# Lesson 7 - Verbalize a Basket

A basket becomes text: `Basket: item; item; item`, quantities dropped, order shuffled per basket.

- Shuffling matters: over 5,000 baskets the first item has 789 distinct values when shuffled versus 376 when alphabetical, so a model cannot learn "the first item is always the earliest letter".
- With the SmolLM2-135M tokenizer a basket is about 109 tokens (median 102, max 344); one item name is about 13 tokens.

Takeaway: order is noise here (a basket is a set), so the text must not teach an order.
