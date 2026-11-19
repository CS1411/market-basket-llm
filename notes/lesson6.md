# Lesson 6 - The Exact Answer

Level-wise Apriori counted with bitsets (Python ints, `int.bit_count`), plus association rules (confidence, lift).

- On 3,000 baskets at 1%: plain Apriori 10.75 s, bitsets 0.027 s, identical results.
- On the full data at min support 0.05%: 56,470 frequent itemsets (977 single, 303 "part" of planted, 32 planted, 23,256 chance, 31,902 extensions). All 32 planted sets are found.
- Dataset generation plus mining takes 16 s. It matches the course reference repo exactly (checked with a scratch comparison script).
- Strongest rules have lift around 1,850: planted sets are far above chance.

Takeaway: this miner is the answer key every model is scored against.
