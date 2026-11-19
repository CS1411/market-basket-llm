# Market Basket LLM

Working through the askThili drill **Teach a Language Model What Sells Together**: build a small shop with known buying patterns, mine the exact answer, then adapt an open language model (SmolLM2-135M) to the baskets and test what it learned against that answer instead of an impression.

Reading: [Fast Algorithms for Mining Association Rules (Agrawal & Srikant, 1994)](https://www.vldb.org/conf/1994/P487.PDF), [Amazon.com Recommendations: Item-to-Item Collaborative Filtering (Linden, Smith & York, 2003)](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf) and [SmolLM2 (Allal et al., 2025)](https://arxiv.org/abs/2502.02737).

## Progress

- [x] Lesson 1 - A Shop's Catalogue
- [x] Lesson 2 - How Much Each Item Sells
- [x] Lesson 3 - Background Baskets
- [x] Lesson 4 - Plant One Itemset
- [x] Lesson 5 - Plant Many
- [x] Lesson 6 - The Exact Answer
- [x] Lesson 7 - Verbalize a Basket
- [x] Lesson 8 - The Off-the-Shelf Baseline
- [x] Lesson 9 - Continued Pretraining
- [x] Lesson 10 - Probe the Model
- [x] Lesson 11 - From Itemsets to Cross-sell and Upsell
- [x] Completion test (passed on askThili)
- [x] Project: [supply-co-usage-insight](https://github.com/CS1411/supply-co-usage-insight)

## Results (this repo's own runs, 60 test cases)

| method | MRR (95% range) | hits@10 | hits@50 |
|---|---|---|---|
| popularity (ignores the basket) | 0.124 (0.066 to 0.196) | 25% | 45% |
| SmolLM2-135M off the shelf | 0.045 (0.017 to 0.087) | 8% | 32% |
| SmolLM2 after 300 training steps | 0.163 (0.089 to 0.251) | 25% | 43% |
| counting item-to-item | 0.225 (0.139 to 0.332) | 33% | 45% |

- Validation loss fell from 3.479 to 0.568 over 300 steps.
- Probing 2,000 model-written baskets recovers 34% of true itemsets at 0.5-1% support, 72% at 1-2%, 100% at 2-5%; 2,000 real baskets recover 81%, 100%, 100%.
- The right brand/size among the variants of the right product is ranked first 46% of the time (22% by chance).
- Honest reading: training clearly helps over the untrained model, but with 60 cases the ranges overlap popularity and the plain counting recommender is nominally best. At this scale the model did not beat counting.

Outputs of every run are in `lessons/lessonN/output.txt`; short notes are in `notes/`.

## Run it

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cd lessons\lesson6 && python lesson6.py   # builds data/dataset (git-ignored)
cd ..\lesson8 && python lesson8.py        # about 14 min on CPU
cd ..\lesson9 && python lesson9.py        # about 55 min on CPU; saves data/pretrained
cd ..\lesson10 && python lesson10.py      # about 27 min
cd ..\lesson11 && python lesson11.py
```

Lessons 1-7 need only numpy (lesson 7 also the tokenizer); lessons 8-11 need torch and transformers and download SmolLM2-135M from Hugging Face on first use.

## Layout

- `lessons/lessonN/` code and `output.txt` for each lesson
- `notes/` short notes per lesson
- `LICENSE-thilimarketbasket` - licence of the course reference repo, whose shop table (`lessons/lesson1/catalog_data.py`) is copied unchanged

## Honest development notes

- Built with AI assistance (Claude) in a single working session; the commit timeline does not reflect days of work.
- Lessons 6 and later were checked against the course reference repo (thili-ai/thilimarketbasket): the generated dataset and exact truth are identical.
- Runs are small: PyTorch on a laptop CPU instead of the course's Apple GPU, 300 training steps, 60 test cases. Numbers are this run's own, not the course author's, and are noisy.
