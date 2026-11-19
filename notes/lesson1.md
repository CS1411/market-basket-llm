# Lesson 1 - A Shop's Catalogue

Built a 1,000-item catalogue (brand, product, size) from a hand-written table of 127 products in 12 categories. The table (`catalog_data.py`) is copied from the course's reference repo, thili-ai/thilimarketbasket (MIT, licence kept in `LICENSE-thilimarketbasket`); the code that builds the catalogue is my own.

- With 4 brands per product there are 1,127 possible SKUs, so 1,000 is a real subset, not the whole space.
- Names are unique; asking for too many or too few items raises a clear error.
- A product has between 1 and 12 SKUs (median 8); 28 SKUs are loose, unbranded goods.

Takeaway: every later number depends on this catalogue being fixed and reproducible (seeded).
