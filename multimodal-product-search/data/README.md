# Product retrieval inputs

The original experiment uses a 2,000-product ABO-Home subset and 20 queries: 10 text, 5 image, and 5 combined image-text queries.

| Input | Expected location |
| --- | --- |
| Product catalog | `data/catalog_subset.parquet` |
| Query records | `data/queries.jsonl` |
| Product and query images | Paths referenced by the catalog and queries, typically `data/raw/images/small/` |
| Recorded annotation pool | `data/qrels_pool.csv` (included) |

The catalog, query JSONL, and images were absent from the uploaded archive. Restore the matching original inputs before running retrieval or interpreting the included pool against a new catalog. The original notebook provides this [input-folder reference](https://drive.google.com/drive/folders/1Z_kG20b3snyV8nEszNdpDiq4wxmKlL0C?usp=sharing); its availability has not been verified.

`qrels_pool.csv` contains 400 recorded query-product judgments with grades and reasons. Finalize it with `src/build_judgment_pool.py --finalize data/qrels_pool.csv`. A fresh pool requires manual review. Images should resolve relative to this module directory; an optional `data/product_images.zip` can contain the same relative image tree.
