# Stage 4 — Evaluation Report

The evaluation uses human qrels from `data/qrels.jsonl`. Relevance grades are 0/1/2; metrics treat grades >0 as relevant for precision, recall, and MAP, while NDCG preserves graded gains.

## Aggregate metrics

| run_name | num_queries | map | ndcg_at_10 | precision_at_10 | judged_coverage |
|---|---|---|---|---|---|
| dense_single_vector | 20 | 0.7375 | 0.7752 | 0.6100 | 0.6975 |
| sparse_splade | 15 | 0.3551 | 0.3926 | 0.3067 | 0.6100 |
| hybrid_prefusion | 20 | 0.6399 | 0.6481 | 0.5550 | 0.8775 |
| hybrid_cross_encoder | 15 | 0.4178 | 0.4592 | 0.3667 | 0.6333 |
| hybrid_cross_encoder_fused | 20 | 0.5509 | 0.5339 | 0.4450 | 0.7950 |

Metrics reported: Precision@5, Precision@10, Recall@10, Recall@20, MAP, NDCG@5, NDCG@10, and judged coverage.

Sparse-only and text cross-encoder-only rows exclude image-only queries because those methods have no original textual query signal. The fused final run includes image-only queries using dense evidence.

## Judged coverage

Judged coverage is the fraction of retrieved top-20 products that appear in the qrels pool, averaged over queries. Unjudged items are treated as nonrelevant by the metric functions, so coverage should be read alongside effectiveness scores.

## Limitations

The qrels are pooled and incomplete, the catalog contains only 2,000 products, and fusion weights are fixed rather than tuned on this test split. A text-only cross-encoder cannot directly inspect image-only queries; visual evidence is therefore preserved through the dense score in the fused system.

See `reports/error_analysis.md` for qualitative cases.
