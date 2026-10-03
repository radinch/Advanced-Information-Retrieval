# Stage 3 - Reranking Summary

- Reranker: `BAAI/bge-reranker-v2-m3`
- Pair format: `(query_text, structured product metadata text)` for text-bearing queries.
- Reranking depth: top 100 Stage 2 candidates per query.
- Score normalization: per-query z-score for cross-encoder, dense, and sparse signals.
- Fusion weights: `{"text": {"ce": 0.65, "dense": 0.2, "sparse": 0.15}, "image": {"ce": 0.0, "dense": 1.0, "sparse": 0.0}, "image_text": {"ce": 0.55, "dense": 0.3, "sparse": 0.15}}`
- Queries reranked: text=10, image=5, image_text=5.
- Image-only policy: text cross-encoder contribution is 0; final ranking preserves dense visual evidence.
- Image+text policy: the cross-encoder uses query text while the dense score preserves the combined visual+text query signal.

## Before/after examples

- q001 (text): top-1 before `B086B5MRFB|amazon.in` -> after `B086B5MRFB|amazon.in`; query=`gray fabric sofa for a small living room, not leather`
- q001 (text): top-1 before `B086B5MRFB|amazon.in` -> after `B086B5MRFB|amazon.in`; query=`gray fabric sofa for a small living room, not leather`
- q011 (image): top-1 before `B086TGRSBG|amazon.com` -> after `B086TGRSBG|amazon.com`; query=`data/raw/images/small/ce/ce04dd2a.jpg`
- q016 (image_text): top-1 before `B07Y5XQ2LS|amazon.in` -> after `B087X5M5LJ|amazon.com`; query=`similar style, but in black and not for dining`
