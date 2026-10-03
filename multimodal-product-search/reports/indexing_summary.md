# Stage 1 - Indexing Summary

## Dense embedding
- Dense model: `Qwen/Qwen3-VL-Embedding-2B`
- Input policy: one combined `{text, image}` object -> one model-produced vector; no manual modality-vector mixing.
- Dimension: 2048
- Vectors: 2000
- Batch size: 2; checkpoint chunk: 64
- Runtime: Tesla T4; dense build minutes in this session: 2.82
- Dense vectors are L2-normalized.

## Sparse embedding
- Sparse model: `prithivida/Splade_PP_en_v2`
- Vocabulary dimension: 30522
- Batch size: 8; max length: 256
- Top-N retained dimensions per product: 256
- Average non-zero dimensions: 151.18 (min=46, max=256, failures=0)

## Indexes
- Dense index: `artifacts/index/dense_index.faiss` — FAISS IndexFlatIP; inner product on normalized vectors is cosine similarity.
- Sparse index: `artifacts/index/sparse_index.npz` — SciPy CSR; search uses sparse dot product similarity.
- Shared ID mapping: `artifacts/embeddings/product_ids.txt` (row i <-> product ID line i).
- Manifest: `artifacts/index/index_manifest.json`

## Sanity-check searches
- DENSE image-query (product B075YNL7R3|amazon.com): self rank=0, top=['B075YNL7R3|amazon.com', 'B07D5MBWJ7|amazon.ca', 'B078S81KPH|amazon.com']
- DENSE text-query (product B07B51H7B1|amazon.com): self rank=0, top=['B07B51H7B1|amazon.com', 'B075X2CVQK|amazon.com', 'B075X2LMTC|amazon.com']
- DENSE image+text (product B07B51H7B1|amazon.com): self rank=0, top=['B07B51H7B1|amazon.com', 'B075X2LMTC|amazon.com', 'B075X2CVQK|amazon.com']
- SPARSE title-query (product B07B51H7B1|amazon.com): self rank=0, top=['B07B51H7B1|amazon.com', 'B075X2CVQK|amazon.com', 'B07CVC1Q7C|amazon.com']

All sanity queries above are re-encoded from image/text inputs rather than reusing stored product vectors.
