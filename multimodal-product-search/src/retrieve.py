"""Stage 2: hybrid dense multimodal + SPLADE candidate retrieval."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_parent, l2_normalize, load_jsonl, minmax_map, resolve_path, write_jsonl
from build_index import splade_encode_batch


def load_dense_encoder(name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer
    if device.startswith("cuda"):
        try:
            return SentenceTransformer(name, device=device, model_kwargs={"torch_dtype": torch.float16})
        except Exception as exc:
            print(f"float16 dense-model load failed ({exc}); using default dtype")
    return SentenceTransformer(name, device=device)


def encode_dense_queries(model, queries: list[dict], *, device: str, batch_size: int,
                         missing_image: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    fn = getattr(model, "encode_query", None) or model.encode
    for q in queries:
        qtype = q["query_type"]
        text = q.get("query_text") or ""
        image = resolve_path(q.get("query_image_path"))
        if qtype in {"image", "image_text"}:
            if not image or not os.path.exists(image):
                if missing_image == "skip":
                    print(f"warning: skipping dense query {q['query_id']}: missing image {image}")
                    continue
                raise FileNotFoundError(f"query image missing for {q['query_id']}: {image}")
        if qtype == "text":
            inp = [text]
        elif qtype == "image":
            # Assignment-specified image-only interface: pass the image path itself.
            inp = [image]
        elif qtype == "image_text":
            inp = [{"text": text, "image": image}]
        else:
            raise ValueError(f"unknown query_type {qtype!r}")
        vec = fn(inp, batch_size=batch_size, show_progress_bar=False,
                 convert_to_numpy=True, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        out[q["query_id"]] = l2_normalize(vec)[0]
    return out


def dense_search(index, q_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    q = np.ascontiguousarray(q_vec.reshape(1, -1), dtype=np.float32)
    scores, idx = index.search(q, top_k)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if int(i) >= 0]


def sparse_search(matrix, q_idx: np.ndarray, q_val: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    import scipy.sparse as sps
    if len(q_idx) == 0:
        return []
    q = sps.csr_matrix(
        (q_val.astype(np.float32), (np.zeros(len(q_idx), dtype=np.int32), q_idx.astype(np.int64))),
        shape=(1, matrix.shape[1]), dtype=np.float32,
    )
    scores = np.asarray(matrix.dot(q.T).todense()).ravel()
    positive = np.flatnonzero(scores > 0)
    if len(positive) == 0:
        return []
    k = min(top_k, len(positive))
    if k == len(positive):
        top = positive[np.argsort(-scores[positive])]
    else:
        part = np.argpartition(-scores[positive], k - 1)[:k]
        top = positive[part]
        top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top]


def merge_and_score(d_hits: list[tuple[int, float]], s_hits: list[tuple[int, float]],
                    *, sparse_applicable: bool, method: str, rrf_k: int,
                    w_dense: float, w_sparse: float):
    d_map = dict(d_hits); s_map = dict(s_hits)
    d_rank = {idx: rank for rank, (idx, _) in enumerate(d_hits, start=1)}
    s_rank = {idx: rank for rank, (idx, _) in enumerate(s_hits, start=1)}
    d_norm = minmax_map(d_hits); s_norm = minmax_map(s_hits)
    rows = []
    for idx in set(d_map) | set(s_map):
        source = []
        if idx in d_map: source.append("dense")
        if idx in s_map: source.append("sparse")
        if method == "rrf":
            score = 0.0
            if idx in d_rank: score += 1.0 / (rrf_k + d_rank[idx])
            if sparse_applicable and idx in s_rank: score += 1.0 / (rrf_k + s_rank[idx])
        elif method == "weighted":
            if sparse_applicable:
                score = w_dense * d_norm.get(idx, 0.0) + w_sparse * s_norm.get(idx, 0.0)
            else:
                score = d_norm.get(idx, 0.0)
        else:
            raise ValueError("prefusion method must be 'rrf' or 'weighted'")
        rows.append((idx, d_map.get(idx), s_map.get(idx), float(score), source))
    rows.sort(key=lambda r: (-r[3], -(r[1] if r[1] is not None else -1e30), r[0]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--top_k_dense", type=int, default=100)
    ap.add_argument("--top_k_sparse", type=int, default=100)
    ap.add_argument("--prefusion", choices=["rrf", "weighted"], default="rrf")
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--w_dense", type=float, default=0.55)
    ap.add_argument("--w_sparse", type=float, default=0.45)
    ap.add_argument("--dense_query_batch_size", type=int, default=1)
    ap.add_argument("--sparse_max_len", type=int, default=256)
    ap.add_argument("--missing_image", choices=["error", "skip"], default="error")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--summary", default="reports/retrieval_summary.md")
    args = ap.parse_args()

    import torch
    import faiss
    from scipy import sparse as sps
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    manifest = json.load(open(Path(args.index_dir) / "index_manifest.json", encoding="utf-8"))
    catalog = pd.read_parquet(args.catalog)
    product_ids = [line.strip() for line in open(Path(args.emb_dir) / "product_ids.txt", encoding="utf-8") if line.strip()]
    if product_ids != catalog["product_id"].astype(str).tolist():
        raise ValueError("product_ids.txt is not row-aligned with the catalog")
    queries = load_jsonl(args.queries)
    dense_index = faiss.read_index(str(Path(args.index_dir) / "dense_index.faiss"))
    sparse_index = sps.load_npz(Path(args.index_dir) / "sparse_index.npz")

    device = args.device
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    dense_model = load_dense_encoder(manifest["dense_model"], device)
    dense_queries = encode_dense_queries(
        dense_model, queries, device=device, batch_size=args.dense_query_batch_size,
        missing_image=args.missing_image,
    )

    tokenizer = AutoTokenizer.from_pretrained(manifest["sparse_model"])
    sparse_model = AutoModelForMaskedLM.from_pretrained(manifest["sparse_model"]).to(device).eval()
    sparse_queries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    text_queries = [q for q in queries if q["query_type"] in {"text", "image_text"}]
    for q in text_queries:
        enc = splade_encode_batch(
            tokenizer, sparse_model, [q.get("query_text") or ""], device=device,
            max_length=args.sparse_max_len, top_n=int(manifest["sparse_top_n"]),
        )[0]
        sparse_queries[q["query_id"]] = enc

    records = []
    unique_counts = []; overlap_counts = []
    for q in queries:
        qid = q["query_id"]
        d_hits = dense_search(dense_index, dense_queries[qid], args.top_k_dense) if qid in dense_queries else []
        sparse_applicable = q["query_type"] != "image"
        if sparse_applicable and qid in sparse_queries:
            q_idx, q_val = sparse_queries[qid]
            s_hits = sparse_search(sparse_index, q_idx, q_val, args.top_k_sparse)
        else:
            s_hits = []
        merged = merge_and_score(
            d_hits, s_hits, sparse_applicable=sparse_applicable, method=args.prefusion,
            rrf_k=args.rrf_k, w_dense=args.w_dense, w_sparse=args.w_sparse,
        )
        results = [
            {
                "rank": rank,
                "product_id": product_ids[idx],
                "dense_score": dense_score,
                "sparse_score": sparse_score if sparse_applicable else None,
                "prefusion_score": prefusion_score,
                "source": source,
            }
            for rank, (idx, dense_score, sparse_score, prefusion_score, source)
            in enumerate(merged, start=1)
        ]
        records.append({
            "query_id": qid,
            "query_type": q["query_type"],
            "run_name": "hybrid_dense_sparse_prefusion",
            "sparse_applicable": sparse_applicable,
            "results": results,
        })
        unique_counts.append(len(results))
        overlap_counts.append(len(set(i for i, _ in d_hits) & set(i for i, _ in s_hits)))

    write_jsonl(args.out, records)
    counts = collections.Counter(q["query_type"] for q in queries)
    examples = []
    for rec in records[:3]:
        top = ", ".join(f"#{h['rank']} {h['product_id']} ({h['prefusion_score']:.4f})" for h in rec["results"][:3])
        examples.append(f"- **{rec['query_id']}** ({rec['query_type']}): {top}")
    ensure_parent(args.summary)
    Path(args.summary).write_text("\n".join([
        "# Stage 2 — Retrieval Summary", "",
        f"- Queries: text={counts['text']}, image={counts['image']}, image_text={counts['image_text']}.",
        f"- Dense model: {manifest['dense_model']}",
        f"- Sparse model: {manifest['sparse_model']}",
        f"- Candidate depths: dense={args.top_k_dense}, sparse={args.top_k_sparse}.",
        f"- Prefusion: {args.prefusion}; RRF k={args.rrf_k}; weighted coefficients dense={args.w_dense}, sparse={args.w_sparse}.",
        f"- Mean unique candidate pool: {float(np.mean(unique_counts)):.1f}.",
        f"- Mean dense/sparse overlap: {float(np.mean(overlap_counts)):.1f}.",
        "", "## Example top results", "", *examples, "",
        "Image-only queries use dense retrieval only; sparse_applicable=false and sparse_score=null.",
    ]) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} queries to {args.out}")


if __name__ == "__main__":
    main()
