"""Stage 3: cross-encoder reranking with retrieval-score fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import build_product_text, ensure_parent, load_jsonl, write_jsonl, zscore


DEFAULT_WEIGHTS = {
    "text": {"ce": 0.65, "dense": 0.20, "sparse": 0.15},
    "image": {"ce": 0.00, "dense": 1.00, "sparse": 0.00},
    "image_text": {"ce": 0.55, "dense": 0.30, "sparse": 0.15},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rerank_top_n", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--product_max_chars", type=int, default=6000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/reranked_results.jsonl")
    ap.add_argument("--strategy_a_out", default="outputs/reranked_strategy_a.jsonl")
    ap.add_argument("--summary", default="reports/reranking_summary.md")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    device = args.device
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog = pd.read_parquet(args.catalog)
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    retrieval = load_jsonl(args.retrieval)
    cat_by_id = {str(r.product_id): r for r in catalog.itertuples()}

    # A text cross-encoder is used only when query text exists. Image-only final
    # ranking remains dense-only; this is explicit and evaluated separately.
    ce_model = CrossEncoder(args.reranker, device=device, max_length=args.max_length)

    fused_records = []
    ce_only_records = []
    examples = []
    for rec in retrieval:
        qid = rec["query_id"]
        q = queries[qid]
        qtype = q["query_type"]
        hits = rec.get("results", [])[:args.rerank_top_n]
        if not hits:
            fused_records.append({
                "query_id": qid, "query_type": qtype,
                "run_name": "hybrid_cross_encoder_fused",
                "reranker_type": "text_cross_encoder_with_multimodal_dense_fusion", "results": [],
            })
            ce_only_records.append({
                "query_id": qid, "query_type": qtype,
                "run_name": "hybrid_cross_encoder",
                "reranker_type": "text_cross_encoder", "results": [],
            })
            continue

        qtext = (q.get("query_text") or "").strip()
        if qtype == "image":
            ce_scores = [0.0] * len(hits)
        else:
            pairs = []
            for h in hits:
                row = cat_by_id[h["product_id"]]
                pairs.append((qtext, build_product_text(row, max_chars=args.product_max_chars)))
            pred = ce_model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            ce_scores = [float(x) for x in np.asarray(pred).reshape(-1)]

        dense_raw = [float(h["dense_score"]) if h.get("dense_score") is not None else 0.0 for h in hits]
        sparse_raw = [float(h["sparse_score"]) if h.get("sparse_score") is not None else 0.0 for h in hits]
        ce_z = zscore(ce_scores)
        dense_z = zscore(dense_raw)
        sparse_z = zscore(sparse_raw) if qtype != "image" else [0.0] * len(hits)
        w = DEFAULT_WEIGHTS[qtype]

        scored = []
        for i, h in enumerate(hits):
            final = w["ce"] * ce_z[i] + w["dense"] * dense_z[i] + w["sparse"] * sparse_z[i]
            row = dict(h)
            row["cross_encoder_score"] = ce_scores[i]
            row["final_score"] = float(final)
            scored.append(row)

        # Strategy A: cross-encoder only. For image-only it is retained as a file
        # for inspection, but Stage 4 deliberately excludes it from metrics.
        ce_sorted = sorted(scored, key=lambda h: (-h["cross_encoder_score"], h["product_id"]))
        ce_only_results = []
        for rank, h in enumerate(ce_sorted, start=1):
            x = dict(h); x["rank"] = rank; x["final_score"] = float(h["cross_encoder_score"])
            ce_only_results.append(x)
        ce_only_records.append({
            "query_id": qid, "query_type": qtype, "run_name": "hybrid_cross_encoder",
            "reranker_type": "text_cross_encoder", "results": ce_only_results,
        })

        fused_sorted = sorted(scored, key=lambda h: (-h["final_score"], -h["prefusion_score"], h["product_id"]))
        fused_results = []
        for rank, h in enumerate(fused_sorted, start=1):
            x = dict(h); x["rank"] = rank; fused_results.append(x)
        fused_records.append({
            "query_id": qid, "query_type": qtype,
            "run_name": "hybrid_cross_encoder_fused",
            "reranker_type": "text_cross_encoder_with_multimodal_dense_fusion",
            "results": fused_results,
        })
        before = hits[0]["product_id"] if hits else "none"
        after = fused_results[0]["product_id"] if fused_results else "none"
        if len(examples) < 5:
            examples.append(f"- {qid} ({qtype}): top-1 before `{before}` -> after `{after}`")

    write_jsonl(args.out, fused_records)
    write_jsonl(args.strategy_a_out, ce_only_records)
    ensure_parent(args.summary)
    Path(args.summary).write_text("\n".join([
        "# Stage 3 — Reranking Summary", "",
        f"- Reranker: {args.reranker}",
        "- Pair format: `(query_text, structured product metadata text)`.",
        f"- Reranking depth: top {args.rerank_top_n} Stage 2 candidates per query.",
        "- Score normalization: per-query z-score for cross-encoder, dense, and sparse evidence.",
        f"- Fusion weights: `{json.dumps(DEFAULT_WEIGHTS)}`",
        "- Image-only policy: the text cross-encoder is not semantically applicable, so its score is set to 0 and the fused ranking uses dense evidence only.",
        "- Image+text policy: cross-encoder reranks from query text while the dense score preserves the combined image+text signal.",
        "", "## Before/after examples", "", *examples, "",
    ]) + "\n", encoding="utf-8")
    print(f"wrote {len(fused_records)} reranked queries to {args.out}")


if __name__ == "__main__":
    main()
