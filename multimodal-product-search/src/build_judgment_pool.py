"""Build a human-annotation pool and finalize reviewed qrels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from common import ensure_parent, load_jsonl, safe_text, write_jsonl


def build_pool(args) -> None:
    catalog = pd.read_parquet(args.catalog)
    cat = {str(r.product_id): r for r in catalog.itertuples()}
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    retrieval = {r["query_id"]: r for r in load_jsonl(args.retrieval)}
    reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}
    strategy_a = {}
    if Path(args.strategy_a).exists():
        strategy_a = {r["query_id"]: r for r in load_jsonl(args.strategy_a)}

    rows = []
    for qid, q in queries.items():
        chosen: list[str] = []
        def add(ids):
            for pid in ids:
                if pid not in chosen and len(chosen) < args.pool_size:
                    chosen.append(pid)

        ret_hits = retrieval.get(qid, {}).get("results", [])
        rer_hits = reranked.get(qid, {}).get("results", [])
        ce_hits = strategy_a.get(qid, {}).get("results", [])
        add([h["product_id"] for h in rer_hits[:args.per_run]])
        def score(h, field):
            value = h.get(field)
            return float(value) if value is not None else -1e30
        add([h["product_id"] for h in sorted(ret_hits, key=lambda h: -score(h, "prefusion_score"))[:args.per_run]])
        add([h["product_id"] for h in sorted(ret_hits, key=lambda h: -score(h, "dense_score"))[:args.per_run]])
        if q["query_type"] != "image":
            add([h["product_id"] for h in sorted(ret_hits, key=lambda h: -score(h, "sparse_score"))[:args.per_run]])
            add([h["product_id"] for h in ce_hits[:args.per_run]])
        add([h["product_id"] for h in ret_hits])

        if len(chosen) < 10:
            raise ValueError(f"{qid} produced only {len(chosen)} unique pool items; need at least 10")
        for pid in chosen:
            r = cat[pid]
            rows.append({
                "query_id": qid,
                "query_type": q["query_type"],
                "query_text": q.get("query_text") or "",
                "query_image_path": q.get("query_image_path") or "",
                "product_id": pid,
                "title": safe_text(getattr(r, "title", "")),
                "product_type": safe_text(getattr(r, "product_type", "")),
                "color": safe_text(getattr(r, "color", "")),
                "material": safe_text(getattr(r, "material", "")),
                "relevance": "",
                "reason": "",
            })

    ensure_parent(args.pool_out)
    pd.DataFrame(rows).to_csv(args.pool_out, index=False)
    print(f"wrote {len(rows)} rows to {args.pool_out}")
    print("Human step: fill relevance with 0/1/2 and write a short reason for every row.")


def finalize(args) -> None:
    df = pd.read_csv(args.finalize, keep_default_na=False)
    required = {"query_id", "product_id", "relevance", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"annotation CSV missing columns: {sorted(missing)}")
    qrels = []
    errors = []
    seen = set()
    for n, row in df.iterrows():
        pair = (str(row["query_id"]), str(row["product_id"]))
        if pair in seen:
            errors.append(f"row {n+2}: duplicate {pair}")
            continue
        seen.add(pair)
        try:
            rel_num = float(str(row["relevance"]).strip())
            if not rel_num.is_integer():
                raise ValueError
            rel = int(rel_num)
        except Exception:
            errors.append(f"row {n+2}: relevance must be 0, 1, or 2")
            continue
        if rel not in {0, 1, 2}:
            errors.append(f"row {n+2}: relevance {rel} is invalid")
        reason = str(row["reason"]).strip()
        if not reason:
            errors.append(f"row {n+2}: reason is empty")
        if reason.lower().startswith("auto:"):
            errors.append(f"row {n+2}: auto-generated labels are not allowed")
        qrels.append({"query_id": pair[0], "product_id": pair[1], "relevance": rel, "reason": reason})
    if errors:
        raise ValueError("Cannot finalize qrels:\n" + "\n".join(errors[:30]))
    write_jsonl(args.qrels_out, qrels)
    print(f"wrote {len(qrels)} human judgments to {args.qrels_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--strategy_a", default="outputs/reranked_strategy_a.jsonl")
    ap.add_argument("--pool_size", type=int, default=20)
    ap.add_argument("--per_run", type=int, default=8)
    ap.add_argument("--pool_out", default="data/qrels_pool.csv")
    ap.add_argument("--finalize")
    ap.add_argument("--qrels_out", default="data/qrels.jsonl")
    args = ap.parse_args()
    if args.finalize:
        finalize(args)
    else:
        build_pool(args)


if __name__ == "__main__":
    main()
