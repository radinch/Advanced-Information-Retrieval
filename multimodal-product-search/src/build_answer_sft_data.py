"""Build optional Stage 5 supervised fine-tuning examples from accepted answers.

The LLM is still not used for retrieval. This script reconstructs the same grounded
prompt from the saved query/rerank/catalog artifacts and pairs it with an existing
schema-valid target answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import load_jsonl, write_jsonl
from generate_answer import build_prompt, fallback_answer, validate_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--out", default="data/answer_sft_seed.jsonl")
    ap.add_argument("--target_source", choices=["existing", "existing_or_fallback"], default="existing_or_fallback")
    args = ap.parse_args()

    catalog = pd.read_parquet(args.catalog)
    cat_by_id = {str(r.product_id): r for r in catalog.itertuples()}
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}
    answers = {a["query_id"]: a for a in load_jsonl(args.answers)} if Path(args.answers).exists() else {}

    rows = []
    for qid, query in queries.items():
        hits = reranked.get(qid, {}).get("results", [])[:args.top_k]
        if not hits:
            raise ValueError(f"{qid} has no reranked candidates")
        allowed_ids = {h["product_id"] for h in hits}
        target = answers.get(qid)
        if target is not None:
            try:
                validate_answer(target, qid=qid, allowed_ids=allowed_ids)
            except Exception:
                target = None
        if target is None:
            if args.target_source != "existing_or_fallback":
                raise ValueError(f"{qid} has no valid existing target in {args.answers}")
            target = fallback_answer(query, hits, cat_by_id)
        rows.append({
            "query_id": qid,
            "prompt": build_prompt(query, hits, cat_by_id),
            "completion": json.dumps(target, ensure_ascii=False),
        })

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} SFT examples to {args.out}")


if __name__ == "__main__":
    main()
