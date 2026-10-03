"""Stage 4: IR metrics, comparison table, and qualitative error analysis."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import ensure_parent, load_jsonl


def precision_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for pid in ranked[:k] if qrels.get(pid, 0) > 0) / float(k)


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {pid for pid, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / float(len(relevant))


def average_precision(ranked: list[str], qrels: dict[str, int]) -> float:
    relevant = {pid for pid, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    seen = 0
    acc = 0.0
    for rank, pid in enumerate(ranked, start=1):
        if pid in relevant:
            seen += 1
            acc += seen / rank
    return acc / len(relevant)


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    if not qrels or k <= 0:
        return 0.0
    def dcg(rels: Iterable[int]) -> float:
        return sum((2 ** int(rel) - 1) / math.log2(i + 2) for i, rel in enumerate(rels))
    actual = [qrels.get(pid, 0) for pid in ranked[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else 0.0


def judged_coverage(ranked: list[str], qrels: dict[str, int], k: int = 20) -> float:
    if not ranked:
        return 0.0
    cutoff = ranked[:k]
    if not cutoff:
        return 0.0
    return sum(pid in qrels for pid in cutoff) / len(cutoff)


def _rank_by(results: list[dict], field: str) -> list[str]:
    usable = [h for h in results if h.get(field) is not None]
    usable.sort(key=lambda h: (-float(h[field]), int(h.get("rank", 10**9))))
    return [h["product_id"] for h in usable]


def _load_runs(retrieval_path: str, reranked_path: str, strategy_a_path: str):
    retrieval = {r["query_id"]: r for r in load_jsonl(retrieval_path)}
    reranked = {r["query_id"]: r for r in load_jsonl(reranked_path)}
    strategy_a = {}
    if Path(strategy_a_path).exists():
        strategy_a = {r["query_id"]: r for r in load_jsonl(strategy_a_path)}

    runs: dict[str, dict[str, list[str]]] = {
        "dense_single_vector": {},
        "sparse_splade": {},
        "hybrid_prefusion": {},
        "hybrid_cross_encoder": {},
        "hybrid_cross_encoder_fused": {},
    }
    for qid, rec in retrieval.items():
        hits = rec.get("results", [])
        runs["dense_single_vector"][qid] = _rank_by(hits, "dense_score")
        if rec.get("query_type") != "image":
            runs["sparse_splade"][qid] = _rank_by(hits, "sparse_score")
        runs["hybrid_prefusion"][qid] = _rank_by(hits, "prefusion_score")
    for qid, rec in reranked.items():
        hits = rec.get("results", [])
        qtype = rec.get("query_type")
        if qtype != "image":
            if qid in strategy_a:
                runs["hybrid_cross_encoder"][qid] = [h["product_id"] for h in strategy_a[qid].get("results", [])]
            else:
                runs["hybrid_cross_encoder"][qid] = _rank_by(hits, "cross_encoder_score")
        runs["hybrid_cross_encoder_fused"][qid] = _rank_by(hits, "final_score")
    return runs, retrieval, reranked


def _per_query_metrics(ranked: list[str], qrels: dict[str, int]) -> dict[str, float]:
    return {
        "precision_at_5": precision_at_k(ranked, qrels, 5),
        "precision_at_10": precision_at_k(ranked, qrels, 10),
        "recall_at_10": recall_at_k(ranked, qrels, 10),
        "recall_at_20": recall_at_k(ranked, qrels, 20),
        "map": average_precision(ranked, qrels),
        "ndcg_at_5": ndcg_at_k(ranked, qrels, 5),
        "ndcg_at_10": ndcg_at_k(ranked, qrels, 10),
        "judged_coverage": judged_coverage(ranked, qrels, 20),
    }


def _format_top(rec: dict, qrels: dict[str, int], score_field: str, k: int = 5) -> str:
    hits = sorted(
        rec.get("results", []),
        key=lambda h: -float(h.get(score_field) if h.get(score_field) is not None else -1e30),
    )[:k]
    if not hits:
        return "(no results)"
    return "; ".join(
        f"#{i+1} `{h['product_id']}` rel={qrels.get(h['product_id'], 'unjudged')}"
        for i, h in enumerate(hits)
    )


def _write_error_analysis(path: str, queries: dict, qrels_by_q: dict,
                          runs: dict, retrieval: dict, reranked: dict) -> None:
    def delta(qid: str, a: str, b: str) -> float:
        qr = qrels_by_q.get(qid, {})
        return ndcg_at_k(runs[a].get(qid, []), qr, 10) - ndcg_at_k(runs[b].get(qid, []), qr, 10)

    text_qids = [qid for qid, q in queries.items() if q["query_type"] in {"text", "image_text"}]
    all_qids = list(queries)
    dense_case = max(text_qids, key=lambda qid: delta(qid, "dense_single_vector", "sparse_splade"), default=all_qids[0])
    sparse_case = max(text_qids, key=lambda qid: delta(qid, "sparse_splade", "dense_single_vector"), default=all_qids[0])
    improve_case = max(all_qids, key=lambda qid: delta(qid, "hybrid_cross_encoder_fused", "hybrid_prefusion"), default=all_qids[0])
    hurt_case = min(all_qids, key=lambda qid: delta(qid, "hybrid_cross_encoder_fused", "hybrid_prefusion"), default=all_qids[0])
    image_candidates = [qid for qid, q in queries.items() if q["query_type"] in {"image", "image_text"}]
    image_case = image_candidates[0] if image_candidates else all_qids[0]

    cases = [
        ("Dense retrieval better than sparse", dense_case, "This case is selected by the largest NDCG@10 advantage of dense over sparse retrieval."),
        ("Sparse retrieval better than dense", sparse_case, "This case is selected by the largest NDCG@10 advantage of sparse over dense retrieval."),
        ("Cross-encoder fusion improves ranking", improve_case, "This case has the largest NDCG@10 gain from prefusion to the final fused reranking."),
        ("Cross-encoder fusion hurts ranking", hurt_case, "This case has the smallest (possibly negative) NDCG@10 change after reranking."),
        ("Image or image+text case", image_case, "This case illustrates visual retrieval behavior; image-only queries remain dense-only with the text cross-encoder disabled."),
    ]
    lines = ["# Error Analysis", ""]
    for title, qid, note in cases:
        q = queries[qid]; qr = qrels_by_q.get(qid, {})
        lines += [
            f"## {title} — {qid}", "",
            f"**Query type:** {q['query_type']}",
            f"**Query:** {q.get('query_text') or '[image query: ' + str(q.get('query_image_path')) + ']'}", "",
            "**Top results before reranking:** " + _format_top(retrieval.get(qid, {}), qr, "prefusion_score"), "",
            "**Top results after reranking:** " + _format_top(reranked.get(qid, {}), qr, "final_score"), "",
            f"**Analysis:** {note} Dense NDCG@10={ndcg_at_k(runs['dense_single_vector'].get(qid, []), qr, 10):.3f}; "
            f"sparse NDCG@10={ndcg_at_k(runs['sparse_splade'].get(qid, []), qr, 10):.3f}; "
            f"prefusion NDCG@10={ndcg_at_k(runs['hybrid_prefusion'].get(qid, []), qr, 10):.3f}; "
            f"final NDCG@10={ndcg_at_k(runs['hybrid_cross_encoder_fused'].get(qid, []), qr, 10):.3f}.", "",
        ]
    ensure_parent(path)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--qrels", default="data/qrels.jsonl")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Optional compatibility argument; first retrieval and reranked paths are inferred by filename.")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--strategy_a", default="outputs/reranked_strategy_a.jsonl")
    ap.add_argument("--out", default="reports/evaluation_metrics.csv")
    ap.add_argument("--report", default="reports/evaluation_report.md")
    ap.add_argument("--error_analysis", default="reports/error_analysis.md")
    args = ap.parse_args()

    if args.runs:
        for p in args.runs:
            name = Path(p).name.lower()
            if "retrieval" in name:
                args.retrieval = p
            elif "rerank" in name and "strategy" not in name:
                args.reranked = p

    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    qrel_rows = load_jsonl(args.qrels)
    qrels_by_q: dict[str, dict[str, int]] = {qid: {} for qid in queries}
    for r in qrel_rows:
        qrels_by_q.setdefault(r["query_id"], {})[r["product_id"]] = int(r["relevance"])

    runs, retrieval, reranked = _load_runs(args.retrieval, args.reranked, args.strategy_a)
    rows = []
    for run_name, rankings in runs.items():
        for qtype in ["text", "image", "image_text", "all"]:
            qids = [qid for qid in rankings if qid in queries and (qtype == "all" or queries[qid]["query_type"] == qtype)]
            if not qids:
                continue
            vals = [_per_query_metrics(rankings[qid], qrels_by_q.get(qid, {})) for qid in qids]
            rows.append({
                "run_name": run_name,
                "query_type": qtype,
                "num_queries": len(qids),
                **{key: float(np.mean([v[key] for v in vals])) for key in vals[0]},
            })
    metrics = pd.DataFrame(rows)
    ensure_parent(args.out)
    metrics.to_csv(args.out, index=False)

    _write_error_analysis(args.error_analysis, queries, qrels_by_q, runs, retrieval, reranked)

    all_rows = metrics[metrics["query_type"] == "all"].copy()
    if not all_rows.empty:
        cols = ["run_name", "num_queries", "map", "ndcg_at_10", "precision_at_10", "judged_coverage"]
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = []
        for _, r in all_rows[cols].iterrows():
            vals = [str(r["run_name"]), str(int(r["num_queries"])), f"{r['map']:.4f}", f"{r['ndcg_at_10']:.4f}", f"{r['precision_at_10']:.4f}", f"{r['judged_coverage']:.4f}"]
            body.append("| " + " | ".join(vals) + " |")
        table = "\n".join([header, sep, *body])
    else:
        table = "No metrics produced."
    ensure_parent(args.report)
    Path(args.report).write_text("\n".join([
        "# Stage 4 — Evaluation Report", "",
        f"The evaluation uses human qrels from `{args.qrels}`. Relevance grades are 0/1/2; metrics treat grades >0 as relevant for precision, recall, and MAP, while NDCG preserves graded gains.", "",
        "## Aggregate metrics", "", table, "",
        "Metrics reported: Precision@5, Precision@10, Recall@10, Recall@20, MAP, NDCG@5, NDCG@10, and judged coverage.", "",
        "Sparse-only and text cross-encoder-only rows exclude image-only queries because those methods have no original textual query signal. The fused final run includes image-only queries using dense evidence.", "",
        "## Judged coverage", "",
        "Judged coverage is the fraction of retrieved top-20 products that appear in the qrels pool, averaged over queries. Unjudged items are treated as nonrelevant by the metric functions, so coverage should be read alongside effectiveness scores.", "",
        "## Limitations", "",
        "The qrels are pooled and incomplete, the catalog contains only 2,000 products, and fusion weights are fixed rather than tuned on this test split. A text-only cross-encoder cannot directly inspect image-only queries; visual evidence is therefore preserved through the dense score in the fused system.", "",
        f"See `{args.error_analysis}` for qualitative cases.",
    ]) + "\n", encoding="utf-8")
    print(f"wrote {args.out}, {args.report}, and {args.error_analysis}")


if __name__ == "__main__":
    main()
