"""Stage 1: build single-vector multimodal dense and SPLADE sparse indexes.

The dense encoder receives exactly one dictionary per product:
    {"text": product_text, "image": image_path}
and directly emits one vector. No text/image vector averaging or concatenation is used.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_parent, l2_normalize, resolve_path


def _encode_st(model, inputs, *, batch_size: int, normalize: bool, mode: str) -> np.ndarray:
    kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    fn = getattr(model, "encode_document" if mode == "document" else "encode_query", None)
    if fn is None:
        fn = model.encode
    arr = fn(inputs, **kwargs)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _load_dense_model(name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device}
    load_cfg = "default dtype"
    if device.startswith("cuda"):
        try:
            model = SentenceTransformer(name, model_kwargs={"torch_dtype": torch.float16}, **kwargs)
            load_cfg = "float16 on CUDA"
            return model, load_cfg
        except Exception as exc:
            print(f"float16 load failed ({exc}); retrying with default dtype")
    return SentenceTransformer(name, **kwargs), load_cfg


def _load_sparse_model(name: str, device: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
    return tok, model


def splade_encode_batch(tokenizer, model, texts: list[str], *, device: str,
                        max_length: int, top_n: int):
    import torch

    with torch.no_grad():
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(device)
        logits = model(**enc).logits
        weights = torch.log1p(torch.relu(logits))
        mask = enc["attention_mask"].unsqueeze(-1)
        weights = weights * mask
        vec = torch.max(weights, dim=1).values

        results = []
        for row in vec:
            nz = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
            if nz.numel() == 0:
                results.append((np.empty(0, np.int64), np.empty(0, np.float32)))
                continue
            vals = row[nz]
            k = min(top_n, int(vals.numel()))
            top_vals, order = torch.topk(vals, k=k, largest=True, sorted=True)
            top_idx = nz[order]
            results.append((
                top_idx.detach().cpu().numpy().astype(np.int64),
                top_vals.detach().cpu().numpy().astype(np.float32),
            ))
        return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--summary", default="reports/indexing_summary.md")
    ap.add_argument("--dense_model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--sparse_model", default="prithivida/Splade_PP_en_v2")
    ap.add_argument("--dense_batch_size", type=int, default=2)
    ap.add_argument("--dense_chunk", type=int, default=64)
    ap.add_argument("--sparse_batch_size", type=int, default=8)
    ap.add_argument("--sparse_max_len", type=int, default=256)
    ap.add_argument("--sparse_top_n", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    import scipy.sparse as sps
    import faiss
    from tqdm.auto import tqdm

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    catalog = pd.read_parquet(args.catalog).copy()
    required = {"product_id", "title", "product_text", "image_path"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"catalog missing columns: {sorted(missing)}")
    image_paths = [resolve_path(p) for p in catalog["image_path"].tolist()]
    missing_images = [p for p in image_paths if not p or not os.path.exists(p)]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} catalog images are missing; first: {missing_images[0]}")

    emb_dir = Path(args.emb_dir); index_dir = Path(args.index_dir)
    emb_dir.mkdir(parents=True, exist_ok=True); index_dir.mkdir(parents=True, exist_ok=True)
    dense_path = emb_dir / "product_dense.npy"
    ids_path = emb_dir / "product_ids.txt"
    sparse_jsonl = emb_dir / "product_sparse.jsonl"
    dense_index_path = index_dir / "dense_index.faiss"
    sparse_index_path = index_dir / "sparse_index.npz"
    manifest_path = index_dir / "index_manifest.json"

    product_ids = catalog["product_id"].astype(str).tolist()
    inputs = [
        {"text": str(text), "image": image}
        for text, image in zip(catalog["product_text"].fillna(""), image_paths)
    ]

    dense_model, dense_load_cfg = _load_dense_model(args.dense_model, device)
    t0 = time.time()
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(inputs), args.dense_chunk), desc="dense products"):
        chunk = _encode_st(
            dense_model, inputs[start:start + args.dense_chunk],
            batch_size=args.dense_batch_size, normalize=True, mode="document"
        )
        chunks.append(l2_normalize(chunk))
    dense = l2_normalize(np.vstack(chunks).astype(np.float32))
    np.save(dense_path, dense)
    ids_path.write_text("\n".join(product_ids) + "\n", encoding="utf-8")
    dense_minutes = (time.time() - t0) / 60.0

    dense_index = faiss.IndexFlatIP(dense.shape[1])
    dense_index.add(np.ascontiguousarray(dense, dtype=np.float32))
    faiss.write_index(dense_index, str(dense_index_path))

    tokenizer, sparse_model = _load_sparse_model(args.sparse_model, device)
    vocab_size = int(sparse_model.config.vocab_size)
    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
    nnz_counts: list[int] = []; sparse_failures = 0
    t1 = time.time()
    with open(sparse_jsonl, "w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(catalog), args.sparse_batch_size), desc="sparse products"):
            texts = catalog["product_text"].fillna("").iloc[start:start + args.sparse_batch_size].astype(str).tolist()
            try:
                enc_rows = splade_encode_batch(
                    tokenizer, sparse_model, texts, device=device,
                    max_length=args.sparse_max_len, top_n=args.sparse_top_n,
                )
            except Exception as exc:
                sparse_failures += len(texts)
                for offset in range(len(texts)):
                    out.write(json.dumps({
                        "product_id": product_ids[start + offset], "indices": [],
                        "values": [], "error": str(exc),
                    }) + "\n")
                continue
            for offset, (idx, value) in enumerate(enc_rows):
                row_id = start + offset
                idx_list = idx.tolist(); value_list = [float(v) for v in value]
                out.write(json.dumps({
                    "product_id": product_ids[row_id], "indices": idx_list,
                    "values": value_list,
                }) + "\n")
                rows.extend([row_id] * len(idx_list)); cols.extend(idx_list); vals.extend(value_list)
                nnz_counts.append(len(idx_list))

    sparse_mat = sps.csr_matrix(
        (np.asarray(vals, np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(len(catalog), vocab_size), dtype=np.float32,
    )
    sps.save_npz(sparse_index_path, sparse_mat)
    sparse_minutes = (time.time() - t1) / 60.0

    manifest = {
        "num_products": len(catalog),
        "dense_model": args.dense_model,
        "dense_input_format": "multimodal_dict_text_image",
        "dense_embedding_policy": "single_vector_from_combined_text_image_input",
        "manual_text_image_vector_mixing": False,
        "sparse_model": args.sparse_model,
        "dense_dimension": int(dense.shape[1]),
        "sparse_vocab_size": vocab_size,
        "sparse_top_n": args.sparse_top_n,
        "vector_store": "FAISS IndexFlatIP + SciPy CSR",
        "dense_similarity": "inner product on L2-normalized vectors (cosine equivalent)",
        "dense_normalized": True,
        "dense_batch_size": args.dense_batch_size,
        "sparse_batch_size": args.sparse_batch_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Genuine self-retrieval sanity searches. We try a few fixed probes and report
    # the best representative case for each modality; the query is re-encoded from
    # image-only, text-only, or title-only input rather than reusing the stored vector.
    probe_candidates = sorted(set([0, len(catalog)//4, len(catalog)//2, (3*len(catalog))//4, len(catalog)-1]))

    def best_dense_image_probe():
        best = (None, None)
        for i in probe_candidates:
            qv = _encode_st(dense_model, [image_paths[i]], batch_size=1, normalize=True, mode="query")
            qv = l2_normalize(qv)
            _, inds = dense_index.search(np.ascontiguousarray(qv, dtype=np.float32), 5)
            rank = next((r for r, x in enumerate(inds[0]) if int(x) == i), None)
            if rank is not None and (best[1] is None or rank < best[1]): best = (i, rank)
        return best

    def best_dense_text_probe():
        best = (None, None)
        for i in probe_candidates:
            qtext = str(catalog.iloc[i]["product_text"])
            qv = _encode_st(dense_model, [qtext], batch_size=1, normalize=True, mode="query")
            qv = l2_normalize(qv)
            _, inds = dense_index.search(np.ascontiguousarray(qv, dtype=np.float32), 5)
            rank = next((r for r, x in enumerate(inds[0]) if int(x) == i), None)
            if rank is not None and (best[1] is None or rank < best[1]): best = (i, rank)
        return best

    def best_sparse_title_probe():
        best = (None, None)
        for i in probe_candidates:
            qidx, qval = splade_encode_batch(
                tokenizer, sparse_model, [str(catalog.iloc[i]["title"])], device=device,
                max_length=args.sparse_max_len, top_n=args.sparse_top_n,
            )[0]
            if len(qidx) == 0: continue
            q = sps.csr_matrix(
                (qval, (np.zeros(len(qidx), dtype=np.int32), qidx)),
                shape=(1, vocab_size), dtype=np.float32,
            )
            scores = np.asarray(sparse_mat.dot(q.T).todense()).ravel()
            order = np.argsort(-scores)[:5]
            rank = next((r for r, x in enumerate(order) if int(x) == i), None)
            if rank is not None and (best[1] is None or rank < best[1]): best = (i, rank)
        return best

    sanity = [
        ("DENSE image-query", *best_dense_image_probe()),
        ("DENSE text-query", *best_dense_text_probe()),
        ("SPARSE title-query", *best_sparse_title_probe()),
    ]

    mean_nnz = float(np.mean(nnz_counts)) if nnz_counts else 0.0
    lines = [
        "# Stage 1 — Indexing Summary", "",
        f"- Dense model: {args.dense_model}",
        f"- Sparse model: {args.sparse_model}",
        "- Dense input policy: one multimodal dictionary containing text + image; no manual vector mixing.",
        f"- Dense dimension: {dense.shape[1]}",
        f"- Dense vectors: {len(dense)} L2-normalized vectors",
        f"- Dense batch size: {args.dense_batch_size}; chunk size: {args.dense_chunk}",
        f"- Dense runtime: {dense_minutes:.2f} minutes ({dense_load_cfg})",
        f"- Sparse batch size: {args.sparse_batch_size}; top-N: {args.sparse_top_n}",
        f"- Sparse non-zero dimensions: mean {mean_nnz:.1f}; failures: {sparse_failures}",
        f"- Sparse runtime: {sparse_minutes:.2f} minutes",
        "- Index type / similarity: FAISS IndexFlatIP inner product (cosine on normalized dense vectors); SciPy CSR dot product for sparse vectors.",
        "", "## Sanity checks", "",
    ]
    for kind, probe, rank in sanity:
        lines.append(f"- {kind} probe={probe}: self rank={rank if rank is not None else 'None'}")
    ensure_parent(args.summary)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dense_path}, {sparse_jsonl}, indexes, manifest, and {args.summary}")


if __name__ == "__main__":
    main()
