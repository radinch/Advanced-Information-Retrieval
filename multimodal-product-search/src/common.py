"""Shared utilities for the hybrid multimodal product retrieval project."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def minmax_map(pairs: list[tuple[int, float]]) -> dict[int, float]:
    if not pairs:
        return {}
    vals = np.asarray([score for _, score in pairs], dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if abs(hi - lo) < 1e-12:
        return {idx: 1.0 if hi != 0.0 else 0.0 for idx, _ in pairs}
    return {idx: float((score - lo) / (hi - lo)) for idx, score in pairs}


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    sd = float(arr.std())
    if sd < 1e-9:
        return [0.0] * len(values)
    return [float(x) for x in ((arr - float(arr.mean())) / sd)]


def resolve_path(path: str | None, project_root: str | os.PathLike[str] = ".") -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    q = Path(project_root) / p
    if q.exists():
        return str(q)
    return str(p)


def build_product_text(row: Any, max_chars: int = 6000) -> str:
    """Build stable reranker/LLM text from available catalog fields."""
    def get(name: str) -> Any:
        if isinstance(row, dict):
            return row.get(name)
        return getattr(row, name, None)

    fields = [
        ("Title", get("title")),
        ("Category", get("category_path")),
        ("Product type", get("product_type")),
        ("Brand", get("brand")),
        ("Color", get("color")),
        ("Material", get("material")),
        ("Style", get("style")),
        ("Dimensions", get("dimensions")),
        ("Features", get("bullet_points")),
        ("Description", get("description")),
    ]
    parts = [f"{label}: {safe_text(value).strip()}" for label, value in fields if safe_text(value).strip()]
    if not parts:
        parts = [safe_text(get("product_text"))]
    return "\n".join(parts)[:max_chars]


def load_sparse_jsonl(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec in load_jsonl(path):
        out[rec["product_id"]] = (
            np.asarray(rec.get("indices", []), dtype=np.int64),
            np.asarray(rec.get("values", []), dtype=np.float32),
        )
    return out
