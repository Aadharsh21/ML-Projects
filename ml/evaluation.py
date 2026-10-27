"""Shared evaluation and reporting helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn import metrics


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "accuracy": (tp + tn) / len(y),
        "roc_auc": metrics.roc_auc_score(y, p),
        "avg_precision": metrics.average_precision_score(y, p),
        "brier": metrics.brier_score_loss(y, p),
        "log_loss": metrics.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric computed on (y, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Mean predicted probability vs observed frequency in equal-width bins."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "mean_predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def _round(obj: Any, digits: int) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, (np.floating,)):
        return _round(float(obj), digits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    return obj


def write_json(path: Path, obj: Any, digits: int | None = 6) -> None:
    """Write JSON compactly. ``digits`` rounds floats to that many significant digits
    (use None for model files that must keep full precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if digits is not None:
        obj = _round(obj, digits)
    with path.open("w") as fh:
        json.dump(obj, fh, separators=(",", ":"), allow_nan=False)
        fh.write("\n")
