"""Export LightGBM models to a compact JSON format the web app can evaluate.

The format mirrors LightGBM's own tree arrays, so the browser walks exactly the
same splits as the C++ library:

* internal nodes are numbered 0..num_leaves-2, leaves are referenced as ``~index``
  (a negative number), just like ``Tree::left_child_`` in LightGBM
* each split stores its missing-value handling, because LightGBM sends NaN left,
  right, or treats it as zero depending on what it saw during training
* node and leaf sample counts are kept so TreeSHAP can run in the browser

``predict_raw`` below is a pure-Python reference implementation of the same walk,
used by the tests to prove the export is lossless.
"""

from __future__ import annotations

import math
from typing import Any

import lightgbm as lgb
import numpy as np

FORMAT = "lgbm-flat-1"
MISSING_TYPES = {"None": 0, "Zero": 1, "NaN": 2}
ZERO_THRESHOLD = 1e-35  # LightGBM's kZeroThreshold


def _flatten_tree(info: dict[str, Any]) -> dict[str, Any]:
    num_leaves = info["num_leaves"]
    root = info["tree_structure"]
    if num_leaves == 1:
        return {
            "split_feature": [],
            "threshold": [],
            "kind": [],
            "left": [],
            "right": [],
            "count": [],
            "leaf_value": [root["leaf_value"]],
            "leaf_count": [root.get("leaf_count", 0)],
        }

    n_internal = num_leaves - 1
    split_feature = [0] * n_internal
    threshold = [0.0] * n_internal
    kind = [0] * n_internal
    left = [0] * n_internal
    right = [0] * n_internal
    count = [0] * n_internal
    leaf_value = [0.0] * num_leaves
    leaf_count = [0] * num_leaves

    def ref(node: dict[str, Any]) -> int:
        return node["split_index"] if "split_index" in node else ~node["leaf_index"]

    stack = [root]
    while stack:
        node = stack.pop()
        if "leaf_index" in node:
            leaf_value[node["leaf_index"]] = node["leaf_value"]
            leaf_count[node["leaf_index"]] = node["leaf_count"]
            continue
        if node["decision_type"] != "<=":
            raise ValueError("Only numerical splits are supported (no categorical features)")
        i = node["split_index"]
        split_feature[i] = node["split_feature"]
        threshold[i] = node["threshold"]
        kind[i] = MISSING_TYPES[node["missing_type"]] * 2 + int(node["default_left"])
        left[i] = ref(node["left_child"])
        right[i] = ref(node["right_child"])
        count[i] = node["internal_count"]
        stack.extend([node["left_child"], node["right_child"]])

    return {
        "split_feature": split_feature,
        "threshold": threshold,
        "kind": kind,
        "left": left,
        "right": right,
        "count": count,
        "leaf_value": leaf_value,
        "leaf_count": leaf_count,
    }


def export_booster(booster: lgb.Booster, num_iteration: int | None = None) -> dict[str, Any]:
    """Convert a trained binary LightGBM booster into the flat JSON structure."""
    dump = booster.dump_model(num_iteration=num_iteration)
    if dump["num_class"] != 1 or not dump["objective"].startswith("binary"):
        raise ValueError(f"Expected a binary classifier, got {dump['objective']!r}")
    return {
        "format": FORMAT,
        "objective": "binary",
        "features": dump["feature_names"],
        "trees": [_flatten_tree(t) for t in dump["tree_info"]],
    }


def _next_node(tree: dict[str, Any], node: int, value: float) -> int:
    kind = tree["kind"][node]
    missing_type, default_left = kind >> 1, kind & 1
    if math.isnan(value) and missing_type != 2:
        value = 0.0
    if (missing_type == 1 and -ZERO_THRESHOLD <= value <= ZERO_THRESHOLD) or (
        missing_type == 2 and math.isnan(value)
    ):
        return tree["left"][node] if default_left else tree["right"][node]
    return tree["left"][node] if value <= tree["threshold"][node] else tree["right"][node]


def predict_raw(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Raw scores (log-odds) from the exported structure. Slow; used for verification."""
    X = np.asarray(X, dtype=float)
    out = np.zeros(len(X))
    for tree in model["trees"]:
        if not tree["split_feature"]:
            out += tree["leaf_value"][0]
            continue
        for r, row in enumerate(X):
            node = 0
            while node >= 0:
                node = _next_node(tree, node, row[tree["split_feature"][node]])
            out[r] += tree["leaf_value"][~node]
    return out


def compact(model: dict[str, Any], leaf_digits: int = 12) -> dict[str, Any]:
    """Round leaf values for a smaller file. Thresholds keep full precision so every
    split decision is identical; leaf rounding changes scores by ~1e-12."""
    out = dict(model)
    out["trees"] = [
        {**t, "leaf_value": [float(f"{v:.{leaf_digits}g}") for v in t["leaf_value"]]}
        for t in model["trees"]
    ]
    return out
