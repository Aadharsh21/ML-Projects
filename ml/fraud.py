"""Card fraud: flag fraudulent transactions in a stream that is 99.83% legitimate.

v1 (notebooks/v1/CREDITFRAUD.ipynb) undersampled the legitimate transactions to a
50/50 mix, trained a logistic regression and reported 88% of fraud caught on a
balanced test sample. On a balanced sample every false alarm is cheap to ignore;
on the real stream there are ~580 legitimate transactions per fraud. v2 evaluates
everything on the last 25% of the timeline at its true class balance, adds a cost
model, and uses gradient boosted trees.

Run with ``python -m ml fraud``.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from . import data
from .evaluation import binary_metrics, bootstrap_ci, write_json
from .lgbm_export import compact, export_booster

PCA_FEATURES = [f"V{k}" for k in range(1, 29)]
FEATURES = PCA_FEATURES + ["LogAmount", "HourSin", "HourCos"]

REVIEW_COST = 5.0  # dollars of analyst time per alert; adjustable in the app

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.02,
    num_leaves=15,
    min_data_in_leaf=20,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    metric="average_precision",
    seed=42,
    deterministic=True,
    force_row_wise=True,
    num_threads=2,
    verbose=-1,
)

OUT_DIR = data.REPO_ROOT / "web" / "public" / "data" / "fraud"
FIXTURE_PATH = data.REPO_ROOT / "web" / "tests" / "fixtures" / "fraud.json"


def features(df: pd.DataFrame) -> pd.DataFrame:
    """PCA components as given, log amount, and time of day on a circle.

    Raw ``Time`` (seconds since the first transaction) is left out: the test
    window comes after every training timestamp, so a model would only be
    extrapolating on it. Hour of day carries the useful part.
    """
    X = df[PCA_FEATURES].astype(float).copy()
    X["LogAmount"] = np.log1p(df["Amount"].astype(float))
    hour = (df["Time"].astype(float) % 86400) / 3600
    X["HourSin"] = np.sin(2 * math.pi * hour / 24)
    X["HourCos"] = np.cos(2 * math.pi * hour / 24)
    return X[FEATURES]


def split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Chronological 60/15/25 split of a time-sorted frame."""
    n = len(df)
    i, j = int(n * 0.60), int(n * 0.75)
    idx = np.arange(n)
    return {"train": idx < i, "val": (idx >= i) & (idx < j), "test": idx >= j}


def v1_model() -> Pipeline:
    """v1's model: logistic regression on the PCA features plus robust-scaled amount."""
    scale_amount = ColumnTransformer([("amount", RobustScaler(), ["Amount"])],
                                     remainder="passthrough")
    return Pipeline([("scale", scale_amount), ("model", LogisticRegression(max_iter=1000))])


V1_COLUMNS = PCA_FEATURES + ["Amount"]


def undersample(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """v1's balancing step: keep every fraud and the same number of random legit rows."""
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(len(fraud), random_state=seed)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=seed)


def weighted_logistic() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=5000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(300, verbose=False)],
    )


def money(y: np.ndarray, amount: np.ndarray, flagged: np.ndarray, review_cost: float) -> dict:
    """What a flagging policy saves: fraud blocked minus the cost of reviewing alerts."""
    blocked = float(amount[flagged & (y == 1)].sum())
    missed = float(amount[~flagged & (y == 1)].sum())
    reviews = float(flagged.sum() * review_cost)
    return {"blocked": blocked, "missed": missed, "review_cost": reviews,
            "net_savings": blocked - reviews, "alerts": int(flagged.sum())}


def evaluate(y: np.ndarray, p: np.ndarray, amount: np.ndarray, hours: float) -> dict:
    m = binary_metrics(y, p, 0.5)
    m.update(money(y, amount, p >= 0.5, REVIEW_COST))
    m["alerts_per_hour"] = m["alerts"] / hours
    m["avg_precision_ci95"] = bootstrap_ci(y, p, average_precision_score, n=1000)
    expected_loss = p * amount >= REVIEW_COST
    m["expected_loss_rule"] = {
        **money(y, amount, expected_loss, REVIEW_COST),
        "tp": int((expected_loss & (y == 1)).sum()),
        "fp": int((expected_loss & (y == 0)).sum()),
    }
    return m


def _logit_milli(p: np.ndarray) -> list[int]:
    """Scores as log-odds in thousandths, clipped to +-15. Rounding is away from
    zero so no score crosses the 0.5 line (log-odds 0) when quantised."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    logit = np.clip(np.log(p / (1 - p)), -15, 15) * 1000
    return [int(v) for v in np.sign(logit) * np.ceil(np.abs(logit))]


def _pr_points(y: np.ndarray, p: np.ndarray, n: int = 200) -> list[list[float]]:
    precision, recall, _ = precision_recall_curve(y, p)
    keep = np.unique(np.linspace(0, len(recall) - 1, n).astype(int))
    return [[float(recall[k]), float(precision[k])] for k in keep]


def run(out_dir: Path = OUT_DIR, fixture_path: Path = FIXTURE_PATH) -> dict:
    df = data.load_creditcard()
    X = features(df)
    y = df["Class"].to_numpy().astype(int)
    s = split(df)
    tr, va, te = s["train"], s["val"], s["test"]
    test_df = df[te]
    yt = y[te]
    amount = test_df["Amount"].to_numpy(dtype=float)
    hours = float((test_df["Time"].max() - test_df["Time"].min()) / 3600)

    v1 = v1_model().fit(*(lambda b: (b[V1_COLUMNS], b["Class"]))(undersample(df[tr])))
    p_v1 = v1.predict_proba(test_df[V1_COLUMNS])[:, 1]

    lrw = weighted_logistic().fit(X[tr], y[tr])
    p_lrw = lrw.predict_proba(X[te])[:, 1]

    booster = train_lightgbm(X[tr], y[tr], X[va], y[va])
    p_v2 = booster.predict(X[te], num_iteration=booster.best_iteration)

    models = {
        "v1_undersampled_logistic": {"label": "v1: logistic regression on a 50/50 undersample",
                                      "test": evaluate(yt, p_v1, amount, hours)},
        "weighted_logistic": {"label": "Logistic regression, class-weighted",
                              "test": evaluate(yt, p_lrw, amount, hours)},
        "v2_lightgbm": {"label": "v2: LightGBM",
                        "test": evaluate(yt, p_v2, amount, hours)},
    }

    # How v1 looked when graded the v1 way: every test fraud plus as many random
    # legitimate transactions (a 50/50 sample), versus the real stream.
    rng = np.random.default_rng(3)
    fraud_idx = np.flatnonzero(yt == 1)
    legit_idx = rng.choice(np.flatnonzero(yt == 0), size=len(fraud_idx), replace=False)
    bal = np.concatenate([fraud_idx, legit_idx])
    balanced_view = binary_metrics(yt[bal], p_v1[bal], 0.5)

    t = test_df["Time"].to_numpy()
    stream = {
        "start_time": float(t[0]),
        "dt": [0] + [int(v) for v in np.diff(t)],
        "amount_cents": [int(round(a * 100)) for a in amount],
        "fraud": [int(k) for k in fraud_idx],
        "v2": _logit_milli(p_v2),
        "v1": _logit_milli(p_v1),
    }
    # The quantised scores must give exactly the same 0.5-threshold decisions.
    for key, p in (("v2", p_v2), ("v1", p_v1)):
        assert np.array_equal(np.array(stream[key]) >= 0, p >= 0.5), key

    # Full feature rows for the transactions the app lets you inspect: every
    # fraud, everything the model scores above 1%, and a sample of the rest.
    scored = np.flatnonzero(p_v2 >= 0.01)
    others = rng.choice(np.setdiff1d(np.arange(len(yt)), np.union1d(scored, fraud_idx)),
                        size=200, replace=False)
    inspect_idx = np.unique(np.concatenate([fraud_idx, scored, others]))
    Xt = X[te].to_numpy()
    inspect = {
        "index": [int(k) for k in inspect_idx],
        "features": [[float(v) for v in Xt[k]] for k in inspect_idx],
    }

    contrib = booster.predict(X[te], num_iteration=booster.best_iteration, pred_contrib=True)
    importance = sorted(
        ({"feature": f, "mean_abs_shap": float(np.abs(contrib[:, i]).mean())}
         for i, f in enumerate(FEATURES)),
        key=lambda r: -r["mean_abs_shap"],
    )

    def window(mask):
        tt = df.loc[mask, "Time"]
        return {"rows": int(mask.sum()), "frauds": int(y[mask].sum()),
                "start_hour": float(tt.min() / 3600), "end_hour": float(tt.max() / 3600)}

    meta = {
        "features": FEATURES,
        "splits": {"train": window(tr), "val": window(va), "test": window(te)},
        "test_hours": hours,
        "test_fraud_amount": float(amount[yt == 1].sum()),
        "review_cost": REVIEW_COST,
        "best_iteration": booster.best_iteration,
        "models": models,
        "v1_balanced_view": balanced_view,
        "pr_curves": {"v1": _pr_points(yt, p_v1), "v2": _pr_points(yt, p_v2)},
        "importance": importance,
        "source": data.DATASETS["creditcard"].source,
    }

    model = compact(export_booster(booster, num_iteration=booster.best_iteration))
    model["base_value"] = float(contrib[0, -1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model.json", model, digits=None)
    write_json(out_dir / "stream.json", stream, digits=None)
    write_json(out_dir / "inspect.json", inspect, digits=None)
    write_json(out_dir / "meta.json", meta)

    picks = inspect_idx[np.linspace(0, len(inspect_idx) - 1, 60).astype(int)]
    fx = X[te].iloc[picks]
    write_json(fixture_path, {
        "index": [int(k) for k in picks],
        "raw": [{"time": float(t[k]), "amount": float(amount[k])} for k in picks],
        "features": fx.to_numpy().tolist(),
        "probability": booster.predict(fx, num_iteration=booster.best_iteration).tolist(),
        "shap": booster.predict(fx, num_iteration=booster.best_iteration,
                                pred_contrib=True).tolist(),
    }, digits=None)
    return meta


def summary(meta: dict) -> str:
    rows = []
    for r in meta["models"].values():
        t = r["test"]
        rows.append(
            f"{r['label']:48s} AP {t['avg_precision']:.3f}  @0.5: caught {t['tp']}/{t['tp'] + t['fn']}"
            f"  false alarms {t['fp']}  net ${t['net_savings']:,.0f}"
        )
    b = meta["v1_balanced_view"]
    rows.append(f"v1 graded on a 50/50 sample: recall {b['recall']:.2f}, false alarms {b['fp']}")
    return "\n".join(rows)
