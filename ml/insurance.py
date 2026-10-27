"""Insurance charges: estimate a customer's yearly medical charges.

v1 (notebooks/v1/LINEARINSURANCE.ipynb) fit two straight lines on age alone, one
for smokers and one for non-smokers. v2 uses every column plus the interaction
that drives this dataset (smokers with a BMI of 30 or more are charged about
$15k more), compares against LightGBM with repeated cross-validation, and returns
a range of likely charges instead of a single number.

Run with ``python -m ml insurance``.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold

from . import data
from .evaluation import write_json

REGIONS = ["northeast", "northwest", "southeast", "southwest"]
INPUTS = ["age", "sex", "bmi", "children", "smoker", "region"]
FEATURES = ["age", "age_squared", "bmi", "children", "smoker", "male",
            "region_northwest", "region_southeast", "region_southwest",
            "obese", "smoker_obese", "smoker_bmi"]
OBESE_BMI = 30.0
RANGE = (0.10, 0.90)  # the app shows the middle 80% of likely charges

OUT_DIR = data.REPO_ROOT / "web" / "public" / "data" / "insurance"
FIXTURE_PATH = data.REPO_ROOT / "web" / "tests" / "fixtures" / "insurance.json"


def design(df: pd.DataFrame, interactions: bool = True) -> pd.DataFrame:
    """Numeric design matrix. ``interactions=False`` gives the plain all-columns model."""
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"].astype(float)
    X["age_squared"] = X["age"] ** 2
    X["bmi"] = df["bmi"].astype(float)
    X["children"] = df["children"].astype(float)
    X["smoker"] = (df["smoker"] == "yes").astype(float)
    X["male"] = (df["sex"] == "male").astype(float)
    for r in REGIONS[1:]:
        X[f"region_{r}"] = (df["region"] == r).astype(float)
    X["obese"] = (X["bmi"] >= OBESE_BMI).astype(float)
    X["smoker_obese"] = X["smoker"] * X["obese"]
    X["smoker_bmi"] = X["smoker"] * X["bmi"]
    if not interactions:
        return X[["age", "bmi", "children", "smoker", "male",
                  "region_northwest", "region_southeast", "region_southwest"]]
    return X[FEATURES]


class AgeOnlyBySmoker:
    """v1: one line of charges against age for smokers and another for non-smokers."""

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "AgeOnlyBySmoker":
        self.models = {}
        for s in ("yes", "no"):
            m = (df["smoker"] == s).to_numpy()
            self.models[s] = LinearRegression().fit(df.loc[m, ["age"]], y[m])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df))
        for s, model in self.models.items():
            m = (df["smoker"] == s).to_numpy()
            if m.any():
                out[m] = model.predict(df.loc[m, ["age"]])
        return out


class Linear:
    def __init__(self, interactions: bool = True):
        self.interactions = interactions

    def fit(self, df, y):
        self.model = LinearRegression().fit(design(df, self.interactions), y)
        return self

    def predict(self, df):
        return self.model.predict(design(df, self.interactions))


class Boosted:
    def fit(self, df, y):
        self.model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8,
                                       min_child_samples=20, random_state=0, verbose=-1)
        self.model.fit(design(df, interactions=False), y)
        return self

    def predict(self, df):
        return self.model.predict(design(df, interactions=False))


CANDIDATES = {
    "v1_age_only": ("v1: age only, separate lines for smokers", AgeOnlyBySmoker),
    "linear_all": ("Linear regression, all columns", lambda: Linear(interactions=False)),
    "lightgbm": ("LightGBM, all columns", Boosted),
    "v2_linear_interactions": ("v2: linear regression with smoker x BMI interactions",
                               lambda: Linear(interactions=True)),
}


def cross_validate(df: pd.DataFrame, y: np.ndarray, factory, seed: int = 0) -> dict:
    scores = {"r2": [], "mae": [], "rmse": []}
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed).split(df):
        model = factory().fit(df.iloc[tr], y[tr])
        p = model.predict(df.iloc[te])
        scores["r2"].append(r2_score(y[te], p))
        scores["mae"].append(mean_absolute_error(y[te], p))
        scores["rmse"].append(math.sqrt(mean_squared_error(y[te], p)))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()}


def out_of_fold_residuals(df: pd.DataFrame, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Residuals from 5-fold cross-fitting: each row is scored by a model that never saw it."""
    res = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(df):
        model = Linear().fit(df.iloc[tr], y[tr])
        res[te] = y[te] - model.predict(df.iloc[te])
    return res


def residual_pools(df: pd.DataFrame, residuals: np.ndarray) -> dict[str, np.ndarray]:
    smoker = (df["smoker"] == "yes").to_numpy()
    return {"smoker": np.sort(residuals[smoker]), "non_smoker": np.sort(residuals[~smoker])}


def interval_coverage(df: pd.DataFrame, y: np.ndarray, repeats: int = 100, seed: int = 0) -> dict:
    """How often the 80% range contains the real charge on held-out customers."""
    rng = np.random.default_rng(seed)
    hits, widths = [], []
    for r in range(repeats):
        idx = rng.permutation(len(df))
        test, train = idx[: len(df) // 5], idx[len(df) // 5:]
        dtr, dte = df.iloc[train], df.iloc[test]
        pools = residual_pools(dtr, out_of_fold_residuals(dtr, y[train], seed=r))
        pred = Linear().fit(dtr, y[train]).predict(dte)
        bounds = {k: np.quantile(pool, RANGE) for k, pool in pools.items()}
        for p, actual, s in zip(pred, y[test], dte["smoker"]):
            lo, hi = bounds["smoker" if s == "yes" else "non_smoker"]
            hits.append(p + lo <= actual <= p + hi)
            widths.append(hi - lo)
    return {"target": RANGE[1] - RANGE[0], "coverage": float(np.mean(hits)),
            "mean_width": float(np.mean(widths)), "repeats": repeats}


def predict_rows(model: Linear, rows: np.ndarray) -> np.ndarray:
    """Predict from rows of [age, male, bmi, children, smoker, region_index]."""
    frame = pd.DataFrame({
        "age": rows[:, 0],
        "sex": np.where(rows[:, 1] == 1, "male", "female"),
        "bmi": rows[:, 2],
        "children": rows[:, 3],
        "smoker": np.where(rows[:, 4] == 1, "yes", "no"),
        "region": [REGIONS[int(i)] for i in rows[:, 5]],
    })
    return model.predict(frame)


def encode_inputs(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["age"].to_numpy(float),
        (df["sex"] == "male").to_numpy(float),
        df["bmi"].to_numpy(float),
        df["children"].to_numpy(float),
        (df["smoker"] == "yes").to_numpy(float),
        df["region"].map({r: i for i, r in enumerate(REGIONS)}).to_numpy(float),
    ])


def shapley(model: Linear, x: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact Shapley values of the six inputs against the average customer.

    v(S) is the mean prediction when the inputs in S are set to the profile's values
    and the rest come from each background customer. With six inputs that is 64
    coalitions, cheap enough to compute exactly (the app does the same in the browser).
    """
    n = len(INPUTS)
    values = {}
    for mask in range(1 << n):
        rows = background.copy()
        for i in range(n):
            if mask >> i & 1:
                rows[:, i] = x[i]
        values[mask] = float(predict_rows(model, rows).mean())
    phi = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            size = bin(mask).count("1")
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            phi[i] += weight * (values[mask | 1 << i] - values[mask])
    return phi, values[0]


def run(out_dir: Path = OUT_DIR, fixture_path: Path = FIXTURE_PATH) -> dict:
    df = data.load_insurance()
    y = df["charges"].to_numpy(float)

    comparison = {
        key: {"label": label, "cv": cross_validate(df, y, factory)}
        for key, (label, factory) in CANDIDATES.items()
    }

    final = Linear().fit(df, y)
    residuals = out_of_fold_residuals(df, y)
    pools = residual_pools(df, residuals)
    coverage = interval_coverage(df, y)
    inputs = encode_inputs(df)

    smoker = df["smoker"] == "yes"
    obese = df["bmi"] >= OBESE_BMI
    groups = {
        "non_smoker": float(y[~smoker].mean()),
        "smoker_bmi_under_30": float(y[smoker & ~obese].mean()),
        "smoker_bmi_30_plus": float(y[smoker & obese].mean()),
    }

    model = {
        "features": FEATURES,
        "intercept": float(final.model.intercept_),
        "coef": [float(c) for c in final.model.coef_],
        "obese_bmi": OBESE_BMI,
        "regions": REGIONS,
        "inputs": INPUTS,
        "range": list(RANGE),
        "residuals": {k: [float(v) for v in pool] for k, pool in pools.items()},
    }
    dataset = {
        "columns": ["age", "male", "bmi", "children", "smoker", "region", "charges"],
        "rows": [[*map(float, row), float(c)] for row, c in zip(inputs, y)],
    }
    meta = {
        "rows": len(df),
        "mean_charges": float(y.mean()),
        "group_means": groups,
        "models": comparison,
        "interval": coverage,
        "coefficients": dict(zip(FEATURES, model["coef"])),
        "source": data.DATASETS["insurance"].source,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model.json", model, digits=None)
    write_json(out_dir / "data.json", dataset, digits=None)
    write_json(out_dir / "meta.json", meta)

    profiles = np.array([
        [19, 0, 27.9, 0, 1, 3], [45, 1, 31.2, 2, 0, 2], [60, 0, 22.0, 3, 0, 0],
        [33, 1, 35.5, 1, 1, 1], [52, 1, 29.99, 0, 1, 2], [27, 0, 30.0, 5, 0, 3],
        [64, 1, 41.0, 0, 1, 0], [18, 0, 18.5, 0, 0, 1],
    ], dtype=float)
    fixture = {"profiles": profiles.tolist(), "prediction": [], "shapley": [], "baseline": None}
    for x in profiles:
        phi, base = shapley(final, x, inputs)
        fixture["prediction"].append(float(predict_rows(final, x[None, :])[0]))
        fixture["shapley"].append(phi.tolist())
        fixture["baseline"] = base
    write_json(fixture_path, fixture, digits=None)
    return meta


def summary(meta: dict) -> str:
    rows = [f"{r['label']:55s} R2 {r['cv']['r2']['mean']:.3f}  MAE ${r['cv']['mae']['mean']:,.0f}"
            for r in meta["models"].values()]
    iv = meta["interval"]
    rows.append(f"80% range coverage on held-out customers: {iv['coverage']:.3f} "
                f"(mean width ${iv['mean_width']:,.0f})")
    return "\n".join(rows)
