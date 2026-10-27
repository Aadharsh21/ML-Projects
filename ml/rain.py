"""Rain tomorrow: predict next-day rain at 49 Australian weather stations.

v1 (notebooks/v1/LOGISTICWEATHER.ipynb) trained a logistic regression on one-hot
encoded readings. v2 keeps the same time-based split so the numbers are directly
comparable, fixes a leak (v1 fit its imputer and scaler on all rows, including the
test years), adds physically meaningful features, and swaps in gradient boosted
trees that handle missing readings natively.

Run with ``python -m ml rain``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from . import data
from .evaluation import binary_metrics, bootstrap_ci, calibration_table, write_json
from .lgbm_export import compact, export_booster

# Approximate coordinates of each station (degrees). Used as features so the model
# can learn regional climate, and to draw the map in the web app.
STATIONS: dict[str, tuple[float, float]] = {
    "Adelaide": (-34.93, 138.60), "Albany": (-35.02, 117.88), "Albury": (-36.08, 146.92),
    "AliceSprings": (-23.70, 133.88), "BadgerysCreek": (-33.88, 150.74),
    "Ballarat": (-37.56, 143.85), "Bendigo": (-36.76, 144.28), "Brisbane": (-27.47, 153.03),
    "Cairns": (-16.92, 145.77), "Canberra": (-35.28, 149.13), "Cobar": (-31.50, 145.84),
    "CoffsHarbour": (-30.30, 153.11), "Dartmoor": (-37.92, 141.27), "Darwin": (-12.46, 130.84),
    "GoldCoast": (-28.02, 153.40), "Hobart": (-42.88, 147.33), "Katherine": (-14.47, 132.26),
    "Launceston": (-41.44, 147.14), "Melbourne": (-37.81, 144.96),
    "MelbourneAirport": (-37.67, 144.84), "Mildura": (-34.21, 142.14), "Moree": (-29.47, 149.84),
    "MountGambier": (-37.83, 140.78), "MountGinini": (-35.53, 148.77),
    "Newcastle": (-32.93, 151.78), "Nhil": (-36.33, 141.65), "NorahHead": (-33.28, 151.57),
    "NorfolkIsland": (-29.04, 167.95), "Nuriootpa": (-34.47, 138.99),
    "PearceRAAF": (-31.67, 116.02), "Penrith": (-33.75, 150.69), "Perth": (-31.95, 115.86),
    "PerthAirport": (-31.94, 115.97), "Portland": (-38.34, 141.60), "Richmond": (-33.60, 150.75),
    "Sale": (-38.10, 147.07), "SalmonGums": (-32.98, 121.64), "Sydney": (-33.87, 151.21),
    "SydneyAirport": (-33.94, 151.17), "Townsville": (-19.26, 146.82),
    "Tuggeranong": (-35.42, 149.09), "Uluru": (-25.34, 131.04), "WaggaWagga": (-35.12, 147.37),
    "Walpole": (-34.98, 116.73), "Watsonia": (-37.71, 145.08), "Williamtown": (-32.80, 151.84),
    "Witchcliffe": (-34.03, 115.10), "Wollongong": (-34.42, 150.89), "Woomera": (-31.20, 136.83),
}

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

RAW_NUMERIC = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
               "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
               "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]
WIND_DIRS = ["WindGustDir", "WindDir9am", "WindDir3pm"]

FEATURES = RAW_NUMERIC + [
    "RainToday", "Latitude", "Longitude", "SeasonSin", "SeasonCos",
    "WindGustDirSin", "WindGustDirCos", "WindDir9amSin", "WindDir9amCos",
    "WindDir3pmSin", "WindDir3pmCos",
    "TempRange", "HumidityChange", "PressureChange", "TempChange", "CloudChange",
]

# How SHAP contributions are grouped and labelled in the app.
GROUPS: list[tuple[str, list[str]]] = [
    ("Humidity at 3pm", ["Humidity3pm"]),
    ("Humidity at 9am", ["Humidity9am"]),
    ("Humidity change, 9am to 3pm", ["HumidityChange"]),
    ("Pressure at 3pm", ["Pressure3pm"]),
    ("Pressure at 9am", ["Pressure9am"]),
    ("Pressure change, 9am to 3pm", ["PressureChange"]),
    ("Sunshine", ["Sunshine"]),
    ("Cloud at 3pm", ["Cloud3pm"]),
    ("Cloud at 9am", ["Cloud9am"]),
    ("Cloud change, 9am to 3pm", ["CloudChange"]),
    ("Wind gust speed", ["WindGustSpeed"]),
    ("Wind gust direction", ["WindGustDirSin", "WindGustDirCos"]),
    ("Wind speed at 9am", ["WindSpeed9am"]),
    ("Wind speed at 3pm", ["WindSpeed3pm"]),
    ("Wind direction at 9am", ["WindDir9amSin", "WindDir9amCos"]),
    ("Wind direction at 3pm", ["WindDir3pmSin", "WindDir3pmCos"]),
    ("Rainfall today", ["Rainfall"]),
    ("Rained today", ["RainToday"]),
    ("Max temperature", ["MaxTemp"]),
    ("Min temperature", ["MinTemp"]),
    ("Temperature range", ["TempRange"]),
    ("Temperature at 9am", ["Temp9am"]),
    ("Temperature at 3pm", ["Temp3pm"]),
    ("Temperature change, 9am to 3pm", ["TempChange"]),
    ("Evaporation", ["Evaporation"]),
    ("Time of year", ["SeasonSin", "SeasonCos"]),
    ("Station location", ["Latitude", "Longitude"]),
]

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    seed=42,
    deterministic=True,
    force_row_wise=True,
    num_threads=2,
    verbose=-1,
)

OUT_DIR = data.REPO_ROOT / "web" / "public" / "data" / "rain"
FIXTURE_PATH = data.REPO_ROOT / "web" / "tests" / "fixtures" / "rain.json"


def compass_vectors() -> dict[str, tuple[float, float]]:
    """(sin, cos) of each compass point, so a NNW wind sits next to a N wind."""
    return {
        d: (math.sin((i * 22.5) * (math.pi / 180)), math.cos((i * 22.5) * (math.pi / 180)))
        for i, d in enumerate(COMPASS)
    }


def pretty_station(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for each row. Missing readings stay NaN; LightGBM routes them."""
    X = pd.DataFrame(index=df.index)
    for c in RAW_NUMERIC:
        X[c] = df[c].astype(float)
    X["RainToday"] = df["RainToday"].map({"Yes": 1.0, "No": 0.0}).astype(float)
    X["Latitude"] = df["Location"].map(lambda s: STATIONS[s][0]).astype(float)
    X["Longitude"] = df["Location"].map(lambda s: STATIONS[s][1]).astype(float)
    doy = df["Date"].dt.dayofyear.astype(float)
    X["SeasonSin"] = np.sin(2 * math.pi * doy / 365.25)
    X["SeasonCos"] = np.cos(2 * math.pi * doy / 365.25)
    vectors = compass_vectors()
    for c in WIND_DIRS:
        X[c + "Sin"] = df[c].map(lambda d: vectors[d][0] if isinstance(d, str) else np.nan).astype(float)
        X[c + "Cos"] = df[c].map(lambda d: vectors[d][1] if isinstance(d, str) else np.nan).astype(float)
    X["TempRange"] = X["MaxTemp"] - X["MinTemp"]
    X["HumidityChange"] = X["Humidity3pm"] - X["Humidity9am"]
    X["PressureChange"] = X["Pressure3pm"] - X["Pressure9am"]
    X["TempChange"] = X["Temp3pm"] - X["Temp9am"]
    X["CloudChange"] = X["Cloud3pm"] - X["Cloud9am"]
    return X[FEATURES]


def target(df: pd.DataFrame) -> np.ndarray:
    return (df["RainTomorrow"] == "Yes").to_numpy().astype(int)


def split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Same split as v1: train before 2015, validate on 2015, test on 2016-2017."""
    year = df["Date"].dt.year.to_numpy()
    return {"train": year < 2015, "val": year == 2015, "test": year > 2015}


def v1_logistic() -> Pipeline:
    """The v1 notebook's model, rebuilt as a pipeline so preprocessing only sees
    training rows."""
    categorical = ["Location"] + WIND_DIRS + ["RainToday"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]),
         RAW_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(solver="liblinear"))])


def v2_logistic() -> Pipeline:
    """Logistic regression on the v2 features, to separate feature gains from model gains."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=3000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )


def _plain(v):
    """JSON-friendly cell value: None for missing, int when integral."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    v = float(v)
    return int(v) if v.is_integer() else v


def encode_days(df: pd.DataFrame) -> dict:
    """Columnar encoding of raw test-set rows for the web app."""
    df = df.sort_values(["Location", "Date"])
    start = df["Date"].min()
    stations = sorted(STATIONS)
    index = {}
    for name, g in df.groupby("Location", sort=True):
        positions = np.flatnonzero(df["Location"].to_numpy() == name)
        index[name] = [int(positions[0]), int(len(g))]
    compass_index = {d: i for i, d in enumerate(COMPASS)}
    columns: dict[str, list] = {
        "day": [int(d) for d in (df["Date"] - start).dt.days],
    }
    for c in RAW_NUMERIC:
        columns[c] = [_plain(v) for v in df[c].to_numpy(dtype=float)]
    for c in WIND_DIRS:
        columns[c] = [compass_index[v] if isinstance(v, str) else None for v in df[c]]
    columns["RainToday"] = [1 if v == "Yes" else 0 for v in df["RainToday"]]
    columns["RainTomorrow"] = [1 if v == "Yes" else 0 for v in df["RainTomorrow"]]
    return {
        "start": start.strftime("%Y-%m-%d"),
        "stations": stations,
        "index": {s: index.get(s, [0, 0]) for s in stations},
        "columns": columns,
    }


def run(out_dir: Path = OUT_DIR, fixture_path: Path = FIXTURE_PATH) -> dict:
    df = data.load_weather()
    X = features(df)
    y = target(df)
    s = split(df)
    tr, va, te = s["train"], s["val"], s["test"]

    results: dict[str, dict] = {}

    def record(name: str, p_val: np.ndarray, p_test: np.ndarray, label: str) -> None:
        results[name] = {
            "label": label,
            "val": binary_metrics(y[va], p_val),
            "test": binary_metrics(y[te], p_test),
        }

    # Baselines: always "no rain", and "tomorrow will be like today".
    record("majority", np.zeros(va.sum()), np.zeros(te.sum()), "Always predict no rain")
    record("persistence", X.loc[va, "RainToday"].to_numpy(), X.loc[te, "RainToday"].to_numpy(),
           "Tomorrow repeats today")

    v1 = v1_logistic().fit(df[tr], y[tr])
    record("v1_logistic", v1.predict_proba(df[va])[:, 1], v1.predict_proba(df[te])[:, 1],
           "v1: logistic regression, original features")

    lr2 = v2_logistic().fit(X[tr], y[tr])
    record("v2_logistic", lr2.predict_proba(X[va])[:, 1], lr2.predict_proba(X[te])[:, 1],
           "Logistic regression, v2 features")

    booster = train_lightgbm(X[tr], y[tr], X[va], y[va])
    p_val = booster.predict(X[va], num_iteration=booster.best_iteration)
    p_test = booster.predict(X[te], num_iteration=booster.best_iteration)
    record("v2_lightgbm", p_val, p_test, "v2: LightGBM, v2 features")

    yt = y[te]
    ci = {
        "accuracy": bootstrap_ci(yt, p_test, lambda a, b: float(((b >= 0.5) == a).mean())),
        "roc_auc": bootstrap_ci(yt, p_test, roc_auc_score),
    }

    # SHAP contributions on the test years, grouped into readable inputs.
    contrib = booster.predict(X[te], num_iteration=booster.best_iteration, pred_contrib=True)
    col = {f: i for i, f in enumerate(FEATURES)}
    importance = []
    for label, cols in GROUPS:
        values = contrib[:, [col[c] for c in cols]].sum(axis=1)
        importance.append({"group": label, "mean_abs_shap": float(np.abs(values).mean())})
    importance.sort(key=lambda r: -r["mean_abs_shap"])

    test_df = df[te]
    per_station = []
    for name in sorted(STATIONS):
        m = (test_df["Location"] == name).to_numpy()
        if m.sum() == 0:
            continue
        ys, ps = yt[m], p_test[m]
        per_station.append({
            "station": name,
            "days": int(m.sum()),
            "rain_rate": float(ys.mean()),
            "accuracy": float(((ps >= 0.5) == ys).mean()),
            "roc_auc": float(roc_auc_score(ys, ps)) if 0 < ys.sum() < len(ys) else None,
        })

    def window(mask):
        d = df.loc[mask, "Date"]
        return {"rows": int(mask.sum()), "start": d.min().strftime("%Y-%m-%d"),
                "end": d.max().strftime("%Y-%m-%d"), "rain_rate": float(y[mask].mean())}

    test_missing = {
        name: {c: float(g[c].isna().mean()) for c in RAW_NUMERIC + WIND_DIRS}
        for name, g in test_df.groupby("Location")
    }
    meta = {
        "stations": [
            {"id": name, "name": pretty_station(name), "lat": lat, "lon": lon,
             "missing": test_missing.get(name, {})}
            for name, (lat, lon) in sorted(STATIONS.items())
        ],
        "compass": COMPASS,
        "compass_vectors": [list(compass_vectors()[d]) for d in COMPASS],
        "features": FEATURES,
        "groups": [{"label": label, "features": cols} for label, cols in GROUPS],
        "splits": {"train": window(tr), "val": window(va), "test": window(te)},
        "models": results,
        "best_iteration": booster.best_iteration,
        "ci95": ci,
        "calibration": calibration_table(yt, p_test),
        "importance": importance,
        "per_station": per_station,
        "source": data.DATASETS["weather"].source,
    }

    model = compact(export_booster(booster, num_iteration=booster.best_iteration))
    model["base_value"] = float(contrib[0, -1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model.json", model, digits=None)
    write_json(out_dir / "days.json", encode_days(test_df), digits=None)
    write_json(out_dir / "meta.json", meta)

    # Parity fixtures for the browser engine: a spread of test rows with the
    # features, probabilities and SHAP values LightGBM itself produces.
    rng = np.random.default_rng(7)
    picks = np.sort(rng.choice(np.flatnonzero(te), size=80, replace=False))
    fx = X.loc[picks]
    fixture = {
        "rows": [
            {"station": df.at[i, "Location"], "date": df.at[i, "Date"].strftime("%Y-%m-%d")}
            for i in picks
        ],
        "features": [[_plain(v) for v in row] for row in fx.to_numpy()],
        "probability": booster.predict(fx, num_iteration=booster.best_iteration).tolist(),
        "shap": booster.predict(fx, num_iteration=booster.best_iteration,
                                pred_contrib=True).tolist(),
    }
    write_json(fixture_path, fixture, digits=None)
    return meta


def summary(meta: dict) -> str:
    rows = []
    for key, r in meta["models"].items():
        t = r["test"]
        rows.append(f"{r['label']:45s} acc {t['accuracy']:.4f}  auc {t['roc_auc']:.4f}  "
                    f"ap {t['avg_precision']:.4f}  brier {t['brier']:.4f}")
    return "\n".join(rows)
