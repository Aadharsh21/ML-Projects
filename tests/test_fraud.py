import json

import numpy as np
import pandas as pd
import pytest

from ml import fraud
from ml.lgbm_export import predict_raw


def _frame(n=20, frauds=(3, 11)):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=fraud.PCA_FEATURES)
    df["Time"] = np.arange(n) * 3600.0
    df["Amount"] = np.linspace(0, 190, n)
    df["Class"] = 0
    df.loc[list(frauds), "Class"] = 1
    return df


def test_features_encode_hour_of_day_on_a_circle():
    df = _frame()
    df.loc[0, "Time"] = 0.0
    df.loc[1, "Time"] = 6 * 3600.0
    df.loc[2, "Time"] = 86400.0 + 3600.0  # 1am on day two
    X = fraud.features(df)
    assert list(X.columns) == fraud.FEATURES
    assert X.loc[0, "HourSin"] == pytest.approx(0.0)
    assert X.loc[0, "HourCos"] == pytest.approx(1.0)
    assert X.loc[1, "HourSin"] == pytest.approx(1.0)
    assert X.loc[2, "HourSin"] == pytest.approx(np.sin(2 * np.pi / 24))
    assert X.loc[0, "LogAmount"] == pytest.approx(np.log1p(df.loc[0, "Amount"]))


def test_split_is_chronological():
    s = fraud.split(_frame(n=100))
    assert s["train"].sum() == 60 and s["val"].sum() == 15 and s["test"].sum() == 25
    assert np.flatnonzero(s["train"]).max() < np.flatnonzero(s["val"]).min()
    assert np.flatnonzero(s["val"]).max() < np.flatnonzero(s["test"]).min()


def test_undersample_keeps_every_fraud():
    df = _frame()
    bal = fraud.undersample(df)
    assert bal["Class"].sum() == 2 and len(bal) == 4


def test_money_counts_blocked_fraud_and_review_cost():
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 50.0, 20.0, 5.0])
    flagged = np.array([True, True, False, False])
    m = fraud.money(y, amount, flagged, review_cost=5.0)
    assert m == {"blocked": 100.0, "missed": 20.0, "review_cost": 10.0,
                 "net_savings": 90.0, "alerts": 2}


def test_quantised_scores_never_cross_one_half():
    p = np.array([0.4999999, 0.5, 0.5000001, 1e-20, 1 - 1e-20])
    q = np.array(fraud._logit_milli(p))
    assert (q >= 0).tolist() == (p >= 0.5).tolist()
    assert q.min() == -15000 and q.max() == 15000


def test_exported_model_reproduces_fixture():
    model_path = fraud.OUT_DIR / "model.json"
    if not model_path.exists() or not fraud.FIXTURE_PATH.exists():
        pytest.skip("run `python -m ml fraud` first")
    model = json.loads(model_path.read_text())
    fixture = json.loads(fraud.FIXTURE_PATH.read_text())
    X = np.array(fixture["features"])
    p = 1 / (1 + np.exp(-predict_raw(model, X)))
    np.testing.assert_allclose(p, fixture["probability"], atol=1e-9)
