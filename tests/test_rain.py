import json
import math

import numpy as np
import pandas as pd
import pytest

from ml import rain
from ml.lgbm_export import predict_raw


def _rows():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2014-12-31", "2015-06-01", "2016-01-02"]),
        "Location": ["Sydney", "Darwin", "Hobart"],
        "MinTemp": [13.4, 20.0, np.nan], "MaxTemp": [22.9, 31.5, 18.0],
        "Rainfall": [0.6, 0.0, 12.2], "Evaporation": [np.nan, 6.2, 2.0],
        "Sunshine": [np.nan, 10.1, 1.5], "WindGustDir": ["N", "E", None],
        "WindGustSpeed": [44.0, 30.0, 61.0], "WindDir9am": ["NNW", "SE", "S"],
        "WindDir3pm": ["W", None, "SW"], "WindSpeed9am": [20.0, 9.0, 15.0],
        "WindSpeed3pm": [24.0, 11.0, 30.0], "Humidity9am": [71.0, 60.0, 90.0],
        "Humidity3pm": [22.0, 40.0, 85.0], "Pressure9am": [1007.7, 1010.0, 1001.2],
        "Pressure3pm": [1007.1, 1008.2, 999.0], "Cloud9am": [8.0, np.nan, 7.0],
        "Cloud3pm": [np.nan, 2.0, 8.0], "Temp9am": [16.9, 25.0, 12.0],
        "Temp3pm": [21.8, 30.1, 16.0], "RainToday": ["No", "No", "Yes"],
        "RainTomorrow": ["No", "Yes", "Yes"],
    })


def test_feature_columns_and_engineering():
    X = rain.features(_rows())
    assert list(X.columns) == rain.FEATURES
    assert X.loc[0, "TempRange"] == pytest.approx(22.9 - 13.4)
    assert math.isnan(X.loc[2, "TempRange"])  # missing MinTemp stays missing
    assert X.loc[0, "PressureChange"] == pytest.approx(-0.6)
    assert X.loc[2, "RainToday"] == 1.0
    assert (X.loc[0, "Latitude"], X.loc[0, "Longitude"]) == rain.STATIONS["Sydney"]


def test_wind_direction_is_a_unit_vector():
    X = rain.features(_rows())
    assert X.loc[0, "WindGustDirSin"] == pytest.approx(0.0)  # N
    assert X.loc[0, "WindGustDirCos"] == pytest.approx(1.0)
    assert X.loc[1, "WindGustDirSin"] == pytest.approx(1.0)  # E
    assert math.isnan(X.loc[2, "WindGustDirSin"])
    # neighbouring compass points are close, opposite points are far apart
    v = rain.compass_vectors()
    assert np.hypot(v["N"][0] - v["NNW"][0], v["N"][1] - v["NNW"][1]) < 0.4
    assert np.hypot(v["N"][0] - v["S"][0], v["N"][1] - v["S"][1]) == pytest.approx(2.0)


def test_split_by_year():
    s = rain.split(_rows())
    assert s["train"].tolist() == [True, False, False]
    assert s["val"].tolist() == [False, True, False]
    assert s["test"].tolist() == [False, False, True]


def test_pretty_station_names():
    assert rain.pretty_station("AliceSprings") == "Alice Springs"
    assert rain.pretty_station("PearceRAAF") == "Pearce RAAF"


def test_exported_model_reproduces_fixture():
    model_path = rain.OUT_DIR / "model.json"
    if not model_path.exists() or not rain.FIXTURE_PATH.exists():
        pytest.skip("run `python -m ml rain` first")
    model = json.loads(model_path.read_text())
    fixture = json.loads(rain.FIXTURE_PATH.read_text())
    X = np.array([[np.nan if v is None else v for v in row] for row in fixture["features"]])
    p = 1 / (1 + np.exp(-predict_raw(model, X)))
    np.testing.assert_allclose(p, fixture["probability"], atol=1e-9)
    shap = np.array(fixture["shap"])
    np.testing.assert_allclose(shap.sum(axis=1), predict_raw(model, X), atol=1e-6)
    assert shap[:, -1] == pytest.approx(model["base_value"])
