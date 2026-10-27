import json

import numpy as np
import pandas as pd
import pytest

from conftest import require_dataset
from ml import data, insurance


def _people():
    return pd.DataFrame({
        "age": [19, 40, 52], "sex": ["female", "male", "male"], "bmi": [27.9, 30.0, 35.2],
        "children": [0, 2, 1], "smoker": ["yes", "yes", "no"],
        "region": ["southwest", "northeast", "southeast"],
    })


def test_design_matrix():
    X = insurance.design(_people())
    assert list(X.columns) == insurance.FEATURES
    assert X["obese"].tolist() == [0.0, 1.0, 1.0]  # BMI 30 counts as obese
    assert X["smoker_obese"].tolist() == [0.0, 1.0, 0.0]  # only smokers get the jump
    assert X["smoker_bmi"].tolist() == [27.9, 30.0, 0.0]
    assert X["region_southwest"].tolist() == [1.0, 0.0, 0.0]
    assert X.loc[1, ["region_northwest", "region_southeast", "region_southwest"]].sum() == 0


def test_v1_replica_matches_original_notebook():
    require_dataset("insurance")
    df = data.load_insurance()
    v1 = insurance.AgeOnlyBySmoker().fit(df, df["charges"].to_numpy())
    ask = pd.DataFrame({"age": [23, 37, 54, 23, 37, 45],
                        "smoker": ["yes"] * 3 + ["no"] * 3})
    # values printed by notebooks/v1/LINEARINSURANCE.ipynb
    expected = [27314.59297544, 31587.91940498, 36776.95864085,
                4055.30443855, 7796.78921819, 9934.78052084]
    np.testing.assert_allclose(v1.predict(ask), expected, rtol=1e-9)


def test_shapley_values_add_up_to_the_prediction():
    require_dataset("insurance")
    df = data.load_insurance()
    model = insurance.Linear().fit(df, df["charges"].to_numpy())
    background = insurance.encode_inputs(df)
    x = np.array([45, 1, 33.0, 2, 1, 2], dtype=float)
    phi, base = insurance.shapley(model, x, background)
    pred = insurance.predict_rows(model, x[None, :])[0]
    assert phi.sum() + base == pytest.approx(pred, abs=1e-6)
    assert base == pytest.approx(df["charges"].mean(), abs=1e-6)
    # smoking is the biggest driver for an obese smoker
    assert np.argmax(phi) == insurance.INPUTS.index("smoker")


def test_fixture_is_consistent():
    if not insurance.FIXTURE_PATH.exists():
        pytest.skip("run `python -m ml insurance` first")
    fx = json.loads(insurance.FIXTURE_PATH.read_text())
    for pred, phi in zip(fx["prediction"], fx["shapley"]):
        assert sum(phi) + fx["baseline"] == pytest.approx(pred, abs=1e-6)
