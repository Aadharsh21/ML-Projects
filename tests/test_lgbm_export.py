import json

import lightgbm as lgb
import numpy as np
import pytest

from ml.lgbm_export import compact, export_booster, predict_raw


def _train(zero_as_missing=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, 5))
    X[rng.random(3000) < 0.25, 1] = np.nan  # learns a NaN direction
    X[rng.random(3000) < 0.30, 2] = 0.0  # many exact zeros
    y = (X[:, 0] + np.nan_to_num(X[:, 1], nan=1.0) - X[:, 2] + rng.normal(size=3000) > 0).astype(int)
    params = dict(objective="binary", num_leaves=15, learning_rate=0.1, min_data_in_leaf=20,
                  zero_as_missing=zero_as_missing, verbose=-1, seed=seed)
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=40), X


@pytest.mark.parametrize("zero_as_missing", [False, True])
def test_export_matches_lightgbm(zero_as_missing):
    booster, X = _train(zero_as_missing)
    model = export_booster(booster)
    probe = X[:500].copy()
    probe[:50, 0] = np.nan  # feature never missing in training: LightGBM treats NaN as 0
    probe[50:100, 3] = 0.0
    expected = booster.predict(probe, raw_score=True)
    np.testing.assert_allclose(predict_raw(model, probe), expected, rtol=0, atol=1e-12)


def test_export_round_trips_through_json():
    booster, X = _train()
    model = json.loads(json.dumps(compact(export_booster(booster))))
    expected = booster.predict(X[:200], raw_score=True)
    np.testing.assert_allclose(predict_raw(model, X[:200]), expected, rtol=0, atol=1e-9)


def test_export_keeps_counts_for_treeshap():
    booster, _ = _train()
    model = export_booster(booster)
    for tree in model["trees"]:
        assert len(tree["leaf_value"]) == len(tree["split_feature"]) + 1
        # every internal node's count equals the sum of its children's counts
        for i, n in enumerate(tree["count"]):
            kids = [tree["left"][i], tree["right"][i]]
            total = sum(tree["count"][k] if k >= 0 else tree["leaf_count"][~k] for k in kids)
            assert total == n
