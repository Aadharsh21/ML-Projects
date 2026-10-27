import numpy as np

from ml import clustering, data


def test_matched_confusion_ignores_cluster_numbering():
    species = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([2, 2, 0, 0, 1, 1])  # perfect clustering, different ids
    out = clustering.matched_confusion(species, labels)
    assert out["matched_accuracy"] == 1.0
    assert out["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_matched_confusion_with_more_clusters_than_species():
    species = np.array([0, 0, 1, 1, 1, 2])
    labels = np.array([0, 3, 1, 1, 1, 2])
    out = clustering.matched_confusion(species, labels)
    assert out["matched_accuracy"] == 5 / 6
    assert len(out["matrix"][0]) == 4


def test_sweep_inertia_falls_as_k_grows():
    X = data.load_iris()[clustering.FEATURES].to_numpy()
    species = data.load_iris()["species"].cat.codes.to_numpy()
    rows = clustering.sweep(X, species)
    inertia = [r["inertia"] for r in rows]
    assert inertia == sorted(inertia, reverse=True)
    assert rows[0]["silhouette"] is None
    best = max(rows[1:], key=lambda r: r["silhouette"])
    assert best["k"] == 2  # setosa vs the rest: the honest answer for iris
