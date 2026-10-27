"""Iris clustering: how well do unsupervised methods rediscover the three species?

v1 (notebooks/v1/clustering.ipynb) ran k-means with k=3 and k=6 and plotted the
clusters. v2 chooses k with the elbow and silhouette methods, checks the effect of
feature scaling, scores every clustering against the true species with the
adjusted Rand index, and compares k-means with a Gaussian mixture, whose
elongated clusters fit the iris species far better.

Run with ``python -m ml clustering``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import data
from .evaluation import write_json

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = ["setosa", "versicolor", "virginica"]
K_RANGE = range(1, 9)

OUT_DIR = data.REPO_ROOT / "web" / "public" / "data" / "iris"
FIXTURE_PATH = data.REPO_ROOT / "web" / "tests" / "fixtures" / "iris.json"


def matched_confusion(species: np.ndarray, labels: np.ndarray) -> dict:
    """Confusion matrix after matching each cluster to its best species
    (Hungarian algorithm), plus the share of flowers that land in their species' cluster."""
    cm = confusion_matrix(species, labels)
    rows, cols = linear_sum_assignment(-cm)
    matched_to = dict(zip(rows, cols))
    order = [matched_to[r] for r in range(cm.shape[0]) if r in matched_to]
    order += [c for c in range(cm.shape[1]) if c not in order]
    return {
        "matrix": cm[:, order].tolist(),
        "matched_accuracy": float(cm[rows, cols].sum() / cm.sum()),
    }


def sweep(X: np.ndarray, species: np.ndarray, seed: int = 0) -> list[dict]:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km.labels_)) if k > 1 else None,
            "ari": float(adjusted_rand_score(species, km.labels_)),
        })
    return rows


def run(out_dir: Path = OUT_DIR, fixture_path: Path = FIXTURE_PATH) -> dict:
    df = data.load_iris()
    X = df[FEATURES].to_numpy(float)
    species = df["species"].cat.codes.to_numpy()
    Xs = StandardScaler().fit_transform(X)

    raw_sweep = sweep(X, species)
    scaled_sweep = sweep(Xs, species)

    km3 = KMeans(n_clusters=3, n_init=20, random_state=0).fit(X)
    gmm_bic = []
    for k in range(1, 7):
        g = GaussianMixture(k, covariance_type="full", n_init=10, random_state=0).fit(X)
        gmm_bic.append({"k": k, "bic": float(g.bic(X))})
    gmm3 = GaussianMixture(3, covariance_type="full", n_init=10, random_state=0).fit(X)
    ward3 = AgglomerativeClustering(3, linkage="ward").fit(X)

    methods = {
        "kmeans_raw": ("k-means, k=3, raw measurements", km3.labels_),
        "kmeans_scaled": ("k-means, k=3, standardized",
                          KMeans(n_clusters=3, n_init=20, random_state=0).fit(Xs).labels_),
        "ward_raw": ("Agglomerative (Ward), k=3", ward3.labels_),
        "gmm_raw": ("Gaussian mixture, k=3, full covariance", gmm3.predict(X)),
    }
    comparison = {
        key: {"label": label, "ari": float(adjusted_rand_score(species, labels)),
              **matched_confusion(species, labels)}
        for key, (label, labels) in methods.items()
    }

    pca = PCA(2).fit(X)
    meta = {
        "features": FEATURES,
        "species": SPECIES,
        "sweep": {"raw": raw_sweep, "scaled": scaled_sweep},
        "gmm_bic": gmm_bic,
        "methods": comparison,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "data.json", {
        "features": FEATURES,
        "species": SPECIES,
        "rows": [[*map(float, x), int(s)] for x, s in zip(X, species)],
        "pca": {"mean": pca.mean_.tolist(), "components": pca.components_.tolist()},
    }, digits=None)
    write_json(out_dir / "meta.json", meta)

    # Reference runs for the browser implementations.
    rng = np.random.default_rng(11)
    kmeans_runs = []
    for k in (2, 3, 5):
        init = X[rng.choice(len(X), size=k, replace=False)]
        km = KMeans(n_clusters=k, init=init, n_init=1, max_iter=300, tol=0,
                    algorithm="lloyd").fit(X)
        kmeans_runs.append({"init": init.tolist(), "centers": km.cluster_centers_.tolist(),
                            "labels": km.labels_.tolist(), "inertia": float(km.inertia_)})
    gmm_runs = []
    for k, steps in ((2, 1), (3, 1), (3, 5), (3, 50)):
        init_means = X[rng.choice(len(X), size=k, replace=False)]
        init_cov = np.array([np.cov(X.T) for _ in range(k)])
        weights = np.full(k, 1 / k)
        g = GaussianMixture(k, covariance_type="full", max_iter=steps, tol=0, reg_covar=1e-6,
                            weights_init=weights, means_init=init_means,
                            precisions_init=np.linalg.inv(init_cov))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g.fit(X)
        gmm_runs.append({
            "steps": steps, "init_means": init_means.tolist(), "init_cov": init_cov.tolist(),
            "weights": g.weights_.tolist(), "means": g.means_.tolist(),
            "covariances": g.covariances_.tolist(), "labels": g.predict(X).tolist(),
        })
    write_json(fixture_path, {
        "kmeans": kmeans_runs,
        "gmm": gmm_runs,
        "silhouette_k3": float(silhouette_score(X, km3.labels_)),
        "ari_k3": float(adjusted_rand_score(species, km3.labels_)),
        "labels_k3": km3.labels_.tolist(),
    }, digits=None)
    return meta


def summary(meta: dict) -> str:
    rows = []
    for r in meta["sweep"]["raw"]:
        sil = f"{r['silhouette']:.3f}" if r["silhouette"] is not None else "  -  "
        rows.append(f"k={r['k']}  inertia {r['inertia']:7.2f}  silhouette {sil}  ARI {r['ari']:.3f}")
    for m in meta["methods"].values():
        rows.append(f"{m['label']:42s} ARI {m['ari']:.3f}  matched {m['matched_accuracy']:.1%}")
    return "\n".join(rows)
