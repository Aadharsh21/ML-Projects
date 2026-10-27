"""Dataset downloads with checksum verification.

The original notebooks pulled data from Kaggle, which needs an account and an API
key. These loaders fetch the same files from public mirrors instead, verify them
against a known SHA-256, and cache them under ``data/`` (or ``$ML_PROJECTS_DATA``).
If you already have a copy, drop it into that folder with the same file name.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("ML_PROJECTS_DATA", REPO_ROOT / "data"))


@dataclass(frozen=True)
class Dataset:
    filename: str
    url: str
    sha256: str
    source: str


DATASETS: dict[str, Dataset] = {
    "weather": Dataset(
        filename="weatherAUS.csv",
        url="https://raw.githubusercontent.com/amankharwal/Website-data/master/weatherAUS.csv",
        sha256="a4f2b071196fef2a06849899187ff91dbcb097baf573e49391f0b75692fecb2b",
        source="Rain in Australia (Kaggle: jsphyg/weather-dataset-rattle-package), "
        "observations from the Australian Bureau of Meteorology",
    ),
    "creditcard": Dataset(
        filename="creditcard.csv",
        url="https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
        sha256="33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59",
        source="Credit Card Fraud Detection (Kaggle: mlg-ulb/creditcardfraud), "
        "Worldline and the ULB Machine Learning Group",
    ),
    "insurance": Dataset(
        filename="medical-charges.csv",
        url="https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv",
        sha256="388eff679557d08ac19f463d025de5e0b4adc482537c8456d19934d78621fd47",
        source="Medical Cost Personal Datasets (Kaggle: mirichoi0218/insurance)",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str) -> Path:
    """Return the local path of a dataset, downloading and verifying it if needed."""
    ds = DATASETS[name]
    target = data_dir() / ds.filename
    if target.exists():
        if _sha256(target) != ds.sha256:
            warnings.warn(
                f"{target} does not match the expected checksum; using it anyway. "
                "Results may differ slightly from the published numbers.",
                stacklevel=2,
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ds.filename} ...", flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        with urllib.request.urlopen(ds.url, timeout=120) as response:
            shutil.copyfileobj(response, tmp)
        tmp_path = Path(tmp.name)
    actual = _sha256(tmp_path)
    if actual != ds.sha256:
        tmp_path.unlink()
        raise OSError(f"Checksum mismatch for {ds.filename}: expected {ds.sha256}, got {actual}")
    tmp_path.replace(target)
    return target


def load_weather() -> pd.DataFrame:
    """Daily weather observations from 49 Australian stations, 2007-2017.

    Rows without a RainToday or RainTomorrow label are dropped. RISK_MM (tomorrow's
    rainfall, present in older copies of the file) is removed because it is the
    answer in disguise.
    """
    df = pd.read_csv(fetch("weather"), float_precision="round_trip")
    df = df.drop(columns=["RISK_MM"], errors="ignore")
    df = df.dropna(subset=["RainToday", "RainTomorrow"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_creditcard() -> pd.DataFrame:
    """284,807 card transactions over two days in September 2013, sorted by time."""
    df = pd.read_csv(fetch("creditcard"), float_precision="round_trip")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def load_insurance() -> pd.DataFrame:
    """1,338 US customers with demographics and yearly medical charges."""
    return pd.read_csv(fetch("insurance"), float_precision="round_trip")


def load_iris() -> pd.DataFrame:
    """Fisher's iris measurements (bundled with scikit-learn, no download)."""
    from sklearn.datasets import load_iris as _load

    bunch = _load(as_frame=True)
    df = bunch.data.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df["species"] = pd.Categorical.from_codes(bunch.target, bunch.target_names)
    return df
