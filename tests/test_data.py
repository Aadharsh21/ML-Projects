import hashlib

import pytest

from ml import data


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n3,4\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    monkeypatch.setenv("ML_PROJECTS_DATA", str(cache))
    monkeypatch.setitem(
        data.DATASETS, "fake", data.Dataset("fake.csv", source.as_uri(), sha, "test fixture")
    )
    return cache


def test_fetch_downloads_and_caches(fake_dataset):
    path = data.fetch("fake")
    assert path == fake_dataset / "fake.csv"
    assert path.read_text().startswith("a,b")
    # second call reuses the cached copy
    assert data.fetch("fake") == path


def test_fetch_rejects_corrupt_download(fake_dataset, monkeypatch):
    ds = data.DATASETS["fake"]
    monkeypatch.setitem(data.DATASETS, "fake", data.Dataset(ds.filename, ds.url, "0" * 64, ds.source))
    with pytest.raises(OSError, match="Checksum mismatch"):
        data.fetch("fake")
    assert not (fake_dataset / "fake.csv").exists()


def test_iris_is_bundled():
    df = data.load_iris()
    assert df.shape == (150, 5)
    assert df["species"].value_counts().tolist() == [50, 50, 50]
