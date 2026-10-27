import os

import pytest

from ml import data


def require_dataset(name: str) -> None:
    """Skip a test when the dataset is not cached and downloads are disabled."""
    path = data.data_dir() / data.DATASETS[name].filename
    if not path.exists() and os.environ.get("ML_PROJECTS_OFFLINE") == "1":
        pytest.skip(f"{name} dataset not available offline")
