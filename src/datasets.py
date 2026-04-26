from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    source: str
    X: np.ndarray
    y: np.ndarray
    target_names: list[str] | None = None


SKLEARN_DATASETS = {
    "breast_cancer": "load_breast_cancer",
    "digits": "load_digits",
    "iris": "load_iris",
    "wine": "load_wine",
}


def list_builtin_datasets() -> list[str]:
    return sorted(f"sklearn:{name}" for name in SKLEARN_DATASETS)


def load_dataset(dataset: str) -> DatasetBundle:
    if dataset.startswith("sklearn:"):
        return _load_sklearn_dataset(dataset.split(":", 1)[1])
    if dataset.startswith("libsvm:"):
        return _load_libsvm_dataset(dataset.split(":", 1)[1])

    raise ValueError(
        "Unsupported dataset spec. Use 'sklearn:<name>' or 'libsvm:/path/to/file'."
    )


def _load_sklearn_dataset(name: str) -> DatasetBundle:
    if name not in SKLEARN_DATASETS:
        available = ", ".join(sorted(SKLEARN_DATASETS))
        raise ValueError(f"Unknown sklearn dataset '{name}'. Available: {available}")

    import sklearn.datasets as sklearn_datasets

    loader = getattr(sklearn_datasets, SKLEARN_DATASETS[name])
    data = loader()
    X = _to_dense_array(data.data)
    y = _encode_labels(data.target)
    target_names = [str(v) for v in getattr(data, "target_names", [])]
    return DatasetBundle(
        name=name,
        source=f"sklearn:{name}",
        X=X,
        y=y,
        target_names=target_names or None,
    )


def _load_libsvm_dataset(path_value: str) -> DatasetBundle:
    from sklearn.datasets import load_svmlight_file

    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LIBSVM dataset not found: {path}")

    X_sparse, y_raw = load_svmlight_file(path)
    X = _to_dense_array(X_sparse)
    y = _encode_labels(y_raw)
    return DatasetBundle(name=path.stem, source=f"libsvm:{path}", X=X, y=y)


def _to_dense_array(X) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _encode_labels(y) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder

    y = np.asarray(y)
    return LabelEncoder().fit_transform(y)
