from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets import DatasetBundle
from models import build_model
from strategies import StrategySpec


@dataclass
class ExperimentConfig:
    dataset: DatasetBundle
    strategy: StrategySpec
    model_name: str
    seed: int
    n_trials: int
    test_size: float
    init_labels: int
    budget: int
    batch_size: int
    scale: bool
    output_dir: Path
    run_name: str | None = None


@dataclass
class QueryState:
    X: np.ndarray
    y: np.ndarray
    labeled_idx: np.ndarray
    unlabeled_idx: np.ndarray
    model: object
    rng: np.random.Generator
    seed: int


def run_experiment(config: ExperimentConfig) -> Path:
    run_name = config.run_name or _default_run_name(config)
    run_dir = config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    trial_summaries = []
    learning_curves = {}
    f1_curves = {}
    detail_rows = []
    class_rows = []

    for trial_offset in range(config.n_trials):
        seed = config.seed + trial_offset
        result = _run_trial(config, seed)
        trial_summaries.append(
            {
                "seed": seed,
                "train_aubc": _aubc(result["round"], result["train_accuracy"]),
                "test_aubc": _aubc(result["round"], result["test_accuracy"]),
            }
        )
        learning_curves[seed] = pd.Series(
            result["test_accuracy"], index=result["round"], dtype=float
        )
        f1_curves[seed] = pd.Series(result["f1"], index=result["round"], dtype=float)
        detail_rows.extend(result["detail"])
        class_rows.append({"seed": seed, "class_dist": json.dumps(result["class_dist"])})

    pd.DataFrame(trial_summaries).to_csv(run_dir / "aubc.csv", index=False)
    pd.DataFrame(learning_curves).T.to_csv(run_dir / "learn_curve.csv")
    pd.DataFrame(f1_curves).T.to_csv(run_dir / "f1_score_curve.csv")
    pd.DataFrame(detail_rows).to_csv(run_dir / "detail.csv", index=False)
    pd.DataFrame(class_rows).to_csv(run_dir / "class_dist.csv", index=False)

    metadata = {
        "dataset": config.dataset.source,
        "strategy": config.strategy.name,
        "strategy_category": config.strategy.category,
        "strategy_framework": config.strategy.framework,
        "model": config.model_name,
        "seed": config.seed,
        "n_trials": config.n_trials,
        "test_size": config.test_size,
        "init_labels": config.init_labels,
        "budget": config.budget,
        "batch_size": config.batch_size,
        "scale": config.scale,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def _run_trial(config: ExperimentConfig, seed: int) -> dict:
    X = np.asarray(config.dataset.X)
    y = np.asarray(config.dataset.y)
    classes, class_counts = np.unique(y, return_counts=True)
    if config.init_labels < len(classes):
        raise ValueError(
            f"--init-labels must be at least the number of classes ({len(classes)})."
        )

    idx = np.arange(y.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=config.test_size,
        random_state=seed,
        stratify=y,
    )
    rng = np.random.default_rng(seed)
    labeled_idx = _initial_labeled_indices(y, train_idx, config.init_labels, rng)
    unlabeled_idx = np.setdiff1d(train_idx, labeled_idx, assume_unique=False)

    if config.scale:
        scaler = StandardScaler()
        X = X.copy()
        X[train_idx] = scaler.fit_transform(X[train_idx])
        X[test_idx] = scaler.transform(X[test_idx])

    budget = min(config.budget, unlabeled_idx.shape[0])
    model = build_model(config.model_name, seed)

    history = {
        "round": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "f1": [],
        "detail": [],
        "class_dist": {
            "all": dict(zip([int(c) for c in classes], [int(v) for v in class_counts])),
            "train": _class_dist(y[train_idx]),
            "test": _class_dist(y[test_idx]),
        },
    }

    _fit_and_score(model, X, y, labeled_idx, train_idx, test_idx, history, seed, None, 0.0, 0.0)

    used = 0
    while used < budget:
        batch_size = min(config.batch_size, budget - used, unlabeled_idx.shape[0])
        current_model = clone(model)
        start_train = time.time()
        current_model.fit(X[labeled_idx], y[labeled_idx])
        train_time = time.time() - start_train

        state = QueryState(
            X=X,
            y=y,
            labeled_idx=labeled_idx,
            unlabeled_idx=unlabeled_idx,
            model=current_model,
            rng=rng,
            seed=seed,
        )
        start_query = time.time()
        queried = np.asarray(config.strategy.selector(state, batch_size), dtype=int)
        query_time = time.time() - start_query

        labeled_idx = np.concatenate([labeled_idx, queried])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, queried, assume_unique=False)
        model = current_model
        _fit_and_score(
            model,
            X,
            y,
            labeled_idx,
            train_idx,
            test_idx,
            history,
            seed,
            queried.tolist(),
            train_time,
            query_time,
        )
        used += batch_size

    return history


def _initial_labeled_indices(
    y: np.ndarray,
    train_idx: np.ndarray,
    init_labels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for cls in np.unique(y):
        cls_idx = train_idx[y[train_idx] == cls]
        selected.append(rng.choice(cls_idx))
    remaining = np.setdiff1d(train_idx, np.asarray(selected), assume_unique=False)
    extra = init_labels - len(selected)
    if extra > 0:
        selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def _fit_and_score(
    model,
    X,
    y,
    labeled_idx,
    train_idx,
    test_idx,
    history,
    seed,
    queried,
    train_time,
    query_time,
) -> None:
    model.fit(X[labeled_idx], y[labeled_idx])
    train_pred = model.predict(X[train_idx])
    test_pred = model.predict(X[test_idx])
    train_acc = accuracy_score(y[train_idx], train_pred)
    test_acc = accuracy_score(y[test_idx], test_pred)
    f1 = f1_score(y[test_idx], test_pred, average="weighted")
    round_value = int(labeled_idx.shape[0])
    history["round"].append(round_value)
    history["train_accuracy"].append(train_acc)
    history["test_accuracy"].append(test_acc)
    history["f1"].append(f1)
    history["detail"].append(
        {
            "seed": seed,
            "round": round_value,
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "f1": f1,
            "train_time": train_time,
            "query_time": query_time,
            "query_idx": json.dumps(queried),
        }
    )


def _class_dist(y: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip([int(c) for c in classes], [int(v) for v in counts]))


def _aubc(rounds: list[int], values: list[float]) -> float:
    if len(values) == 1:
        return float(values[0])
    span = rounds[-1] - rounds[0]
    if span == 0:
        return float(values[-1])
    return float(np.trapz(values, x=rounds) / span)


def _default_run_name(config: ExperimentConfig) -> str:
    safe_dataset = config.dataset.name.replace("/", "_").replace(":", "_")
    return f"{safe_dataset}-{config.strategy.name}-{config.model_name}-seed{config.seed}"
