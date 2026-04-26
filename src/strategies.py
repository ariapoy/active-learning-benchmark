from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class StrategySpec:
    name: str
    display_name: str
    category: str
    framework: str
    legacy_name: str | None
    limitation: str
    selector: Callable | None


BASELINE = "Baseline"
MODEL_UNCERTAINTY = "Model uncertainty"
EXPECTED_CHANGE = "Expected model changing / improvement"
REPRESENTATIVE = "Representative Exploiting"
HYBRID = "Hybrid criteria"
BAYESIAN = "Bayesian methods"
REDESIGNED = "Redesigned Learning Framework"
ORACLE = "Oracle / answer-peeking diagnostic"


def select_random(state, batch_size: int) -> np.ndarray:
    return state.rng.choice(state.unlabeled_idx, size=batch_size, replace=False)


def select_least_confidence(state, batch_size: int) -> np.ndarray:
    proba = _predict_proba(state)
    score = 1.0 - np.max(proba, axis=1)
    return _top_by_score(state.unlabeled_idx, score, batch_size)


def select_margin(state, batch_size: int) -> np.ndarray:
    proba = _predict_proba(state)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    return _top_by_score(state.unlabeled_idx, -margin, batch_size)


def select_entropy(state, batch_size: int) -> np.ndarray:
    proba = _predict_proba(state)
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)
    return _top_by_score(state.unlabeled_idx, entropy, batch_size)


def select_qbc(state, batch_size: int) -> np.ndarray:
    from sklearn.base import clone
    from sklearn.ensemble import BaggingClassifier

    estimator = clone(state.model)
    committee = BaggingClassifier(
        estimator=estimator,
        n_estimators=10,
        bootstrap=True,
        random_state=state.seed,
        n_jobs=-1,
    )
    committee.fit(state.X[state.labeled_idx], state.y[state.labeled_idx])
    votes = np.asarray(
        [member.predict(state.X[state.unlabeled_idx]) for member in committee.estimators_]
    ).T
    classes = np.unique(state.y)
    vote_proba = np.zeros((votes.shape[0], classes.shape[0]), dtype=float)
    for row_idx, row in enumerate(votes):
        for cls_idx, cls in enumerate(classes):
            vote_proba[row_idx, cls_idx] = np.mean(row == cls)
    entropy = -np.sum(vote_proba * np.log(np.clip(vote_proba, 1e-12, 1.0)), axis=1)
    return _top_by_score(state.unlabeled_idx, entropy, batch_size)


def select_coreset(state, batch_size: int) -> np.ndarray:
    return _farthest_first(state.X, state.labeled_idx, state.unlabeled_idx, batch_size)


def select_expected_average_precision(state, batch_size: int) -> np.ndarray:
    return select_entropy(state, batch_size)


def select_expected_error_reduction(state, batch_size: int) -> np.ndarray:
    return select_entropy(state, batch_size)


def select_mc_eer(state, batch_size: int) -> np.ndarray:
    return select_expected_error_reduction(state, batch_size)


def select_epistemic_uncertainty(state, batch_size: int) -> np.ndarray:
    return select_qbc(state, batch_size)


def unsupported_strategy(name: str):
    def _selector(state, batch_size: int):
        raise NotImplementedError(
            f"Strategy '{name}' is listed for taxonomy/legacy discovery, but the "
            "new sklearn-native CLI does not implement it yet. Use legacy_main.py "
            "with the matching legacy strategy, or choose a supported CLI strategy."
        )

    return _selector


STRATEGIES = {
    "random": StrategySpec("random", "Random sampling", BASELINE, "sklearn-native", "skal_uniform", "", select_random),
    "least_confidence": StrategySpec("least_confidence", "Least Confidence", MODEL_UNCERTAINTY, "sklearn-native", "skal_us_lc", "", select_least_confidence),
    "margin": StrategySpec("margin", "Margin Sampling", MODEL_UNCERTAINTY, "sklearn-native", "skal_us_margin", "", select_margin),
    "entropy": StrategySpec("entropy", "Entropy Sampling", MODEL_UNCERTAINTY, "sklearn-native", "skal_us_ent", "", select_entropy),
    "qbc": StrategySpec("qbc", "Query-by-Committee", MODEL_UNCERTAINTY, "sklearn-native", "qbc", "Uses a bagging committee in the new CLI.", select_qbc),
    "expected_average_precision": StrategySpec("expected_average_precision", "Expected Average Precision", EXPECTED_CHANGE, "sklearn-native", "skal_us_eap", "Approximated with entropy in the new CLI.", select_expected_average_precision),
    "eer": StrategySpec("eer", "Expected Error Reduction", EXPECTED_CHANGE, "sklearn-native", "eer", "Approximated with entropy in the new CLI.", select_expected_error_reduction),
    "mc_eer": StrategySpec("mc_eer", "Monte Carlo Expected Error Reduction", EXPECTED_CHANGE, "sklearn-native", "skal_eer", "Approximated with entropy in the new CLI.", select_mc_eer),
    "variance_reduction": StrategySpec("variance_reduction", "Variance Reduction", EXPECTED_CHANGE, "libact", "vr", "Requires a libact build with VarianceReduction support.", unsupported_strategy("variance_reduction")),
    "coreset": StrategySpec("coreset", "CoreSet", REPRESENTATIVE, "sklearn-native", "skal_coreset", "", select_coreset),
    "kcenter": StrategySpec("kcenter", "K-Center Greedy", REPRESENTATIVE, "sklearn-native", "kcenter", "", select_coreset),
    "graph_density": StrategySpec("graph_density", "Graph Density", REPRESENTATIVE, "Google AL", "graph", "Use legacy_main.py for the original graph-density implementation.", unsupported_strategy("graph_density")),
    "hierarchical": StrategySpec("hierarchical", "Hierarchical Sampling", REPRESENTATIVE, "Google AL", "hier", "Use legacy_main.py for the original hierarchical implementation.", unsupported_strategy("hierarchical")),
    "quire": StrategySpec("quire", "QUIRE", HYBRID, "libact", "quire", "Use legacy_main.py for the original QUIRE implementation.", unsupported_strategy("quire")),
    "dwus": StrategySpec("dwus", "DWUS", HYBRID, "libact", "dwus", "Use legacy_main.py for the original DWUS implementation.", unsupported_strategy("dwus")),
    "infodiv": StrategySpec("infodiv", "Information-Diversity", HYBRID, "Google AL", "infodiv", "Use legacy_main.py for the original implementation.", unsupported_strategy("infodiv")),
    "mcm": StrategySpec("mcm", "Margin Cluster Mean", HYBRID, "Google AL", "mcm", "Use legacy_main.py for the original implementation.", unsupported_strategy("mcm")),
    "bmdr": StrategySpec("bmdr", "BMDR", HYBRID, "ALiPy", "bmdr", "Requires ALiPy and a compatible QP solver in legacy_main.py.", unsupported_strategy("bmdr")),
    "spal": StrategySpec("spal", "SPAL", HYBRID, "ALiPy", "spal", "Requires ALiPy and a compatible QP solver in legacy_main.py.", unsupported_strategy("spal")),
    "bald": StrategySpec("bald", "BALD", BAYESIAN, "scikit-activeml", "skal_bald", "Use legacy_main.py for scikit-activeml BALD.", unsupported_strategy("bald")),
    "greedy_bald": StrategySpec("greedy_bald", "GreedyBALD", BAYESIAN, "scikit-activeml", "skal_greedbald", "Use legacy_main.py for scikit-activeml GreedyBALD.", unsupported_strategy("greedy_bald")),
    "batch_bald": StrategySpec("batch_bald", "BatchBALD", BAYESIAN, "scikit-activeml", None, "Use a scikit-activeml integration for full BatchBALD.", unsupported_strategy("batch_bald")),
    "epistemic_uncertainty": StrategySpec("epistemic_uncertainty", "Epistemic Uncertainty Sampling", BAYESIAN, "sklearn-native", None, "Approximated with committee disagreement in the new CLI.", select_epistemic_uncertainty),
    "albl": StrategySpec("albl", "ALBL", REDESIGNED, "libact", "albl", "Bandit-learning framework; use legacy_main.py for original ALBL.", unsupported_strategy("albl")),
    "lal": StrategySpec("lal", "LAL", REDESIGNED, "ALiPy", "lal", "Requires ALiPy LAL assets in legacy_main.py.", unsupported_strategy("lal")),
    "bso": StrategySpec("bso", "BSO / beam-search optimization", ORACLE, "local", "bso", "Oracle diagnostic that peeks at answers; not fair deployable AL.", unsupported_strategy("bso")),
}


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}") from exc


def list_strategies() -> list[StrategySpec]:
    return sorted(STRATEGIES.values(), key=lambda item: (item.category, item.name))


def _predict_proba(state) -> np.ndarray:
    X_unlabeled = state.X[state.unlabeled_idx]
    if hasattr(state.model, "predict_proba"):
        return state.model.predict_proba(X_unlabeled)

    decision = state.model.decision_function(X_unlabeled)
    if decision.ndim == 1:
        proba_pos = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack([1.0 - proba_pos, proba_pos])
    exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _top_by_score(indices: np.ndarray, score: np.ndarray, batch_size: int) -> np.ndarray:
    order = np.argsort(score)[::-1]
    return indices[order[:batch_size]]


def _farthest_first(
    X: np.ndarray,
    labeled_idx: np.ndarray,
    unlabeled_idx: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    from sklearn.metrics import pairwise_distances

    selected = []
    candidate_idx = unlabeled_idx.copy()
    anchor_idx = labeled_idx.copy()
    for _ in range(batch_size):
        distances = pairwise_distances(X[candidate_idx], X[anchor_idx])
        nearest = distances.min(axis=1)
        pick_pos = int(np.argmax(nearest))
        pick = candidate_idx[pick_pos]
        selected.append(pick)
        anchor_idx = np.append(anchor_idx, pick)
        candidate_idx = np.delete(candidate_idx, pick_pos)
    return np.asarray(selected, dtype=int)
