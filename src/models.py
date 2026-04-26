from __future__ import annotations

def list_models() -> list[str]:
    return ["LR", "RBFSVM", "RandomForest", "XGBoost"]


def build_model(name: str, seed: int):
    if name == "LR":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "RBFSVM":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", gamma="scale", probability=True, random_state=seed)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "XGBoost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install xgboost or choose another model."
            ) from exc
        return xgb.XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=seed,
        )

    available = ", ".join(list_models())
    raise ValueError(f"Unknown model '{name}'. Available: {available}")
