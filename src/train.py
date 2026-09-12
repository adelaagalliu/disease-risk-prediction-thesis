import json
import joblib
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from config import (
    DATASETS, GROUP_COL, MODELS_DIR, MODEL_NAMES,
    MODEL_SELECTION_FILE, RANDOM_STATE, SEARCH_N_ITER, SEARCH_CV_FOLDS,
    SEARCH_SCORING, load_split,
)
from preprocessing import get_feature_columns, build_preprocessor

def get_model_spec():
    """Return the single predictive model used in the dissertation."""
    estimator = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    param_dist = {
        "model__C": loguniform(0.01, 10),
        "model__solver": ["lbfgs"],
    }
    return estimator, param_dist

def train_one_dataset(name: str, data_cfg: dict):
    print(f"\nTraining models for: {name}:")
    target = data_cfg["target"]

    train_df = load_split(name, "train")
    feature_cols = get_feature_columns(train_df, target, data_cfg["drop_cols"])

    X_train, y_train = train_df[feature_cols], train_df[target]
    if GROUP_COL not in train_df:
        raise ValueError(
            f"[{name}] {GROUP_COL} is missing; rerun split_data.py before training"
        )
    groups = train_df[GROUP_COL]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"  features used ({len(feature_cols)}): {feature_cols}")
    print(f"  train class balance: {pos} positive / {neg} negative "
          f"(prevalence={pos / len(y_train):.3f})")

    cv = StratifiedGroupKFold(
        n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    candidates = []

    for model_name in MODEL_NAMES:
        estimator, param_dist = get_model_spec()

        preprocessor = build_preprocessor(feature_cols, scale_continuous=True)
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=SEARCH_N_ITER,
            scoring=SEARCH_SCORING,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train, groups=groups)
        best_pipeline = search.best_estimator_
        print(f" tuned {model_name:15s} best cv {SEARCH_SCORING}={search.best_score_:.3f}")
        print(f"    best params: {search.best_params_}")

        out_path = MODELS_DIR / f"{name}_{model_name}.joblib"
        joblib.dump(
            {
                "pipeline": best_pipeline,
                "feature_cols": feature_cols,
                "target": target,
                "cv_score": float(search.best_score_),
                "cv_scoring": SEARCH_SCORING,
            },
            out_path,
        )
        candidates.append({
            "model": model_name,
            "cv_score": float(search.best_score_),
        })
        print(f" saved {model_name:15s} -> {out_path}")

    if not candidates:
        raise RuntimeError(f"[{name}] no model could be trained")
    selected = candidates[0]
    selected.update({"criterion": SEARCH_SCORING, "source": "grouped_training_cv"})
    print(
        f" selected Logistic Regression model from grouped training CV: {selected['model']} "
        f"({SEARCH_SCORING}={selected['cv_score']:.3f})"
    )
    return selected


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selections = {}
    for name, data_cfg in DATASETS.items():
        selections[name] = train_one_dataset(name, data_cfg)
    with open(MODEL_SELECTION_FILE, "w") as f:
        json.dump(selections, f, indent=2)
    print(f"\nSaved Logistic Regression training metadata to {MODEL_SELECTION_FILE}")


if __name__ == "__main__":
    main()
