import json
import joblib
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from config import (
    DATASETS, GROUP_COL, MODELS_DIR, MODEL_NAMES,
    MODEL_SELECTION_FILE, RANDOM_STATE, SEARCH_N_ITER, SEARCH_CV_FOLDS,
    SEARCH_SCORING, load_split,
)
from preprocessing import get_feature_columns, build_preprocessor

def get_model_spec(model_name: str, pos_weight_ratio: float):
    """
    Rn_jobs is set to 1 on the individual estimators (rather than -1) because
    RandomizedSearchCV itself parallelizes across CV folds x param
    candidates with n_jobs=-1; parallelizing at both levels oversubscribes
    CPU cores and is usually slower, not faster.
    """
    if model_name == "logreg":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        param_dist = {
            "model__C": uniform(0.01, 10),
            "model__solver": ["lbfgs"],
        }
        return estimator, True, param_dist

    if model_name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        param_dist = {
            "model__n_estimators": randint(200, 600), # number of trees
            "model__max_depth": randint(4, 20),
            "model__min_samples_leaf": randint(2, 20),
            "model__max_features": ["sqrt", "log2", 0.5],
        }
        return estimator, False, param_dist

    if model_name == "xgboost":
        estimator = XGBClassifier(
            scale_pos_weight=pos_weight_ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        param_dist = {
            "model__n_estimators": randint(150, 600),
            "model__max_depth": randint(3, 9),
            "model__learning_rate": uniform(0.01, 0.29),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__gamma": uniform(0, 5),
        }
        return estimator, False, param_dist

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            return None, None, None
        estimator = LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        )
        param_dist = {
            "model__n_estimators": randint(150, 600),
            "model__num_leaves": randint(15, 127),
            "model__max_depth": randint(3, 12),
            "model__learning_rate": uniform(0.01, 0.29),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_samples": randint(5, 60),
        }
        return estimator, False, param_dist

    if model_name == "catboost":
        if CatBoostClassifier is None:
            return None, None, None
        estimator = CatBoostClassifier(
            auto_class_weights="Balanced",
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        param_dist = {
            "model__iterations": randint(200, 700),
            "model__depth": randint(4, 10),
            "model__learning_rate": uniform(0.01, 0.29),
            "model__l2_leaf_reg": uniform(1, 9),
        }
        return estimator, False, param_dist

    raise ValueError(f"Unknown model: {model_name}")

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
    pos_weight_ratio = neg / pos
    print(f"  features used ({len(feature_cols)}): {feature_cols}")
    print(f"  train class balance: {pos} positive / {neg} negative "
          f"(scale_pos_weight={pos_weight_ratio:.2f})")

    cv = StratifiedGroupKFold(
        n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    candidates = []

    for model_name in MODEL_NAMES:
        estimator, scale_continuous, param_dist = get_model_spec(model_name, pos_weight_ratio)
        if estimator is None:
            print(f"  skipping {model_name}: package not installed "
                  f"(pip install {model_name})")
            continue

        preprocessor = build_preprocessor(feature_cols, scale_continuous)
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
    selected = max(candidates, key=lambda item: item["cv_score"])
    selected.update({"criterion": SEARCH_SCORING, "source": "grouped_training_cv"})
    print(
        f" selected primary model from grouped training CV: {selected['model']} "
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
    print(f"\nSaved CV-only model selections to {MODEL_SELECTION_FILE}")


if __name__ == "__main__":
    main()
