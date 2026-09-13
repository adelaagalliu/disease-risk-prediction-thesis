import json
import joblib
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from config import (
    DATASETS, GROUP_COL, MODELS_DIR, MODEL_SELECTION_FILE, RANDOM_STATE,
    SEARCH_N_ITER, SEARCH_CV_FOLDS, SEARCH_SCORING, load_split,
)
from preprocessing import get_feature_columns, build_preprocessor

MODEL_NAME = "logreg"

def train_one_dataset(name: str, data_cfg: dict):
    print(f"\nPo trajnohet modeli per: {name}")
    target = data_cfg["target"]
    train_df = load_split(name, "train")
    feature_cols = get_feature_columns(train_df, target, data_cfg["drop_cols"])
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    if GROUP_COL not in train_df:
        raise ValueError(
            f"[{name}] mungon {GROUP_COL}; ekzekuto perseri split_data.py para trajnimit"
        )

    groups = train_df[GROUP_COL]
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    print(f"  kolonat e perdorura ({len(feature_cols)}): {feature_cols}")
    print(
        f"  ekuilibri i klasave ne trajnim: {pos} pozitive / {neg} negative "
        f"(prevalenca={pos / len(y_train):.3f})"
    )

    estimator = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    param_dist = {
        "model__C": loguniform(0.01, 10),
        "model__solver": ["lbfgs"],
    }
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(feature_cols)),
        ("model", estimator),
    ])
    cv = StratifiedGroupKFold(
        n_splits=SEARCH_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
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

    out_path = MODELS_DIR / f"{name}_{MODEL_NAME}.joblib"
    joblib.dump(
        {
            "pipeline": search.best_estimator_,
            "feature_cols": feature_cols,
            "target": target,
            "cv_score": float(search.best_score_),
            "cv_scoring": SEARCH_SCORING,
        },
        out_path,
    )

    print(
        f"  rezultati me i mire i CV te grupuar: "
        f"{SEARCH_SCORING}={search.best_score_:.3f}"
    )
    print(f"  parametrat me te mire: {search.best_params_}")
    print(f"  modeli u ruajt në: {out_path}")

    return {
        "model": MODEL_NAME,
        "cv_score": float(search.best_score_),
        "criterion": SEARCH_SCORING,
        "source": "grouped_training_cv",
    }

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selections = {}
    for name, data_cfg in DATASETS.items():
        selections[name] = train_one_dataset(name, data_cfg)
    with open(MODEL_SELECTION_FILE, "w", encoding="utf-8") as file:
        json.dump(selections, file, indent=2, ensure_ascii=False)
    print(
        "\nMetadatat e trajnimit te regresionit logjistik "
        f"u ruajten ne: {MODEL_SELECTION_FILE}"
    )

if __name__ == "__main__":
    main()