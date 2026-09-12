import json
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from config import (
    DATASETS, GROUP_COL, MODELS_DIR,
    MODEL_SELECTION_FILE, RESULTS_DIR, THRESHOLD_METRIC, N_BOOTSTRAP,
    BOOTSTRAP_ALPHA, CALIBRATION_BINS, load_split,
)

DATASET_LABELS = {
    "diabetes": "Diabeti",
    "heart_disease": "Sëmundja e zemrës",
    "hypertension": "Hipertensioni",
    "stroke": "Insulti cerebral",
}
MODEL_LABELS = {
    "logreg": "Regresioni Logjistik",
    "prevalence_baseline": "Modeli bazë i prevalencës",
}
SPLIT_LABELS = {"val": "validim", "test": "test"}

def _score_at_threshold(y_true, y_proba, threshold, metric):
    y_pred = (y_proba >= threshold).astype(int)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    raise ValueError(f"Unsupported threshold metric: {metric}")

def find_best_threshold(y_true, y_proba, metric=THRESHOLD_METRIC, n_steps=99):
    thresholds = np.linspace(0.01, 0.99, n_steps)
    scores = [_score_at_threshold(y_true, y_proba, t, metric) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx])

def bootstrap_ci(y_true, y_proba, metric_fn, groups=None, n_boot=N_BOOTSTRAP,
                 alpha=BOOTSTRAP_ALPHA, seed=42) -> dict:
    """
    Resamples whole duplicate-profile groups with replacement, recomputes the
    metric, and returns the mean plus a (1-alpha) percentile CI. If groups is
    omitted, each row is treated as its own group.
    Use this instead of comparing single point estimates across models --
    with these dataset sizes, ROC-AUC/PR-AUC differences of ~0.01 between
    models are usually well within noise, and the CI is what shows that.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    if len(groups) != len(y_true):
        raise ValueError("groups must have the same length as y_true")
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    group_probabilities = np.full(len(unique_groups), 1 / len(unique_groups))

    # Scores never change across bootstrap samples, so sort and identify tied
    # score groups once. This is equivalent to calling sklearn with frequency
    # weights, but avoids repeating an expensive sort 1000 times.
    descending = metric_fn is average_precision_score
    order = np.argsort(y_proba, kind="mergesort")
    if descending:
        order = order[::-1]
    sorted_scores = y_proba[order]
    sorted_target = y_true[order]
    tie_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]

    def weighted_score(sample_weight):
        weight = sample_weight[order]
        positive = np.add.reduceat(weight * sorted_target, tie_starts)
        negative = np.add.reduceat(weight * (1 - sorted_target), tie_starts)
        total_positive = positive.sum()
        total_negative = negative.sum()
        if metric_fn is roc_auc_score:
            negative_before = np.cumsum(negative) - negative
            return np.sum(positive * (negative_before + 0.5 * negative)) / (
                total_positive * total_negative
            )
        if metric_fn is average_precision_score:
            true_positive = np.cumsum(positive)
            false_positive = np.cumsum(negative)
            denominator = true_positive + false_positive
            precision = np.divide(
                true_positive,
                denominator,
                out=np.zeros_like(true_positive, dtype=float),
                where=denominator > 0,
            )
            recall = true_positive / total_positive
            return np.sum(np.diff(np.r_[0, recall]) * precision)
        return metric_fn(y_true, y_proba, sample_weight=sample_weight)

    scores = []
    for _ in range(n_boot):
        group_counts = rng.multinomial(len(unique_groups), group_probabilities)
        sample_weight = group_counts[group_index]
        if not sample_weight[y_true == 0].sum() or not sample_weight[y_true == 1].sum():
            continue  # skip degenerate resamples with only one class
        scores.append(weighted_score(sample_weight))
    scores = np.array(scores)
    if not len(scores):
        raise ValueError("No valid bootstrap samples contained both target classes")
    return {
        "mean": float(scores.mean()),
        "lower": float(np.percentile(scores, 100 * alpha / 2)),
        "upper": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "resampling_unit": "group",
        "n_groups": int(len(unique_groups)),
    }

def compute_calibration(y_true, y_proba, n_bins=CALIBRATION_BINS) -> dict:
    """
    Brier score (lower is better; combines calibration + discrimination)
    plus reliability-curve points. Threshold-independent -- computed once
    per split, not per threshold.
    """
    brier = brier_score_loss(y_true, y_proba)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": float(brier),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

def save_calibration_plot(y_true, y_proba, dataset_name, model_name, split_name, n_bins=CALIBRATION_BINS):
    if not HAS_MPL:
        return None
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    out_dir = RESULTS_DIR / "calibration_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_{model_name}_{split_name}.png"

    fig, ax = plt.subplots(figsize=(5, 5))
    model_label = MODEL_LABELS.get(model_name, model_name)
    dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
    split_label = SPLIT_LABELS.get(split_name, split_name)
    ax.plot(prob_pred, prob_true, marker="o", label=model_label)
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray",
        label="Kalibrim i përsosur",
    )
    ax.set_xlabel("Probabiliteti mesatar i parashikuar")
    ax.set_ylabel("Përqindja e rasteve pozitive")
    ax.set_title(f"{dataset_label} – {model_label} ({split_label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)

def compute_metrics(y_true, y_proba, threshold, split_name) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "split": split_name,
        "threshold": threshold,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def evaluate_model(pipeline, feature_cols, target, splits: dict,
                    dataset_name: str = None, model_name: str = None) -> dict:
    """
    splits: {"val": df, "test": df}.
    Reports metrics at the default 0.50 threshold AND at a threshold tuned
    on val, then reused unchanged for test (THRESHOLD_METRIC in config.py).
    Also reports, per split:
    - bootstrap 95% CIs for ROC-AUC and PR-AUC (threshold-independent)
    - calibration (Brier score + reliability curve points)
    A calibration reliability plot is saved to disk for the test split only.
    """
    val_df = splits["val"]
    val_proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    best_threshold = find_best_threshold(val_df[target], val_proba)

    results = {}
    for split_name, df in splits.items():
        X, y_true = df[feature_cols], df[target]
        y_proba = pipeline.predict_proba(X)[:, 1]
        groups = df[GROUP_COL] if GROUP_COL in df else None

        results[split_name] = {
            "default_threshold": compute_metrics(y_true, y_proba, 0.5, split_name),
            "tuned_threshold": compute_metrics(y_true, y_proba, best_threshold, split_name),
            "roc_auc_ci": bootstrap_ci(y_true, y_proba, roc_auc_score, groups=groups),
            "pr_auc_ci": bootstrap_ci(y_true, y_proba, average_precision_score, groups=groups),
            "calibration": compute_calibration(y_true, y_proba),
        }
        if split_name == "test" and dataset_name and model_name:
            save_calibration_plot(y_true, y_proba, dataset_name, model_name, split_name)

    return results

def evaluate_prevalence_baseline(train_df, target, splits: dict,
                                 dataset_name: str = None) -> dict:
    """Evaluate a constant probability equal to training-set prevalence."""
    prevalence = float(train_df[target].mean())
    results = {}
    for split_name, df in splits.items():
        y_true = df[target]
        y_proba = np.full(len(df), prevalence)
        groups = df[GROUP_COL] if GROUP_COL in df else None
        results[split_name] = {
            "prediction_rule": "probabilitet konstant, i barabartë me prevalencën në grupin e trajnimit",
            "training_prevalence": prevalence,
            "default_threshold": compute_metrics(y_true, y_proba, 0.5, split_name),
            "roc_auc_ci": bootstrap_ci(y_true, y_proba, roc_auc_score, groups=groups),
            "pr_auc_ci": bootstrap_ci(
                y_true, y_proba, average_precision_score, groups=groups
            ),
            "calibration": compute_calibration(y_true, y_proba),
        }
        if split_name == "test" and dataset_name:
            save_calibration_plot(
                y_true, y_proba, dataset_name, "prevalence_baseline", split_name
            )
    return results

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"No CV-only model-selection file at {MODEL_SELECTION_FILE}; run train.py first"
        )
    with open(MODEL_SELECTION_FILE) as f:
        selections = json.load(f)
    all_results = {}

    for name, data_cfg in DATASETS.items():
        target = data_cfg["target"]
        print(f"\nEvaluating Logistic Regression and prevalence baseline for: {name}")
        selection = selections[name]
        selected_name = "logreg"
        if selection["model"] != selected_name:
            raise ValueError(
                f"[{name}] model_selection.json is stale; rerun train.py"
            )
        all_results[name] = {"selection": selection, "models": {}}

        splits = {split_name: load_split(name, split_name) for split_name in ("val", "test")}
        train_df = load_split(name, "train")
        model_path = MODELS_DIR / f"{name}_{selected_name}.joblib"
        saved_model = joblib.load(model_path)
        pipeline, feature_cols = saved_model["pipeline"], saved_model["feature_cols"]

        model_results = evaluate_model(
            pipeline, feature_cols, target, splits,
            dataset_name=name, model_name=selected_name,
        )
        baseline_results = evaluate_prevalence_baseline(
            train_df, target, splits, dataset_name=name
        )
        all_results[name]["models"][selected_name] = model_results
        all_results[name]["models"]["prevalence_baseline"] = baseline_results

        for split_name in ("val", "test"):
            d = model_results[split_name]["default_threshold"]
            t = model_results[split_name]["tuned_threshold"]
            b = baseline_results[split_name]["default_threshold"]
            print(
                f"  [{split_name}] logreg: ROC-AUC={d['roc_auc']:.3f}, "
                f"PR-AUC={d['pr_auc']:.3f}, F1={t['f1']:.3f}, "
                f"sensitivity={t['sensitivity']:.3f}, specificity={t['specificity']:.3f}, "
                f"Brier={model_results[split_name]['calibration']['brier_score']:.3f}"
            )
            print(
                f"  [{split_name}] prevalence baseline: ROC-AUC={b['roc_auc']:.3f}, "
                f"PR-AUC={b['pr_auc']:.3f}, F1={b['f1']:.3f}, "
                f"sensitivity={b['sensitivity']:.3f}, specificity={b['specificity']:.3f}, "
                f"Brier={baseline_results[split_name]['calibration']['brier_score']:.3f}"
            )

    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_path}")

if __name__ == "__main__":
    main()
