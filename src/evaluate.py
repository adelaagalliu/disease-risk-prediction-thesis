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
    DATASETS, GROUP_COL, MODELS_DIR, MODEL_NAMES,
    MODEL_SELECTION_FILE, RESULTS_DIR, THRESHOLD_METRIC, N_BOOTSTRAP,
    BOOTSTRAP_ALPHA, CALIBRATION_BINS, load_split,
)

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
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    scores = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_rows[group] for group in sampled_groups])
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip degenerate resamples with only one class
        scores.append(metric_fn(y_true[idx], y_proba[idx]))
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
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{dataset_name} - {model_name} ({split_name})")
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
        print(f"\nEvaluating models for: {name}")
        selection = selections[name]
        selected_name = selection["model"]
        all_results[name] = {"selection": selection, "models": {}}

        splits = {split_name: load_split(name, split_name) for split_name in ("val", "test")}

        evaluation_order = [selected_name] + [m for m in MODEL_NAMES if m != selected_name]
        for model_name in evaluation_order:
            model_path = MODELS_DIR / f"{name}_{model_name}.joblib"
            if not model_path.exists():
                print(f"  skipping {model_name}: no saved model at {model_path}")
                continue

            saved_model = joblib.load(model_path)
            pipeline, feature_cols = saved_model["pipeline"], saved_model["feature_cols"]

            model_results = evaluate_model(pipeline, feature_cols, target, splits,
                                            dataset_name=name, model_name=model_name)
            all_results[name]["models"][model_name] = model_results

            role = "PRIMARY (selected by grouped training CV)" if model_name == selected_name else "secondary benchmark"
            print(f"  {model_name}: {role}")

            for split_name in ("val", "test"):
                d = model_results[split_name]["default_threshold"]
                t = model_results[split_name]["tuned_threshold"]
                roc_ci = model_results[split_name]["roc_auc_ci"]
                pr_ci = model_results[split_name]["pr_auc_ci"]
                brier = model_results[split_name]["calibration"]["brier_score"]
                print(f"  [{split_name}] {model_name:15s} "
                      f"\n@0.50 -> prec={d['precision']:.3f} rec={d['recall']:.3f} f1={d['f1']:.3f}  |  "
                      f"\n@{t['threshold']:.2f} -> prec={t['precision']:.3f} rec={t['recall']:.3f} f1={t['f1']:.3f}  |  "
                      f"roc_auc={d['roc_auc']:.3f} pr_auc={d['pr_auc']:.3f}"
                      f"\n  95% CI -> roc_auc=[{roc_ci['lower']:.3f}, {roc_ci['upper']:.3f}]  "
                      f"pr_auc=[{pr_ci['lower']:.3f}, {pr_ci['upper']:.3f}]  brier={brier:.3f}\n")

    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_path}")

if __name__ == "__main__":
    main()
