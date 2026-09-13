import numpy as np
import pandas as pd
import joblib
import shap

from config import (
    DATASETS, MODELS_DIR, MODEL_NAMES, RESULTS_DIR, RANDOM_STATE,
    SHAP_BACKGROUND_SIZE, SHAP_EXPLAIN_SIZE, TOP_N_LOCAL_EXAMPLES, load_split
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def get_transformed_feature_names(preprocessor, feature_cols):
    try:
        names = preprocessor.get_feature_names_out()
        return [n.split("__", 1)[-1] for n in names]
    except Exception:
        return feature_cols

def build_explainer(model_name, model, background):
    if model_name != "logreg":
        raise ValueError(f"No SHAP explainer configured for model type: {model_name}")
    return shap.LinearExplainer(model, background)

def compute_shap_values(model_name, model, background, X_explain):
    explainer = build_explainer(model_name, model, background)
    return explainer(X_explain), "log-odds (linear score)"

def extract_values_and_base(shap_values):
    values = shap_values.values
    base_values = shap_values.base_values
    if values.ndim == 3:
        values = values[:, :, 1]
        base_values = base_values[:, 1] if np.ndim(base_values) > 1 else base_values
    return values, base_values

def explain_one_model(dataset_name: str, model_name: str, out_root):
    model_path = MODELS_DIR / f"{dataset_name}_{model_name}.joblib"
    if not model_path.exists():
        print(f"  skipping {model_name}: no saved model at {model_path}")
        return

    saved_model = joblib.load(model_path)
    pipeline, feature_cols = saved_model["pipeline"], saved_model["feature_cols"]
    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    train_df = load_split(dataset_name, "train")
    test_df = load_split(dataset_name, "test")

    rng = np.random.RandomState(RANDOM_STATE)
    bg_n = min(SHAP_BACKGROUND_SIZE, len(train_df))
    bg_idx = rng.choice(len(train_df), size=bg_n, replace=False)
    background = preprocessor.transform(train_df[feature_cols].iloc[bg_idx])

    ex_n = min(SHAP_EXPLAIN_SIZE, len(test_df))
    ex_idx = rng.choice(len(test_df), size=ex_n, replace=False)
    X_explain_raw = test_df[feature_cols].iloc[ex_idx]
    X_explain = preprocessor.transform(X_explain_raw)
    feature_names = get_transformed_feature_names(preprocessor, feature_cols)

    shap_values, units = compute_shap_values(model_name, model, background, X_explain)
    values, base_values = extract_values_and_base(shap_values)

    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if HAS_MPL:
        shap.summary_plot(values, X_explain, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "summary_beeswarm.png", dpi=150)
        plt.close()

        shap.summary_plot(values, X_explain, feature_names=feature_names,
                           plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "summary_bar.png", dpi=150)
        plt.close()

        X_explain_arr = np.asarray(X_explain)
        for i in range(min(TOP_N_LOCAL_EXAMPLES, len(X_explain_arr))):
            expl = shap.Explanation(
                values=values[i],
                base_values=base_values[i] if np.ndim(base_values) > 0 else base_values,
                data=X_explain_arr[i],
                feature_names=feature_names,
            )
            shap.plots.waterfall(expl, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / f"waterfall_example_{i}.png", dpi=150)
            plt.close()
    else:
        print("    matplotlib not available: skipping plots, saving importance table only")

    mean_abs = np.abs(values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    with open(out_dir / "NOTES.txt", "w", encoding="utf-8") as f:
        f.write(f"Vlerat SHAP për {dataset_name}/{model_name} jane ne njesine: {units}\n"
    )
    if units != "probability":
        f.write("Keto nuk jane njesi probabiliteti. Mos i krahasoni drejtperdrejt me vlerat SHAP te modeleve te shprehura ne probabilitet."
        )
    print(f"  {model_name:15s} -> rezultatet SHAP u ruajten ne {out_dir}  (units: {units})")

def main():
    for dataset_name in DATASETS:
        print(f"\nExplaining models for: {dataset_name}")
        out_root = RESULTS_DIR / "shap" / dataset_name
        for model_name in MODEL_NAMES:
            explain_one_model(dataset_name, model_name, out_root)

if __name__ == "__main__":
    main()
