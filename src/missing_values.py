import numpy as np
import pandas as pd

MIN_SUBGROUP_N = 30 

MISSING_VALUE_DATASETS = {"diabetes", "heart_disease", "hypertension", "stroke"}

SMOKER_IMPUTE_DATASETS = {"diabetes", "heart_disease", "hypertension"}
 
SYSTOLIC_BP_BINS = [0, 120, 140, 160, 180, 1000]

def _binary_mode(values):
    """Return a deterministic 0/1 mode; ties resolve to 1."""
    return int(values.mean() >= 0.5)
 
def drop_unimputable_rows(df, dataset_name):
    if dataset_name == "stroke":
        before = len(df)
        df = df[df["sex_binary"].notna()].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  dropped {dropped} row(s) with missing sex_binary "
                  f"(gender='Other' has no BRFSS-comparable encoding)")
    return df
 
def _fit_conditional(train_df, group_values, value_col, agg, min_n=MIN_SUBGROUP_N):
    """
    Fits `agg` (for example, a median or binary mode) of
    train_df[value_col] within each
    group of `group_values` (a Series aligned to train_df, e.g. an
    age_range column or a systolic_bp band), falling back to the overall
    train statistic for any group with fewer than `min_n` non-missing
    training rows. Returns (per_group_dict, overall_value).
    """
    observed = train_df[value_col].notna()
    overall = agg(train_df.loc[observed, value_col])
    stats = {}
    for g, idx in train_df.loc[observed].groupby(group_values[observed]).groups.items():
        vals = train_df.loc[idx, value_col]
        stats[g] = agg(vals) if len(vals) >= min_n else overall
    return stats, overall
 
def _apply_conditional(df, group_values, value_col, stats, overall):
    """Fills df[value_col]'s NaNs using a per-group statistic fit on train."""
    fill = group_values.map(stats).astype(float)
    df[value_col] = df[value_col].fillna(fill)
    df[value_col] = df[value_col].fillna(overall)  # safety net: group unseen in train
    return df
 
 
def fit_impute_stats(train_df, dataset_name):
    """Compute every imputation statistic from the training split only."""
    if dataset_name not in MISSING_VALUE_DATASETS:
        raise ValueError(f"No missing-value handling defined for dataset: {dataset_name}")

    stats = {}

    if dataset_name in SMOKER_IMPUTE_DATASETS:
        stats["smoker_rate"] = float(train_df["smoker"].mean())
 
    if dataset_name == "hypertension":
        stats["bmi_median"] = train_df["bmi"].median()
        stats["heart_rate_median"] = train_df["heart_rate"].median()
 
        chol_stats, chol_overall = _fit_conditional(
            train_df, train_df["age_range"], "cholesterol", np.median
        )
        stats["cholesterol_by_age_range"] = chol_stats
        stats["cholesterol_overall"] = chol_overall
 
        sbp_band = pd.cut(train_df["systolic_bp"], bins=SYSTOLIC_BP_BINS)
        bpmeds_stats, bpmeds_overall = _fit_conditional(
            train_df, sbp_band, "bp_meds", _binary_mode
        )
        stats["bp_meds_by_sbp_band"] = bpmeds_stats
        stats["bp_meds_overall"] = bpmeds_overall
 
        smoker_cigs = train_df.loc[train_df["smoker"] == 1, "cigs_per_day"]
        stats["cigs_per_day_smoker_median"] = smoker_cigs.median()
 
        overall_glucose_median = train_df["glucose"].median()
        stats["glucose_median"] = overall_glucose_median
        subgroup = {}
        for d in (0, 1):
            vals = train_df.loc[train_df["diabetes"] == d, "glucose"].dropna()
            subgroup[d] = vals.median() if len(vals) >= MIN_SUBGROUP_N else overall_glucose_median
        stats["glucose_subgroup_median"] = subgroup
 
    elif dataset_name == "stroke":
        bmi_stats, bmi_overall = _fit_conditional(
            train_df, train_df["age_range"], "bmi", np.median
        )
        stats["bmi_by_age_range"] = bmi_stats
        stats["bmi_overall"] = bmi_overall
 
    return stats
 
def apply_impute(df, dataset_name, stats):
    """Fill NaNs using statistics from fit_impute_stats() (train-only)."""
    if dataset_name not in MISSING_VALUE_DATASETS:
        raise ValueError(f"No missing-value handling defined for dataset: {dataset_name}")

    df = df.copy()

    if dataset_name in SMOKER_IMPUTE_DATASETS:
        df["smoker_was_missing"] = df["smoker"].isna().astype("int64")
        df["smoker"] = df["smoker"].fillna(stats["smoker_rate"])
 
    if dataset_name == "hypertension":
        df["bmi"] = df["bmi"].fillna(stats["bmi_median"])
        df["heart_rate"] = df["heart_rate"].fillna(stats["heart_rate_median"])
 
        df = _apply_conditional(
            df, df["age_range"], "cholesterol",
            stats["cholesterol_by_age_range"], stats["cholesterol_overall"],
        )
 
        df["bp_meds_was_missing"] = df["bp_meds"].isna().astype("int64")
        sbp_band = pd.cut(df["systolic_bp"], bins=SYSTOLIC_BP_BINS)
        df = _apply_conditional(
            df, sbp_band, "bp_meds",
            stats["bp_meds_by_sbp_band"], stats["bp_meds_overall"],
        )
        df["bp_meds"] = df["bp_meds"].astype("int64")
 
        smoker_mask = df["smoker"] == 1
        df.loc[smoker_mask, "cigs_per_day"] = df.loc[smoker_mask, "cigs_per_day"].fillna(
            stats["cigs_per_day_smoker_median"]
        )
        df["cigs_per_day"] = df["cigs_per_day"].fillna(0.0) 
 
        subgroup = stats["glucose_subgroup_median"]
        fallback = df["diabetes"].map(subgroup).astype(float)
        df["glucose"] = df["glucose"].fillna(fallback)
        df["glucose"] = df["glucose"].fillna(stats["glucose_median"])
 
        df["high_chol"] = (df["cholesterol"] > 200).astype("int64")
 
    elif dataset_name == "stroke":
        df["bmi_was_missing"] = df["bmi"].isna().astype("int64")
        df = _apply_conditional(
            df, df["age_range"], "bmi",
            stats["bmi_by_age_range"], stats["bmi_overall"],
        )
 
    return df
 
