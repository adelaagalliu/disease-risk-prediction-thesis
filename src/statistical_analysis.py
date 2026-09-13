import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    mannwhitneyu,
    norm,
    ttest_ind,
)
from sklearn.linear_model import LogisticRegression

from config import (
    CONTINUOUS_COLS, DATASETS, GROUP_COL, ORDINAL_COLS, RESULTS_DIR, load_split,
)
from preprocessing import get_feature_columns

INFERENCE_EXCLUDE = {
    "age_bmi",
    "age_genhealth",
    "age_glucose",
    "bmi_obese",
    "bp_chol_both",
    "general_health_diffwalk",
    "glucose_bmi",
    "heavy_smoker",
    "hyperglycemia",
    "lifestyle_score",
    "mean_arterial_pressure",
    "metabolic_risk_score",
    "poor_health_days",
    "pulse_pressure",
    "tachycardia",
    "uncontrolled_bp",
    "work_type_private",
}

# formaton mesataren, devijimin standard, medianen dhe IQR
def _fmt_continuous(values: pd.Series) -> tuple[str, str]:
    values = values.dropna().astype(float)
    if values.empty:
        return "NA", "NA"
    mean_sd = f"{values.mean():.2f} ({values.std(ddof=1):.2f})"
    median_iqr = (
        f"{values.median():.2f} "
        f"[{values.quantile(0.25):.2f}, {values.quantile(0.75):.2f}]"
    )
    return mean_sd, median_iqr

# krijon tabelen pershkruese dhe testet statistikore
def descriptive_table(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    continuous = set(CONTINUOUS_COLS)
    for feature in features:
        if feature in continuous:
            group_0 = df.loc[df[target] == 0, feature].dropna()
            group_1 = df.loc[df[target] == 1, feature].dropna()
            mean_0, median_0 = _fmt_continuous(group_0)
            mean_1, median_1 = _fmt_continuous(group_1)
            rows.append({
                "feature": feature,
                "level": "continuous",
                "target_0_mean_sd": mean_0,
                "target_1_mean_sd": mean_1,
                "target_0_median_iqr": median_0,
                "target_1_median_iqr": median_1,
                "test": "Welch t-test; Mann-Whitney U",
                "p_value": float(ttest_ind(group_0, group_1, equal_var=False).pvalue),
                "mann_whitney_p_value": float(
                    mannwhitneyu(group_0, group_1, alternative="two-sided").pvalue
                ),
            })
            continue

        table = pd.crosstab(df[feature], df[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            test_name, p_value = "not applicable", np.nan
        else:
            _, chi_p, _, expected = chi2_contingency(table)
            if table.shape == (2, 2) and (expected < 5).any():
                _, p_value = fisher_exact(table.to_numpy())
                test_name = "Fisher exact"
            else:
                p_value, test_name = chi_p, "chi-square"

        for level, counts in table.iterrows():
            n_0 = int(counts.get(0, 0))
            n_1 = int(counts.get(1, 0))
            denom_0 = int((df[target] == 0).sum())
            denom_1 = int((df[target] == 1).sum())
            rows.append({
                "feature": feature,
                "level": level,
                "target_0_n_percent": f"{n_0} ({100 * n_0 / denom_0:.1f}%)",
                "target_1_n_percent": f"{n_1} ({100 * n_1 / denom_1:.1f}%)",
                "test": test_name,
                "p_value": float(p_value),
                "mann_whitney_p_value": np.nan,
            })
    return pd.DataFrame(rows)

# pergatit matricen dhe standardizon vlerat numerike
def _prepare_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    matrix = df[features].astype(float).to_numpy(copy=True)
    names = []
    scale_features = set(CONTINUOUS_COLS) | set(ORDINAL_COLS)
    for index, feature in enumerate(features):
        if feature in scale_features:
            std = matrix[:, index].std(ddof=0)
            if std > 0:
                matrix[:, index] = (matrix[:, index] - matrix[:, index].mean()) / std
            names.append(f"{feature} (per 1 SD)")
        else:
            names.append(f"{feature} (per 1 unit)")
    return matrix, names

# llogarit koeficientet, odds ratio, intervalet e besimit (ci) dhe p-vlerat
def _wald_table(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    coefficients = np.concatenate([model.intercept_, model.coef_[0]])
    design = np.column_stack([np.ones(len(X)), X])
    fitted = expit(design @ coefficients)
    weights = np.clip(fitted * (1 - fitted), 1e-12, None)
    covariance = np.linalg.pinv(design.T @ (weights[:, None] * design))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    z_scores = np.divide(
        coefficients,
        standard_errors,
        out=np.zeros_like(coefficients),
        where=standard_errors > 0,
    )
    lower = coefficients - 1.96 * standard_errors
    upper = coefficients + 1.96 * standard_errors
    safe_exp = lambda values: np.exp(np.clip(values, -700, 700))
    return pd.DataFrame({
        "term": ["intercept", *names],
        "coefficient": coefficients,
        "odds_ratio": safe_exp(coefficients),
        "ci_95_lower": safe_exp(lower),
        "ci_95_upper": safe_exp(upper),
        "p_value": 2 * norm.sf(np.abs(z_scores)),
    })

# krijon tabelat e regresionit logjistik njevariabel dhe shumevariabel
def logistic_tables(df: pd.DataFrame, target: str, features: list[str]):
    y = df[target].astype(int).to_numpy()
    univariable = []
    for feature in features:
        X, names = _prepare_matrix(df, [feature])
        table = _wald_table(X, y, names)
        row = table.iloc[1].to_dict()
        row["model"] = "univariable"
        univariable.append(row)

    X, names = _prepare_matrix(df, features)
    multivariable = _wald_table(X, y, names)
    multivariable.insert(0, "model", "multivariable")
    return pd.DataFrame(univariable), multivariable

def main():
    out_dir = RESULTS_DIR / "statistical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = [
        "Analizat përdorin vetëm të dhënat e trajnimit dhe një rresht për çdo grup profilesh të përsëritura.",
        "Variablat e vazhdueshme: mesatarja (SD), mediana [IQR], testi t i Welch dhe Mann-Whitney U.",
        "Variablat kategorike: n (%), testi chi-square ose testi ekzakt i Fisher kur frekuencat e pritshme janë të vogla.",
        "Regresioni logjistik inferencial është i papeshuar dhe pa penalizim; ai është i ndarë nga modeli parashikues.",
        "Odds ratio për variablat e vazhdueshme dhe rendore raportohet për një rritje prej një devijimi standard.",
        "P-vlerat janë të pakorrigjuara dhe lidhjet nuk duhet të interpretohen si efekte shkakësore.",
    ]
    for dataset_name, data_cfg in DATASETS.items():
        target = data_cfg["target"]
        df = load_split(dataset_name, "train")
        if GROUP_COL in df:
            df = df.drop_duplicates(subset=GROUP_COL).copy()
        all_features = get_feature_columns(df, target, data_cfg["drop_cols"])
        features = [
            feature for feature in all_features
            if feature not in INFERENCE_EXCLUDE and df[feature].nunique(dropna=True) > 1
        ]

        descriptive_table(df, target, features).to_csv(
            out_dir / f"{dataset_name}_descriptive.csv", index=False
        )
        univariable, multivariable = logistic_tables(df, target, features)
        univariable.to_csv(
            out_dir / f"{dataset_name}_univariable_logistic.csv", index=False
        )
        multivariable.to_csv(
            out_dir / f"{dataset_name}_multivariable_logistic.csv", index=False
        )
        print(f"U ruajten tabelat statistikore per {dataset_name} ({len(df)} profile)")

    (out_dir / "METHODS_NOTES.txt").write_text("\n".join(notes) + "\n")

if __name__ == "__main__":
    main()
