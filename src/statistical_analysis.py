"""Classical descriptive tests and inferential Logistic Regression tables.

All analyses use the training split only. Duplicate profiles are represented
once so repeated records are not treated as independent observations.
"""

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
    CONTINUOUS_COLS,
    DATASETS,
    GROUP_COL,
    ORDINAL_COLS,
    RESULTS_DIR,
    load_split,
)
from preprocessing import get_feature_columns


# Derived interactions, scores, and deterministic cut-offs are useful for
# prediction but are omitted from classical inference to keep odds ratios
# interpretable and reduce multicollinearity.
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
    # Reference category for the complete set of work-type indicators.
    "work_type_private",
}


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
                "variabla": feature,
                "niveli": "e vazhdueshme",
                "grupi_0_mesatarja_DS": mean_0,
                "grupi_1_mesatarja_DS": mean_1,
                "grupi_0_mediana_IQR": median_0,
                "grupi_1_mediana_IQR": median_1,
                "testi_statistikor": "Testi t i Welch-it; testi U i Mann–Whitney-t",
                "vlera_p": float(ttest_ind(group_0, group_1, equal_var=False).pvalue),
                "vlera_p_Mann_Whitney": float(
                    mannwhitneyu(group_0, group_1, alternative="two-sided").pvalue
                ),
            })
            continue

        table = pd.crosstab(df[feature], df[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            test_name, p_value = "nuk zbatohet", np.nan
        else:
            _, chi_p, _, expected = chi2_contingency(table)
            if table.shape == (2, 2) and (expected < 5).any():
                _, p_value = fisher_exact(table.to_numpy())
                test_name = "Testi ekzakt i Fisher-it"
            else:
                p_value, test_name = chi_p, "Testi khi-katror (χ²)"

        for level, counts in table.iterrows():
            n_0 = int(counts.get(0, 0))
            n_1 = int(counts.get(1, 0))
            denom_0 = int((df[target] == 0).sum())
            denom_1 = int((df[target] == 1).sum())
            rows.append({
                "variabla": feature,
                "niveli": level,
                "grupi_0_n_perqindje": f"{n_0} ({100 * n_0 / denom_0:.1f}%)",
                "grupi_1_n_perqindje": f"{n_1} ({100 * n_1 / denom_1:.1f}%)",
                "testi_statistikor": test_name,
                "vlera_p": float(p_value),
                "vlera_p_Mann_Whitney": np.nan,
            })
    return pd.DataFrame(rows)


def _prepare_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    matrix = df[features].astype(float).to_numpy(copy=True)
    names = []
    scale_features = set(CONTINUOUS_COLS) | set(ORDINAL_COLS)
    for index, feature in enumerate(features):
        if feature in scale_features:
            std = matrix[:, index].std(ddof=0)
            if std > 0:
                matrix[:, index] = (matrix[:, index] - matrix[:, index].mean()) / std
            names.append(f"{feature} (për rritje me 1 DS)")
        else:
            names.append(f"{feature} (për rritje me 1 njësi)")
    return matrix, names


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
        "variabla": ["konstantja", *names],
        "koeficienti": coefficients,
        "raporti_i_gjasave_OR": safe_exp(coefficients),
        "IK_95_kufiri_i_poshtem": safe_exp(lower),
        "IK_95_kufiri_i_siperm": safe_exp(upper),
        "vlera_p": 2 * norm.sf(np.abs(z_scores)),
    })


def logistic_tables(df: pd.DataFrame, target: str, features: list[str]):
    y = df[target].astype(int).to_numpy()
    univariable = []
    for feature in features:
        X, names = _prepare_matrix(df, [feature])
        table = _wald_table(X, y, names)
        row = table.iloc[1].to_dict()
        row["modeli"] = "njëvariabël"
        univariable.append(row)

    X, names = _prepare_matrix(df, features)
    multivariable = _wald_table(X, y, names)
    multivariable.insert(0, "modeli", "shumëvariabël")
    return pd.DataFrame(univariable), multivariable


def main():
    out_dir = RESULTS_DIR / "statistical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = [
        "Analizat përdorin vetëm të dhënat e trajnimit dhe një rresht për çdo profil të përsëritur.",
        "Variablat e vazhdueshme paraqiten me mesatare (DS), medianë [diapazon ndërkuartilor, IQR], testin t të Welch-it dhe testin U të Mann–Whitney-t.",
        "Variablat kategorike paraqiten me n (%), testin khi-katror ose testin ekzakt të Fisher-it kur frekuencat e pritshme janë të vogla.",
        "Regresioni Logjistik inferencial është pa peshim dhe pa penalizim; ai është i ndarë nga modeli parashikues.",
        "Për variablat e vazhdueshme dhe rendore, raporti i gjasave jepet për një rritje me një devijim standard.",
        "Vlerat p nuk janë korrigjuar për krahasime të shumëfishta dhe lidhjet nuk duhet të interpretohen si shkakësore.",
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
        print(f"Saved statistical tables for {dataset_name} ({len(df)} profiles)")

    (out_dir / "METHODS_NOTES.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
