import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2_contingency, fisher_exact, norm, ttest_ind
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

# Variabla te krijuara qe nuk përfshihen ne analize
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
# kthen mesataren dhe devijimin standard ne formatin mesatarja (DS)
def _format_mean_sd(values: pd.Series) -> str:
    values = values.dropna().astype(float)
    if values.empty:
        return "NA"
    mean = values.mean()
    standard_deviation = values.std(ddof=1)
    return f"{mean:.2f} ({standard_deviation:.2f})"
def descriptive_table(df: pd.DataFrame, target: str, features: list[str],) -> pd.DataFrame:
    rows = []
    continuous_features = set(CONTINUOUS_COLS)
    for feature in features:
        # analiza e variablave te vazhdueshme
        if feature in continuous_features:
            group_0 = df.loc[df[target] == 0, feature].dropna()
            group_1 = df.loc[df[target] == 1, feature].dropna()
            if group_0.empty or group_1.empty:
                p_value = np.nan
            else:
                welch_result = ttest_ind(
                    group_0,
                    group_1,
                    equal_var=False,
                    nan_policy="omit",
                )
                p_value = float(welch_result.pvalue)
            rows.append(
                {
                    "feature": feature,
                    "level": "continuous",
                    "target_0_mean_sd": _format_mean_sd(group_0),
                    "target_1_mean_sd": _format_mean_sd(group_1),
                    "test": "Welch t-test",
                    "p_value": p_value,
                }
            )
            continue
        # analiza e variablave kategorike
        contingency_table = pd.crosstab(df[feature], df[target])
        if (
            contingency_table.shape[0] < 2
            or contingency_table.shape[1] < 2
        ):
            test_name = "not applicable"
            p_value = np.nan
        else:
            _, chi_square_p, _, expected = chi2_contingency(
                contingency_table
            )
            # fisher perdoret per tabela 2x2 kur te pakten 1 frekuenc e pritshme <5
            if (
                contingency_table.shape == (2, 2)
                and (expected < 5).any()
            ):
                _, fisher_p = fisher_exact(
                    contingency_table.to_numpy()
                )
                test_name = "Fisher exact"
                p_value = float(fisher_p)
            else:
                test_name = "chi-square"
                p_value = float(chi_square_p)

        target_0_total = int((df[target] == 0).sum())
        target_1_total = int((df[target] == 1).sum())

        for level, counts in contingency_table.iterrows():
            target_0_count = int(counts.get(0, 0))
            target_1_count = int(counts.get(1, 0))
            target_0_percentage = (
                100 * target_0_count / target_0_total
                if target_0_total > 0
                else np.nan
            )
            target_1_percentage = (
                100 * target_1_count / target_1_total
                if target_1_total > 0
                else np.nan
            )
            rows.append(
                {
                    "feature": feature,
                    "level": level,
                    "target_0_n_percent": (
                        f"{target_0_count} "
                        f"({target_0_percentage:.1f}%)"
                    ),
                    "target_1_n_percent": (
                        f"{target_1_count} "
                        f"({target_1_percentage:.1f}%)"
                    ),
                    "test": test_name,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)
# pergatit matricen e regresionit, standardizon variablat e vazhdueshme dhe rendore
def _prepare_matrix(df: pd.DataFrame, features: list[str],) -> tuple[np.ndarray, list[str]]:
    matrix = df[features].astype(float).to_numpy(copy=True)
    names = []
    scale_features = set(CONTINUOUS_COLS) | set(ORDINAL_COLS)
    for index, feature in enumerate(features):
        if feature in scale_features:
            mean = matrix[:, index].mean()
            standard_deviation = matrix[:, index].std(ddof=0)
            if standard_deviation > 0:
                matrix[:, index] = (
                    matrix[:, index] - mean
                ) / standard_deviation
            names.append(f"{feature} (per 1 SD)")
        else:
            names.append(f"{feature} (per 1 unit)")
    return matrix, names
# kryen regresionin logjistik dhe llogarit: koeficientët, odds ratio, intervalet e besimit 95% dhe p-vlerat
def _wald_table(X: np.ndarray, y: np.ndarray, names: list[str],) -> pd.DataFrame:
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    coefficients = np.concatenate(
        [model.intercept_, model.coef_[0]]
    )
    design_matrix = np.column_stack(
        [np.ones(len(X)), X]
    )
    fitted_probabilities = expit(
        design_matrix @ coefficients
    )
    weights = np.clip(
        fitted_probabilities * (1 - fitted_probabilities),
        1e-12,
        None,
    )
    covariance = np.linalg.pinv(
        design_matrix.T
        @ (weights[:, None] * design_matrix)
    )
    standard_errors = np.sqrt(
        np.clip(np.diag(covariance), 0, None)
    )
    z_scores = np.divide(
        coefficients,
        standard_errors,
        out=np.zeros_like(coefficients),
        where=standard_errors > 0,
    )
    lower_limit = coefficients - 1.96 * standard_errors
    upper_limit = coefficients + 1.96 * standard_errors
    odds_ratio = np.exp(
        np.clip(coefficients, -700, 700)
    )
    confidence_interval_lower = np.exp(
        np.clip(lower_limit, -700, 700)
    )
    confidence_interval_upper = np.exp(
        np.clip(upper_limit, -700, 700)
    )
    return pd.DataFrame(
        {
            "term": ["intercept", *names],
            "coefficient": coefficients,
            "odds_ratio": odds_ratio,
            "ci_95_lower": confidence_interval_lower,
            "ci_95_upper": confidence_interval_upper,
            "p_value": 2 * norm.sf(np.abs(z_scores)),
        }
    )
# tabelat e regresionit logjistik njevariabel dhe shumevariabel
def logistic_tables(df: pd.DataFrame, target: str, features: list[str],) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target].astype(int).to_numpy()
    univariable_rows = []
    for feature in features:
        X, names = _prepare_matrix(df, [feature])
        result = _wald_table(X, y, names)

        feature_row = result.iloc[1].to_dict()
        feature_row["model"] = "univariable"
        univariable_rows.append(feature_row)

    X, names = _prepare_matrix(df, features)
    multivariable_table = _wald_table(X, y, names)
    multivariable_table.insert(
        0,
        "model",
        "multivariable",
    )
    return (
        pd.DataFrame(univariable_rows),
        multivariable_table,
    )

def main():
    output_directory = RESULTS_DIR / "statistical_analysis"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    methodology_notes = [
        (
            "Analizat përdorin vetëm të dhënat e trajnimit dhe "
            "një rresht për çdo grup profilesh të përsëritura."
        ),
        (
            "Variablat e vazhdueshme paraqiten me mesatare dhe "
            "devijim standard dhe krahasohen me testin t të Welch-it."
        ),
        (
            "Variablat kategorike paraqiten me frekuenca dhe "
            "përqindje dhe analizohen me testin khi-katror ose "
            "testin ekzakt të Fisher-it kur frekuencat e pritshme "
            "në tabelat 2x2 janë të vogla."
        ),
        (
            "Regresioni logjistik inferencial është i papeshuar "
            "dhe pa penalizim dhe është i ndarë nga modeli parashikues."
        ),
        (
            "Odds ratio për variablat e vazhdueshme dhe rendore "
            "raportohet për një rritje prej një devijimi standard."
        ),
        (
            "P-vlerat janë të pakorrigjuara për krahasime të "
            "shumëfishta dhe lidhjet nuk interpretohen si shkakësore."
        ),
    ]

    for dataset_name, dataset_config in DATASETS.items():
        target = dataset_config["target"]
        df = load_split(dataset_name, "train")
        if GROUP_COL in df.columns:
            df = df.drop_duplicates(subset=GROUP_COL).copy()
        all_features = get_feature_columns(df,target,dataset_config["drop_cols"],)
        features = [
            feature
            for feature in all_features
            if (
                feature not in INFERENCE_EXCLUDE
                and df[feature].nunique(dropna=True) > 1
            )
        ]
        descriptive_results = descriptive_table(df,target,features,)
        descriptive_results.to_csv(output_directory / f"{dataset_name}_descriptive.csv",index=False,)
        (univariable_results, multivariable_results,) = logistic_tables(df,target,features,)

        univariable_results.to_csv(output_directory/ f"{dataset_name}_univariable_logistic.csv",index=False,)
        multivariable_results.to_csv(output_directory / f"{dataset_name}_multivariable_logistic.csv",index=False,)

        print(
            f"U ruajtën tabelat statistikore për "
            f"{dataset_name} ({len(df)} profile)."
        )

    notes_path = output_directory / "METHODS_NOTES.txt"
    notes_path.write_text(
        "\n".join(methodology_notes) + "\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()