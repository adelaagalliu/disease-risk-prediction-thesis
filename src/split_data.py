import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    DATASETS, DATA_PROCESSED, GROUP_COL, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    USE_FEATURE_ENGINEERING,
)
from preprocessing import add_engineered_features
from missing_values import (MISSING_VALUE_DATASETS, drop_unimputable_rows, fit_impute_stats, apply_impute)

N_SPLITS = 10  # 7 folds train / 1 fold val / 2 folds test
SMOKER_MASK_FRACTION = 0.05
SMOKER_MASK_DATASETS = {"diabetes", "heart_disease", "hypertension"}

def make_group_key(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[feature_cols], index=False)

def mask_training_smoking(train_df: pd.DataFrame, dataset_name: str) -> None:
    if dataset_name not in SMOKER_MASK_DATASETS:
        return

    known_idx = train_df.index[train_df["smoker"].notna()].to_numpy()
    n_mask = round(len(known_idx) * SMOKER_MASK_FRACTION)
    if n_mask:
        rng = np.random.default_rng(RANDOM_STATE)
        masked_idx = rng.choice(known_idx, size=n_mask, replace=False)
        train_df.loc[masked_idx, "smoker"] = np.nan
        if dataset_name == "hypertension":
            train_df.loc[masked_idx, "cigs_per_day"] = np.nan
    print(
        f"  u maskuan {n_mask:,}/{len(known_idx):,} vlera te njohura te duhanpirjes "
        f"ne grupin e trajnimit ({SMOKER_MASK_FRACTION:.0%})"
    )

def split_one_dataset(name: str, data_cfg: dict):
    df = pd.read_csv(data_cfg["csv"])
    target = data_cfg["target"]
    assert target in df.columns, f"[{name}] kolona target '{target}' nuk u gjet"

    if name in MISSING_VALUE_DATASETS:
        df = drop_unimputable_rows(df, name)

    raw_feature_cols = [c for c in df.columns if c != target]
    n_rows = len(df)

    group_key = make_group_key(df, raw_feature_cols)
    print(f"  distinct feature-groups: {group_key.nunique():,}")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(sgkf.split(X=df[raw_feature_cols], y=df[target], groups=group_key))

    train_idx = pd.concat([pd.Series(folds[i][1]) for i in range(7)]).unique()
    val_idx = folds[7][1]
    test_idx = pd.concat([pd.Series(folds[i][1]) for i in range(8, 10)]).unique()

    splits = {
        "train": df.iloc[train_idx].copy(),
        "val": df.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy(),
    }
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        splits[split_name][GROUP_COL] = group_key.iloc[idx].to_numpy()

    mask_training_smoking(splits["train"], name)

    if name in MISSING_VALUE_DATASETS:
        impute_stats = fit_impute_stats(splits["train"], name)
        splits = {split_name: apply_impute(split_df, name, impute_stats)
                   for split_name, split_df in splits.items()}

    if USE_FEATURE_ENGINEERING:
        n_cols_before = {split_name: split_df.shape[1] for split_name, split_df in splits.items()}
        splits = {split_name: add_engineered_features(split_df)
                   for split_name, split_df in splits.items()}
        added = splits["train"].shape[1] - n_cols_before["train"]
        print(f"  feature engineering: u shtuan {added} kolona "
              f"({splits['train'].shape[1]} total)")

    # verifikojme qe asnje rresht nuk humbi ose u perserit mes ndarjeve
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == n_rows, f"[{name}] rreshta te humbur gjate ndarjes"
    assert (set(train_idx) & set(val_idx) == set()
            and set(train_idx) & set(test_idx) == set()
            and set(val_idx) & set(test_idx) == set()), f"[{name}] mbivendosje"

    # kontroll: grupet nuk mbivendosen mes train, validation dhe test
    split_groups = {
        split_name: set(group_key.iloc[idx])
        for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
    }
    assert split_groups["train"].isdisjoint(split_groups["val"]), f"[{name}] grupet train/val kane mbivendosje"
    assert split_groups["train"].isdisjoint(split_groups["test"]), f"[{name}] grupet train/test kane mbivendosje"
    assert split_groups["val"].isdisjoint(split_groups["test"]), f"[{name}] grupet val/test kane mbivendosje"

    # verifikojme qe targeti ka vetem vlera binare
    for split_name, split_df in splits.items():
        assert split_df[target].isin([0, 1]).all(), f"[{name}] ka vlera invalide ne {split_name}"
        assert split_df[GROUP_COL].notna().all(), f"[{name}] mungojne identifikuesit e grupeve ne ndarjen {split_name}"

    # verifikojme qe kolonat e modelit nuk kanë vlera qe mungojne 
    excluded_cols = set(data_cfg["drop_cols"]) | {target, GROUP_COL}
    model_cols = [c for c in splits["train"].columns if c not in excluded_cols]
    for split_name, split_df in splits.items():
        missing_by_col = split_df[model_cols].isna().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]
        assert missing_by_col.empty, (
            f"[{name}] pas preprocessing, ne ndarjen {split_name} kane mbetur "
            f"vlera qe mungojne ne kolonat e modelit: {missing_by_col.to_dict()}"
        )

    # kontrollojme qe ndarjet te jene afer raporteve 70/10/20
    expected_frac = {"train": 1 - TEST_SIZE - VAL_SIZE, "val": VAL_SIZE, "test": TEST_SIZE}
    for split_name, split_df in splits.items():
        actual_frac = len(split_df) / n_rows
        if abs(actual_frac - expected_frac[split_name]) > 0.025:
            print(f"  WARNING: {split_name} perben {actual_frac:.2%} te te dhenave "
                  f"(target {expected_frac[split_name]:.0%}); madhesite e grupeve ndikojne ne ndarje")

    # ruaj ndarjet dhe raportojme madhesine dhe normen e klases pozitive
    for split_name, split_df in splits.items():
        out_path = DATA_PROCESSED / f"{name}_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        pos_rate = split_df[target].mean()
        print(
        f"{split_name}: {len(split_df):,} rows "
        f"({len(split_df) / n_rows:.2%}) | "
        f"{target} positive rate = {pos_rate:.2%} -> {out_path}")

    # kontrollojme qe norma e klases pozitive eshte e ngjashme mes ndarjeve
    rates = [s[target].mean() for s in splits.values()]
    drift = max(rates) - min(rates)
    print(f"  class-rate drift across splits: {drift:.2%}")
    if drift > 0.02:
        print(f"  WARNING: ndryshimi i normes se klases tejkalon 2 pike perqindje "
              f"(norma e përgjithshme= {pos_rate:.2%})")

def main():
    assert abs(TEST_SIZE - 0.2) < 1e-9 and abs(VAL_SIZE - 0.1) < 1e-9, (
    )
    for name, data_cfg in DATASETS.items():
        split_one_dataset(name, data_cfg)

if __name__ == "__main__":
    main()
