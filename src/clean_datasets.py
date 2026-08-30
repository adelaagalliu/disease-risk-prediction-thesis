import numpy as np
import pandas as pd

AGE_RANGE_EDGES = [
    18, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 75, 80, np.inf
]
AGE_RANGE_LABELS = list(range(1, 14))

def age_years_to_ranges(age_years):
    age = pd.to_numeric(age_years, errors="coerce")
    return pd.cut(
        age,
        bins=AGE_RANGE_EDGES,
        labels=AGE_RANGE_LABELS,
        right=False
    ).astype("Int8")

def normalize_categorical(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

def convert_to_binary(series):
    s = normalize_categorical(series)
    return s.map({
        "yes": 1,
        "no": 0
    }).astype("Int8")
