"""Basic derived features for application-level data (`HC-M3-05`).

Groups A (credit/income ratios) and B (age/employment stability) from
`docs/feature_engineering_strategy.md`. Every function takes only the
application-level feature dataframe — never `TARGET` — so target leakage
is impossible by construction, not just by convention.
"""

import pandas as pd

from home_credit_default_risk.utils import safe_divide

# DuckDB's read_csv_auto (scripts/profile_data.py) doesn't override this
# column, so it arrives as the raw sentinel Home Credit uses for "not
# currently employed" (confirmed in notebooks/01_data_understanding.ipynb:
# 18.01% of applicants, ~99.96% of them Pensioners).
DAYS_EMPLOYED_SENTINEL = 365243


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group A — credit/income ratios. Returns a copy; does not mutate `df`."""
    df = df.copy()
    df["credit_to_income"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["annuity_to_income"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["credit_to_annuity"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["goods_price_to_credit"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    return df


def add_age_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group B — age & employment stability.

    Recodes the `DAYS_EMPLOYED` sentinel to missing *before* deriving
    `employment_years`, so the ~18% of applicants who hit it don't report
    a fabricated ~1000-year employment history. `is_employment_unknown`
    preserves that information as a feature rather than discarding it —
    per notebook 01, it's almost entirely `NAME_INCOME_TYPE == 'Pensioner'`,
    so it's signal, not noise.
    """
    df = df.copy()
    is_employment_unknown = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    days_employed = df["DAYS_EMPLOYED"].where(~is_employment_unknown)

    df["age_years"] = -df["DAYS_BIRTH"] / 365.25
    df["employment_years"] = -days_employed / 365.25
    df["is_employment_unknown"] = is_employment_unknown.astype(int)
    df["employment_to_age_ratio"] = safe_divide(df["employment_years"], df["age_years"])
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every `HC-M3-05` basic derived feature (groups A + B)."""
    df = add_ratio_features(df)
    df = add_age_employment_features(df)
    return df
