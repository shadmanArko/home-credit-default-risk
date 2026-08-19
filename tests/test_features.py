import numpy as np
import pandas as pd

from home_credit_default_risk.features import (
    DAYS_EMPLOYED_SENTINEL,
    add_age_employment_features,
    add_basic_features,
    add_ratio_features,
)


def make_application_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "AMT_CREDIT": [100_000.0, 200_000.0, 300_000.0, 0.0],
            "AMT_INCOME_TOTAL": [50_000.0, 0.0, 100_000.0, 60_000.0],
            "AMT_ANNUITY": [10_000.0, 20_000.0, np.nan, 5_000.0],
            "AMT_GOODS_PRICE": [90_000.0, 180_000.0, 300_000.0, np.nan],
            "DAYS_BIRTH": [-10_000, -15_000, -20_000, -12_000],
            "DAYS_EMPLOYED": [-2_000, DAYS_EMPLOYED_SENTINEL, -500, -1_000],
        }
    )


def test_ratio_features_compute_correctly():
    df = add_ratio_features(make_application_df())
    assert df.loc[0, "credit_to_income"] == 100_000.0 / 50_000.0
    assert df.loc[0, "annuity_to_income"] == 10_000.0 / 50_000.0


def test_zero_denominator_becomes_nan_not_inf():
    df = add_ratio_features(make_application_df())
    # row 1 has AMT_INCOME_TOTAL == 0
    assert pd.isna(df.loc[1, "credit_to_income"])
    assert pd.isna(df.loc[1, "annuity_to_income"])
    # row 3 has AMT_CREDIT == 0, used as a denominator for goods_price_to_credit
    assert pd.isna(df.loc[3, "goods_price_to_credit"])


def test_no_infinite_values_anywhere():
    df = add_ratio_features(make_application_df())
    numeric = df.select_dtypes(include="number")
    assert not np.isinf(numeric.to_numpy()).any()


def test_missing_numerator_propagates_as_nan():
    df = add_ratio_features(make_application_df())
    # row 2 has AMT_ANNUITY == NaN
    assert pd.isna(df.loc[2, "annuity_to_income"])


def test_days_employed_sentinel_recoded_to_missing():
    df = add_age_employment_features(make_application_df())
    assert pd.isna(df.loc[1, "employment_years"])
    assert df.loc[1, "is_employment_unknown"] == 1
    assert df.loc[0, "is_employment_unknown"] == 0


def test_employment_years_computed_for_normal_rows():
    df = add_age_employment_features(make_application_df())
    assert df.loc[0, "employment_years"] == 2_000 / 365.25
    assert df.loc[0, "age_years"] == 10_000 / 365.25


def test_add_basic_features_applies_both_groups():
    df = add_basic_features(make_application_df())
    for col in [
        "credit_to_income",
        "age_years",
        "employment_years",
        "is_employment_unknown",
    ]:
        assert col in df.columns


def test_does_not_require_or_touch_target():
    df = make_application_df()
    assert "TARGET" not in df.columns
    # Functions must not raise or implicitly require a TARGET column —
    # this is what makes leakage impossible by construction, not convention.
    result = add_basic_features(df)
    assert "TARGET" not in result.columns


def test_input_dataframe_is_not_mutated():
    df = make_application_df()
    original_columns = list(df.columns)
    add_basic_features(df)
    assert list(df.columns) == original_columns
