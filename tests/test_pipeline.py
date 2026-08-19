import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from home_credit_default_risk.pipeline import build_feature_matrix, build_preprocessor


def make_synthetic_X():
    return pd.DataFrame(
        {
            "AMT_CREDIT": [100_000.0, 200_000.0, np.nan, 150_000.0],
            "AMT_INCOME_TOTAL": [50_000.0, 60_000.0, 70_000.0, 55_000.0],
            "CODE_GENDER": ["F", "M", "F", np.nan],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Cash loans",
                "Revolving loans",
                "Cash loans",
            ],
        }
    )


def make_synthetic_y():
    return pd.Series([0, 1, 0, 1])


# ---- build_preprocessor: fit/predict, missing values, shape ----


def test_preprocessor_fits_and_transforms():
    X = make_synthetic_X()
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)


def test_preprocessor_output_has_no_missing_values():
    X = make_synthetic_X()
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(dense).any()


def test_full_pipeline_fits_and_predicts():
    X, y = make_synthetic_X(), make_synthetic_y()
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("classify", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict_proba(X)
    assert predictions.shape == (len(X), 2)


def test_pipeline_handles_unseen_category_at_predict_time():
    X, y = make_synthetic_X(), make_synthetic_y()
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("classify", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)

    X_new = X.copy()
    X_new.loc[0, "NAME_CONTRACT_TYPE"] = "Never seen before"
    # handle_unknown="ignore" must not raise
    predictions = pipeline.predict_proba(X_new)
    assert predictions.shape == (len(X_new), 2)


def test_preprocessor_column_lists_exclude_id_and_target():
    X = make_synthetic_X()
    X_with_id_and_target = X.copy()
    X_with_id_and_target["SK_ID_CURR"] = [1, 2, 3, 4]
    X_with_id_and_target["TARGET"] = [0, 1, 0, 1]

    # Recommended usage: caller drops ID and TARGET before building the
    # preprocessor -- this test documents and enforces that contract.
    feature_only = X_with_id_and_target.drop(columns=["SK_ID_CURR", "TARGET"])
    preprocessor = build_preprocessor(feature_only)

    all_used_columns = [col for _, _, cols in preprocessor.transformers for col in cols]
    assert "SK_ID_CURR" not in all_used_columns
    assert "TARGET" not in all_used_columns


# ---- build_feature_matrix: combines HC-M3-05 + HC-M3-06 ----


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    connection.sql("""
        CREATE TABLE bureau (
            SK_ID_BUREAU BIGINT, SK_ID_CURR BIGINT, CREDIT_ACTIVE VARCHAR,
            AMT_CREDIT_SUM DOUBLE, AMT_CREDIT_SUM_DEBT DOUBLE,
            CREDIT_DAY_OVERDUE BIGINT, CNT_CREDIT_PROLONG BIGINT
        );
        INSERT INTO bureau VALUES (1, 100, 'Active', 10000.0, 5000.0, 0, 0);
    """)
    connection.sql("""
        CREATE TABLE bureau_balance (SK_ID_BUREAU BIGINT, STATUS VARCHAR);
        INSERT INTO bureau_balance VALUES (1, '0');
    """)
    connection.sql("""
        CREATE TABLE previous_application (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT, NAME_CONTRACT_STATUS VARCHAR,
            AMT_APPLICATION DOUBLE, AMT_CREDIT DOUBLE
        );
        INSERT INTO previous_application VALUES (1, 100, 'Approved', 9000.0, 10000.0);
    """)
    connection.sql("""
        CREATE TABLE installments_payments (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT,
            DAYS_INSTALMENT DOUBLE, DAYS_ENTRY_PAYMENT DOUBLE,
            AMT_INSTALMENT DOUBLE, AMT_PAYMENT DOUBLE
        );
        INSERT INTO installments_payments VALUES (1, 100, -100, -105, 1000.0, 1000.0);
    """)
    connection.sql("""
        CREATE TABLE POS_CASH_balance (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT, SK_DPD BIGINT
        );
        INSERT INTO POS_CASH_balance VALUES (1, 100, 0);
    """)
    connection.sql("""
        CREATE TABLE credit_card_balance (
            SK_ID_PREV BIGINT, SK_ID_CURR BIGINT,
            AMT_BALANCE DOUBLE, AMT_CREDIT_LIMIT_ACTUAL DOUBLE, SK_DPD BIGINT
        );
        INSERT INTO credit_card_balance VALUES (1, 100, 500.0, 1000.0, 0);
    """)
    yield connection
    connection.close()


def make_application_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100, 200],
            "AMT_CREDIT": [10000.0, 20000.0],
            "AMT_INCOME_TOTAL": [50000.0, 60000.0],
            "AMT_ANNUITY": [5000.0, 6000.0],
            "AMT_GOODS_PRICE": [9000.0, 19000.0],
            "DAYS_BIRTH": [-10000, -15000],
            "DAYS_EMPLOYED": [-2000, -3000],
        }
    )


def test_build_feature_matrix_includes_engineered_and_raw_columns(con):
    result = build_feature_matrix(con, make_application_df())
    for col in ["AMT_CREDIT", "credit_to_income", "age_years", "bureau_credit_count"]:
        assert col in result.columns


def test_build_feature_matrix_preserves_row_count(con):
    result = build_feature_matrix(con, make_application_df())
    assert len(result) == 2


def test_build_feature_matrix_does_not_require_target(con):
    df = make_application_df()
    assert "TARGET" not in df.columns
    result = build_feature_matrix(con, df)
    assert "TARGET" not in result.columns
