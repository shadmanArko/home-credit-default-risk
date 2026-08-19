"""Leakage-safe preprocessing pipeline (`HC-M3-09`).

Two separate concerns, deliberately not fused into one step:

1. **Feature engineering** (`build_feature_matrix`) — `HC-M3-05`'s ratio/
   age features and `HC-M3-06`'s historical aggregations. These are pure
   per-row / per-key deterministic computations (no cross-row statistic
   is ever fit — verified in `docs/feature_leakage_audit.md`), so it's
   safe to apply this once to a full dataset rather than separately per
   CV fold; the result is identical either way.
2. **Preprocessing** (`build_preprocessor`) — median imputation +
   `RobustScaler` for numeric columns, most-frequent imputation +
   one-hot encoding for categorical columns. Unlike step 1, this
   genuinely fits statistics from data (imputation values, scaler
   ranges, the encoder's category vocabulary), so it must only ever be
   fit inside a `Pipeline`, on a training fold — never on the full
   dataset before a CV split. `build_preprocessor` returns an unfitted
   `ColumnTransformer`; callers are responsible for putting it inside a
   `Pipeline` with an estimator and fitting only on training data. Same
   discipline established in `HC-M1-07`, now factored into reusable code.
"""

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from home_credit_default_risk.aggregations import build_historical_features
from home_credit_default_risk.features import add_basic_features

# Never a feature -- an identifier. Every function below either requires
# it (to join aggregates) or drops it before preprocessing; it must never
# reach a ColumnTransformer.
ID_COLUMN = "SK_ID_CURR"


def build_feature_matrix(
    con: duckdb.DuckDBPyConnection, application_df: pd.DataFrame
) -> pd.DataFrame:
    """Apply `HC-M3-05` + `HC-M3-06` feature engineering.

    `application_df` must not contain `TARGET` -- this function never
    references it, by the same construction as `features.py` and
    `aggregations.py`, so leakage through this step is impossible.
    """
    with_basic_features = add_basic_features(application_df)
    historical = build_historical_features(con, with_basic_features[ID_COLUMN])
    return with_basic_features.merge(historical, on=ID_COLUMN, how="left")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Unfitted `ColumnTransformer`: median-impute + scale numeric
    columns, most-frequent-impute + one-hot encode categorical columns.

    Column lists are derived from `X`'s dtypes at call time, so the same
    function works whether `X` is the plain application table or the
    full engineered feature matrix -- the set of columns doesn't change
    across CV folds, only which rows are present, so building this once
    per experiment (not once per fold) is correct.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
