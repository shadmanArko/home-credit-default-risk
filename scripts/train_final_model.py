"""Fit and persist the final model artifact (`HC-M3-19`/`20`'s frozen configuration).

Reproduces exactly what `notebooks/04_modeling.ipynb` does in `HC-M3-20` --
same feature matrix, same untuned `LGBMClassifier` defaults, same
`random_state` -- as a standalone, importable step so the final model can
be regenerated without re-running the whole notebook (including its
one-time hyperparameter search).

`load_development_data()` and `build_final_pipeline()` are also imported
directly by `scripts/train_with_mlflow.py` (`HC-M4-03`) -- one training
recipe, two ways to persist the result (a loose joblib file here, the
MLflow Model Registry there), per this project's DRY discipline.
"""

import time

import duckdb
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from home_credit_default_risk import config
from home_credit_default_risk.pipeline import build_feature_matrix, build_preprocessor

FINAL_MODEL_PARAMS = {
    "class_weight": "balanced",
    "random_state": config.RANDOM_STATE,
    "verbose": -1,
}
MODEL_PATH = config.PROJECT_ROOT / "models" / "lightgbm_final_pipeline.joblib"


def load_development_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and feature-engineer the development pool (`HC-M3-09`/`20`)."""
    con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    application_train = con.sql("SELECT * FROM application_train").df()
    con.close()

    split = pd.read_csv(config.SPLIT_PATH)
    dev_ids = split.loc[split["split"] == "train", "SK_ID_CURR"]
    development = application_train[
        application_train["SK_ID_CURR"].isin(dev_ids)
    ].reset_index(drop=True)

    con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    feature_matrix = build_feature_matrix(con, development.drop(columns=["TARGET"]))
    con.close()

    y = development["TARGET"]
    X = feature_matrix.drop(columns=["SK_ID_CURR"])
    return X, y


def build_final_pipeline(X: pd.DataFrame) -> Pipeline:
    """The frozen `HC-M3-19` configuration, unfitted."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("classify", LGBMClassifier(**FINAL_MODEL_PARAMS)),
        ]
    )


def main() -> None:
    X, y = load_development_data()
    pipeline = build_final_pipeline(X)

    start = time.time()
    pipeline.fit(X, y)
    fit_seconds = time.time() - start

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Fit on {len(X):,} rows in {fit_seconds:.1f}s")
    print(f"Saved final pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
