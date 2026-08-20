"""`HC-M4-12` — data + prediction drift report (Evidently).

Compares the development pool (what the model was trained/CV'd on) against
`application_test` -- the one batch in this dataset that has never been
used for training, model selection, tuning, or the holdout evaluation
(`HC-M3-21`). It's the closest thing this project has to genuinely "new"
production data, so it's the honest choice for a drift baseline rather
than reusing the already-evaluated holdout.

Both feature drift (are `application_test`'s inputs distributed
differently from what the model was trained on?) and prediction drift
(has the model's predicted-probability distribution shifted?) are
checked in one report -- the predicted probability is just added as an
extra numeric column before running Evidently's drift preset, so both
checks share the same statistical-test machinery rather than needing two
separate reports.

Output: `reports/monitoring/drift_report.html` (gitignored, regenerable
-- see `HC-M4-13` for what a real deployment would do when this flags
drift).
"""

import time

import duckdb
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from home_credit_default_risk import config
from home_credit_default_risk.adapters.mlflow_registry import MlflowModelRegistry
from home_credit_default_risk.pipeline import build_feature_matrix


def _load_engineered_features(
    con: duckdb.DuckDBPyConnection, table: str
) -> pd.DataFrame:
    raw = con.sql(f"SELECT * FROM {table}").df()
    raw = raw.drop(columns=["TARGET"], errors="ignore")
    features = build_feature_matrix(con, raw)
    return features.drop(columns=["SK_ID_CURR"])


def main() -> None:
    con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    split = pd.read_csv(config.SPLIT_PATH)
    dev_ids = split.loc[split["split"] == "train", "SK_ID_CURR"]

    application_train = con.sql("SELECT * FROM application_train").df()
    development = application_train[
        application_train["SK_ID_CURR"].isin(dev_ids)
    ].reset_index(drop=True)

    reference_con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    reference = build_feature_matrix(
        reference_con, development.drop(columns=["TARGET"])
    ).drop(columns=["SK_ID_CURR"])
    reference_con.close()

    current_con = duckdb.connect(str(config.CACHE_DB), read_only=True)
    current = _load_engineered_features(current_con, "application_test")
    current_con.close()
    con.close()

    registry = MlflowModelRegistry(
        tracking_uri=config.MLFLOW_TRACKING_URI,
        model_name=config.MLFLOW_MODEL_NAME,
        alias=config.MLFLOW_PRODUCTION_ALIAS,
    )
    model = registry.get_production_model()

    reference = reference.copy()
    current = current.copy()
    reference["predicted_probability"] = model.predict(reference)
    current["predicted_probability"] = model.predict(current)

    data_definition = DataDefinition()
    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    start = time.time()
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_dataset, reference_dataset)
    elapsed = time.time() - start

    config.MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.MONITORING_DIR / "drift_report.html"
    snapshot.save_html(str(output_path))

    print(f"Reference (development pool): {len(reference):,} rows")
    print(f"Current (application_test):   {len(current):,} rows")
    print(f"Drift computation: {elapsed:.1f}s")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
