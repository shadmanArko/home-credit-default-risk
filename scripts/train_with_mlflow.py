"""`HC-M4-02`/`03` — train the final model and register it in MLflow.

Same training recipe as `train_final_model.py` (`load_development_data()`,
`build_final_pipeline()`, imported directly -- not duplicated), but the
fitted pipeline is persisted through the `ModelRegistry` port
(`MlflowModelRegistry`) instead of a loose joblib file: every run is
logged with its parameters and metrics, versioned, and the version is
promoted to the `production` alias that serving code (Milestone 4,
Chunk 3) will read from.

The CV/holdout metrics logged here are the real, already-verified numbers
from `HC-M3-15`/`19`-`21` -- not re-derived, since re-running the holdout
evaluation would violate this project's "touched exactly once" rule.
"""

from home_credit_default_risk import config
from home_credit_default_risk.adapters.mlflow_registry import MlflowModelRegistry

KNOWN_METRICS = {
    "cv_roc_auc": 0.7747,
    "holdout_roc_auc": 0.7791,
    "holdout_pr_auc": 0.2758,
}


def main() -> None:
    from train_final_model import (
        FINAL_MODEL_PARAMS,
        build_final_pipeline,
        load_development_data,
    )

    X, y = load_development_data()
    pipeline = build_final_pipeline(X)
    pipeline.fit(X, y)

    registry = MlflowModelRegistry(
        tracking_uri=config.MLFLOW_TRACKING_URI,
        model_name=config.MLFLOW_MODEL_NAME,
        alias=config.MLFLOW_PRODUCTION_ALIAS,
    )
    version = registry.register_and_promote(
        pipeline, metrics=KNOWN_METRICS, params=FINAL_MODEL_PARAMS
    )

    print(f"Registered '{config.MLFLOW_MODEL_NAME}' version {version}")
    print(f"Promoted to alias '{config.MLFLOW_PRODUCTION_ALIAS}'")
    print(f"Tracking URI: {config.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
