"""`HC-M4-10` — batch scoring CLI.

A second composition root, alongside the FastAPI app
(`adapters/api/main.py`): builds the same concrete adapters, wires them
into the same `ScoreApplicantUseCase`, and drives it from a CSV of
`SK_ID_CURR` values instead of HTTP requests. Proves the use case is
genuinely entrypoint-agnostic -- neither this file nor the API imports
the other, and neither contains any scoring logic of its own.

Usage:
    uv run python scripts/score_batch.py applicants.csv predictions.csv

`applicants.csv` must have a `SK_ID_CURR` column. Unknown applicants are
recorded in the output with an `error` column rather than silently
dropped or crashing the whole batch.
"""

import argparse
import logging

import pandas as pd

from home_credit_default_risk import config
from home_credit_default_risk.adapters.local_store import LocalFeatureStore
from home_credit_default_risk.adapters.mlflow_registry import MlflowModelRegistry
from home_credit_default_risk.application.score import (
    ScoreApplicantUseCase,
    UnknownApplicantError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("score_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV with a SK_ID_CURR column")
    parser.add_argument("output_csv", help="Where to write predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    applicants = pd.read_csv(args.input_csv)
    if "SK_ID_CURR" not in applicants.columns:
        raise ValueError(f"{args.input_csv} must have a SK_ID_CURR column")

    feature_store = LocalFeatureStore(config.FEATURE_STORE_PATH)
    model_registry = MlflowModelRegistry(
        tracking_uri=config.MLFLOW_TRACKING_URI,
        model_name=config.MLFLOW_MODEL_NAME,
        alias=config.MLFLOW_PRODUCTION_ALIAS,
    )
    use_case = ScoreApplicantUseCase(feature_store, model_registry)

    rows = []
    for sk_id_curr in applicants["SK_ID_CURR"]:
        try:
            decision = use_case.score(int(sk_id_curr))
        except UnknownApplicantError as exc:
            logger.warning("SK_ID_CURR=%s: %s", sk_id_curr, exc)
            rows.append({"SK_ID_CURR": sk_id_curr, "error": str(exc)})
            continue

        rows.append(
            {
                "SK_ID_CURR": sk_id_curr,
                "probability": decision.probability,
                "is_high_risk": decision.is_high_risk,
                "threshold": decision.threshold,
                "error": None,
            }
        )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output_csv, index=False)

    scored = predictions["error"].isna().sum()
    logger.info(
        "Scored %d/%d applicants, wrote %s", scored, len(predictions), args.output_csv
    )


if __name__ == "__main__":
    main()
