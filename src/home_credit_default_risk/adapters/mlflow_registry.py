"""`HC-M4-03` — MLflow implementation of the `ModelRegistry` port.

Uses `mlflow.pyfunc` rather than the `mlflow.sklearn` flavor: the full
`mlflow` package (which ships that flavor) requires `pandas<3`, which
conflicts with this project's `pandas>=3.0.5` — the same class of
dependency conflict already documented for `ydata-profiling` in the
README. `mlflow-skinny` (no such pin) plus a small `pyfunc.PythonModel`
wrapper around the existing joblib-serializable sklearn `Pipeline` gets
the same result without the conflict.

The returned production model exposes a single `.predict(dataframe)`
method returning the probability of the positive class — the wrapper
below is what makes that true regardless of which sklearn estimator is
actually registered.
"""

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.client import MlflowClient

from home_credit_default_risk.domain.ports import ModelRegistry


class _SklearnPipelineModel(mlflow.pyfunc.PythonModel):
    """Wraps an arbitrary joblib-serialized sklearn `Pipeline` so MLflow
    can log/load it without the `mlflow.sklearn` flavor."""

    def load_context(self, context: Any) -> None:
        self._pipeline = joblib.load(context.artifacts["pipeline"])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict | None = None
    ):
        return self._pipeline.predict_proba(model_input)[:, 1]


class MlflowModelRegistry(ModelRegistry):
    """`ModelRegistry` backed by an MLflow tracking server + model registry."""

    def __init__(self, tracking_uri: str, model_name: str, alias: str) -> None:
        self._tracking_uri = tracking_uri
        self._model_name = model_name
        self._alias = alias
        mlflow.set_tracking_uri(tracking_uri)

    def get_production_model(self) -> mlflow.pyfunc.PyFuncModel:
        mlflow.set_tracking_uri(self._tracking_uri)
        model_uri = f"models:/{self._model_name}@{self._alias}"
        return mlflow.pyfunc.load_model(model_uri)

    def register_and_promote(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
    ) -> str:
        mlflow.set_tracking_uri(self._tracking_uri)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline_path = Path(tmp_dir) / "pipeline.joblib"
            joblib.dump(model, pipeline_path)

            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=_SklearnPipelineModel(),
                    artifacts={"pipeline": str(pipeline_path)},
                )
                run_id = run.info.run_id

        model_version = mlflow.register_model(f"runs:/{run_id}/model", self._model_name)

        client = MlflowClient(tracking_uri=self._tracking_uri)
        client.set_registered_model_alias(
            self._model_name, self._alias, model_version.version
        )
        return str(model_version.version)
