"""`HC-M4-08` — `ScoreApplicantUseCase`.

Orchestrates the `FeatureStore` and `ModelRegistry` ports to score one
applicant. This is the *only* place in this codebase that knows the
model needs a one-row DataFrame, that `add_basic_features()` must run
before `predict()`, and that `predict()` already returns the probability
of `TARGET = 1` -- every entrypoint (FastAPI, the batch CLI, and later a
Lambda handler) calls this and nothing else, per this project's DRY
discipline.
"""

import pandas as pd

from home_credit_default_risk.domain.ports import FeatureStore, ModelRegistry
from home_credit_default_risk.domain.scoring import Decision, decide
from home_credit_default_risk.features import add_basic_features


class UnknownApplicantError(Exception):
    """`sk_id_curr` has no materialized features in the feature store."""


class ScoreApplicantUseCase:
    def __init__(
        self, feature_store: FeatureStore, model_registry: ModelRegistry
    ) -> None:
        self._feature_store = feature_store
        self._model_registry = model_registry

    def score(self, sk_id_curr: int) -> Decision:
        try:
            stored_features = self._feature_store.get_online_features(sk_id_curr)
        except KeyError as exc:
            raise UnknownApplicantError(str(exc)) from exc

        row = pd.DataFrame([stored_features])
        row = add_basic_features(row)

        model = self._model_registry.get_production_model()
        probability = float(model.predict(row)[0])

        return decide(probability)
