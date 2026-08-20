import pytest

from home_credit_default_risk.application.score import (
    ScoreApplicantUseCase,
    UnknownApplicantError,
)
from home_credit_default_risk.domain.ports import FeatureStore, ModelRegistry


class FakeFeatureStore(FeatureStore):
    """Satisfies the port with an in-memory dict -- no Parquet, no I/O."""

    def __init__(self, data: dict):
        self._data = data

    def get_online_features(self, sk_id_curr):
        return self._data[sk_id_curr]

    def get_default_features(self):
        raise NotImplementedError("not exercised by ScoreApplicantUseCase")


class FakeModel:
    """Stands in for a loaded MLflow `pyfunc` model. Records its input so
    tests can assert on what the use case actually fed it."""

    def __init__(self, probability):
        self._probability = probability
        self.last_input = None

    def predict(self, model_input):
        self.last_input = model_input
        return [self._probability] * len(model_input)


class FakeModelRegistry(ModelRegistry):
    def __init__(self, model):
        self._model = model

    def get_production_model(self):
        return self._model

    def register_and_promote(self, model, metrics, params):
        raise NotImplementedError("not exercised by ScoreApplicantUseCase")


def make_stored_features():
    return {
        "AMT_CREDIT": 500000.0,
        "AMT_INCOME_TOTAL": 200000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_EMPLOYED": -1000,
        "DAYS_BIRTH": -12000,
    }


def test_score_returns_a_decision_from_the_registered_model():
    feature_store = FakeFeatureStore({100001: make_stored_features()})
    model_registry = FakeModelRegistry(FakeModel(probability=0.9))
    use_case = ScoreApplicantUseCase(feature_store, model_registry)

    decision = use_case.score(100001)

    assert decision.probability == 0.9
    assert decision.is_high_risk is True


def test_score_applies_basic_feature_engineering_before_scoring():
    feature_store = FakeFeatureStore({100001: make_stored_features()})
    model = FakeModel(probability=0.1)
    model_registry = FakeModelRegistry(model)
    use_case = ScoreApplicantUseCase(feature_store, model_registry)

    use_case.score(100001)

    assert "credit_to_income" in model.last_input.columns
    assert "age_years" in model.last_input.columns


def test_score_raises_unknown_applicant_error_for_a_missing_id():
    feature_store = FakeFeatureStore({})
    model_registry = FakeModelRegistry(FakeModel(probability=0.5))
    use_case = ScoreApplicantUseCase(feature_store, model_registry)

    with pytest.raises(UnknownApplicantError):
        use_case.score(999999)
