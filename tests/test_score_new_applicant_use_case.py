from home_credit_default_risk.application.score_new_applicant import (
    NewApplicantInput,
    ScoreNewApplicantUseCase,
)
from home_credit_default_risk.domain.ports import FeatureStore, ModelRegistry
from home_credit_default_risk.features import DAYS_EMPLOYED_SENTINEL


class FakeFeatureStore(FeatureStore):
    def __init__(self, defaults: dict):
        self._defaults = defaults

    def get_online_features(self, sk_id_curr):
        raise NotImplementedError("not exercised by ScoreNewApplicantUseCase")

    def get_default_features(self):
        return dict(self._defaults)


class FakeModel:
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
        raise NotImplementedError("not exercised by ScoreNewApplicantUseCase")


def make_applicant(**overrides):
    defaults = {
        "amt_income_total": 100000.0,
        "amt_credit": 500000.0,
        "amt_annuity": 30000.0,
        "amt_goods_price": 450000.0,
        "age_years": 30.0,
        "is_employed": True,
        "years_employed": 3.0,
        "cnt_children": 1,
        "code_gender": "M",
        "name_education_type": "Secondary / secondary special",
        "name_family_status": "Married",
        "name_housing_type": "House / apartment",
        "flag_own_car": "N",
        "flag_own_realty": "N",
        "credit_standing": "fair",
    }
    defaults.update(overrides)
    return NewApplicantInput(**defaults)


def test_score_merges_defaults_with_the_applicants_answers():
    feature_store = FakeFeatureStore(
        {"bureau_credit_count": 0, "prev_refused_rate": float("nan")}
    )
    model = FakeModel(probability=0.5)
    use_case = ScoreNewApplicantUseCase(feature_store, FakeModelRegistry(model))

    use_case.score(make_applicant())

    fed_row = model.last_input.iloc[0]
    assert fed_row["bureau_credit_count"] == 0
    assert fed_row["AMT_INCOME_TOTAL"] == 100000.0


def test_score_applies_basic_feature_engineering():
    feature_store = FakeFeatureStore({})
    model = FakeModel(probability=0.5)
    use_case = ScoreNewApplicantUseCase(feature_store, FakeModelRegistry(model))

    use_case.score(make_applicant())

    assert "credit_to_income" in model.last_input.columns
    assert "age_years" in model.last_input.columns


def test_employed_applicant_converts_years_employed_to_negative_days():
    feature_store = FakeFeatureStore({})
    model = FakeModel(probability=0.5)
    use_case = ScoreNewApplicantUseCase(feature_store, FakeModelRegistry(model))

    use_case.score(make_applicant(is_employed=True, years_employed=10.0))

    fed_row = model.last_input.iloc[0]
    assert fed_row["DAYS_EMPLOYED"] == -10.0 * 365.25


def test_unemployed_applicant_gets_the_days_employed_sentinel():
    feature_store = FakeFeatureStore({})
    model = FakeModel(probability=0.5)
    use_case = ScoreNewApplicantUseCase(feature_store, FakeModelRegistry(model))

    use_case.score(make_applicant(is_employed=False, years_employed=None))

    fed_row = model.last_input.iloc[0]
    assert fed_row["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    # is_employment_unknown is HC-M3-05's own signal for this sentinel --
    # confirms add_basic_features() actually ran on the merged row.
    assert fed_row["is_employment_unknown"] == 1


def test_credit_standing_maps_to_all_three_ext_source_columns():
    feature_store = FakeFeatureStore({})
    model = FakeModel(probability=0.5)
    use_case = ScoreNewApplicantUseCase(feature_store, FakeModelRegistry(model))

    use_case.score(make_applicant(credit_standing="excellent"))

    fed_row = model.last_input.iloc[0]
    assert fed_row["EXT_SOURCE_1"] == fed_row["EXT_SOURCE_2"] == fed_row["EXT_SOURCE_3"]
    assert fed_row["EXT_SOURCE_1"] > 0.5


def test_score_returns_the_models_probability_as_a_decision():
    feature_store = FakeFeatureStore({})
    model = FakeModel(probability=0.9)
    use_case = ScoreNewApplicantUseCase(feature_store, FakeModelRegistry(model))

    decision = use_case.score(make_applicant())

    assert decision.probability == 0.9
    assert decision.is_high_risk is True
