"""`HC-M4-21` — `ScoreNewApplicantUseCase`.

Scores a person who has never applied before -- no `SK_ID_CURR`, no
bureau/previous-loan history. The key move: a brand-new applicant's
missing history is represented *exactly* the way `aggregations.py`
already represents any real applicant with no history (count columns at
`0`, everything else `NaN`) via `FeatureStore.get_default_features()`
(`HC-M4-19`/`20`) -- no fake bureau call, no second notion of "missing."
The handful of fields the demo form actually collects are overlaid on
top of that template, `HC-M3-05`'s existing `add_basic_features()` runs
unchanged, and scoring proceeds through the same `ModelRegistry` port and
the same `domain/scoring.decide()` as `ScoreApplicantUseCase`.
"""

from dataclasses import dataclass

import pandas as pd

from home_credit_default_risk.domain.ports import FeatureStore, ModelRegistry
from home_credit_default_risk.domain.scoring import Decision, decide
from home_credit_default_risk.features import DAYS_EMPLOYED_SENTINEL, add_basic_features

_DAYS_PER_YEAR = 365.25

# EXT_SOURCE_1/2/3 (bureau-provided credit scores) are ~48% of the
# model's total decision weight (HC-M3-26) but are, by definition,
# unavailable for a brand-new applicant with no bureau record yet. Real
# EXT_SOURCE_1/2/3 values range ~0-0.96 with medians ~0.5-0.57
# (reports/data_profile/application_train.json) -- these four buckets
# are honest, roughly quartile-spaced representative values, not a
# precise mapping to any real scoring scale. Without this, every new
# applicant's three strongest features would be identically "unknown"
# regardless of input, and the demo could never show a high-risk result.
CREDIT_STANDING_TO_EXT_SOURCE = {
    "poor": 0.15,
    "fair": 0.35,
    "good": 0.65,
    "excellent": 0.85,
}


@dataclass(frozen=True)
class NewApplicantInput:
    """The minimal field set confirmed for the demo form -- everything
    else is left to `get_default_features()`'s "no data" defaults."""

    amt_income_total: float
    amt_credit: float
    amt_annuity: float
    amt_goods_price: float
    age_years: float
    is_employed: bool
    years_employed: float | None
    cnt_children: int
    code_gender: str
    name_education_type: str
    name_family_status: str
    name_housing_type: str
    flag_own_car: str
    flag_own_realty: str
    credit_standing: str

    def to_raw_fields(self) -> dict:
        days_employed = (
            -self.years_employed * _DAYS_PER_YEAR
            if self.is_employed
            else DAYS_EMPLOYED_SENTINEL
        )
        ext_source = CREDIT_STANDING_TO_EXT_SOURCE[self.credit_standing]
        return {
            "AMT_INCOME_TOTAL": self.amt_income_total,
            "AMT_CREDIT": self.amt_credit,
            "AMT_ANNUITY": self.amt_annuity,
            "AMT_GOODS_PRICE": self.amt_goods_price,
            "DAYS_BIRTH": -self.age_years * _DAYS_PER_YEAR,
            "DAYS_EMPLOYED": days_employed,
            "CNT_CHILDREN": self.cnt_children,
            "CODE_GENDER": self.code_gender,
            "NAME_EDUCATION_TYPE": self.name_education_type,
            "NAME_FAMILY_STATUS": self.name_family_status,
            "NAME_HOUSING_TYPE": self.name_housing_type,
            "FLAG_OWN_CAR": self.flag_own_car,
            "FLAG_OWN_REALTY": self.flag_own_realty,
            "EXT_SOURCE_1": ext_source,
            "EXT_SOURCE_2": ext_source,
            "EXT_SOURCE_3": ext_source,
        }


class ScoreNewApplicantUseCase:
    def __init__(
        self, feature_store: FeatureStore, model_registry: ModelRegistry
    ) -> None:
        self._feature_store = feature_store
        self._model_registry = model_registry

    def score(self, applicant: NewApplicantInput) -> Decision:
        row = self._feature_store.get_default_features()
        row.update(applicant.to_raw_fields())

        frame = pd.DataFrame([row])
        frame = add_basic_features(frame)

        model = self._model_registry.get_production_model()
        probability = float(model.predict(frame)[0])

        return decide(probability)
