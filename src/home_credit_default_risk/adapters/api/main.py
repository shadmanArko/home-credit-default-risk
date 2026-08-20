"""`HC-M4-09` — FastAPI composition root.

Builds the concrete adapters (`LocalFeatureStore`, `MlflowModelRegistry`)
once at startup, wires them into `ScoreApplicantUseCase`, and exposes it
as a FastAPI dependency (`get_use_case`) rather than a module-level
global -- this is what lets tests substitute a fake use case via
`app.dependency_overrides` without touching the real feature store or
MLflow registry (see `tests/test_api.py`).

This module owns zero business logic: it validates the request shape,
calls the use case, translates its result/errors into an HTTP response.
Every rule about *how* an applicant gets scored lives in
`application/score.py` and `domain/scoring.py`.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from home_credit_default_risk import config
from home_credit_default_risk.adapters.local_store import LocalFeatureStore
from home_credit_default_risk.adapters.mlflow_registry import MlflowModelRegistry
from home_credit_default_risk.application.score import (
    ScoreApplicantUseCase,
    UnknownApplicantError,
)
from home_credit_default_risk.application.score_new_applicant import (
    CREDIT_STANDING_TO_EXT_SOURCE,
    NewApplicantInput,
    ScoreNewApplicantUseCase,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("home_credit_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Loading feature store from %s", config.FEATURE_STORE_PATH)
    feature_store = LocalFeatureStore(config.FEATURE_STORE_PATH)

    logger.info("Connecting to model registry at %s", config.MLFLOW_TRACKING_URI)
    model_registry = MlflowModelRegistry(
        tracking_uri=config.MLFLOW_TRACKING_URI,
        model_name=config.MLFLOW_MODEL_NAME,
        alias=config.MLFLOW_PRODUCTION_ALIAS,
    )

    app.state.use_case = ScoreApplicantUseCase(feature_store, model_registry)
    app.state.new_applicant_use_case = ScoreNewApplicantUseCase(
        feature_store, model_registry
    )
    logger.info("Startup complete")
    yield


app = FastAPI(title="Home Credit Default Risk Scoring API", lifespan=lifespan)


def get_use_case(request: Request) -> ScoreApplicantUseCase:
    return request.app.state.use_case


def get_new_applicant_use_case(request: Request) -> ScoreNewApplicantUseCase:
    return request.app.state.new_applicant_use_case


class ScoreRequest(BaseModel):
    sk_id_curr: int


class ScoreResponse(BaseModel):
    sk_id_curr: int
    probability: float
    is_high_risk: bool
    threshold: float


class ApplyRequest(BaseModel):
    """`HC-M4-22` — the demo form's fields. `Literal` types on every
    dropdown give real schema validation against this dataset's actual
    category values (`docs/`-verified, not guessed) instead of accepting
    arbitrary strings that would silently fall into the model's
    "unknown category" bucket.
    """

    amt_income_total: float
    amt_credit: float
    amt_annuity: float
    amt_goods_price: float
    age_years: float
    is_employed: bool
    years_employed: float | None = None
    cnt_children: int
    code_gender: Literal["F", "M"]
    name_education_type: Literal[
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
    name_family_status: Literal[
        "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
    ]
    name_housing_type: Literal[
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ]
    flag_own_car: Literal["Y", "N"]
    flag_own_realty: Literal["Y", "N"]
    credit_standing: Literal[tuple(CREDIT_STANDING_TO_EXT_SOURCE)]

    @model_validator(mode="after")
    def years_employed_required_if_employed(self) -> "ApplyRequest":
        if self.is_employed and self.years_employed is None:
            raise ValueError("years_employed is required when is_employed is true")
        return self


class ApplyResponse(BaseModel):
    probability: float
    is_high_risk: bool
    threshold: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(
    payload: ScoreRequest,
    use_case: ScoreApplicantUseCase = Depends(get_use_case),  # noqa: B008
) -> ScoreResponse:
    logger.info("Scoring request for SK_ID_CURR=%s", payload.sk_id_curr)
    try:
        decision = use_case.score(payload.sk_id_curr)
    except UnknownApplicantError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ScoreResponse(
        sk_id_curr=payload.sk_id_curr,
        probability=decision.probability,
        is_high_risk=decision.is_high_risk,
        threshold=decision.threshold,
    )


@app.post("/apply", response_model=ApplyResponse)
def apply(
    payload: ApplyRequest,
    use_case: ScoreNewApplicantUseCase = Depends(get_new_applicant_use_case),  # noqa: B008
) -> ApplyResponse:
    logger.info("New-applicant scoring request")
    applicant = NewApplicantInput(
        amt_income_total=payload.amt_income_total,
        amt_credit=payload.amt_credit,
        amt_annuity=payload.amt_annuity,
        amt_goods_price=payload.amt_goods_price,
        age_years=payload.age_years,
        is_employed=payload.is_employed,
        years_employed=payload.years_employed,
        cnt_children=payload.cnt_children,
        code_gender=payload.code_gender,
        name_education_type=payload.name_education_type,
        name_family_status=payload.name_family_status,
        name_housing_type=payload.name_housing_type,
        flag_own_car=payload.flag_own_car,
        flag_own_realty=payload.flag_own_realty,
        credit_standing=payload.credit_standing,
    )
    decision = use_case.score(applicant)

    return ApplyResponse(
        probability=decision.probability,
        is_high_risk=decision.is_high_risk,
        threshold=decision.threshold,
    )


# Mounted last, deliberately -- explicit routes above always take
# precedence over this catch-all. Serves HC-M4-23's demo form at "/"
# with zero CORS configuration, since it's served by the same app it
# calls.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
