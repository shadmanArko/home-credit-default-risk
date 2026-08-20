"""API-level tests for `HC-M4-09`.

Deliberately never enters `TestClient` as a context manager, which is
what would trigger the real `lifespan()` (real `LocalFeatureStore` +
real MLflow connection) -- these tests only exercise the route logic,
with `get_use_case` overridden to a fake. No materialized feature store
or MLflow tracking server is required to run this file.
"""

from fastapi.testclient import TestClient

from home_credit_default_risk.adapters.api.main import app, get_use_case
from home_credit_default_risk.application.score import UnknownApplicantError
from home_credit_default_risk.domain.scoring import decide


class FakeUseCase:
    def __init__(self, decision=None, error=None):
        self._decision = decision
        self._error = error

    def score(self, sk_id_curr):
        if self._error is not None:
            raise self._error
        return self._decision


def make_client(fake_use_case):
    app.dependency_overrides[get_use_case] = lambda: fake_use_case
    return TestClient(app)


def test_health_returns_ok():
    client = make_client(FakeUseCase())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    app.dependency_overrides.clear()


def test_score_returns_the_use_cases_decision():
    decision = decide(0.9, threshold=0.485)
    client = make_client(FakeUseCase(decision=decision))

    response = client.post("/score", json={"sk_id_curr": 100001})

    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] == 100001
    assert body["probability"] == 0.9
    assert body["is_high_risk"] is True
    assert body["threshold"] == 0.485
    app.dependency_overrides.clear()


def test_score_returns_404_for_an_unknown_applicant():
    client = make_client(FakeUseCase(error=UnknownApplicantError("no such applicant")))

    response = client.post("/score", json={"sk_id_curr": 999999})

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_score_rejects_a_non_integer_sk_id_curr():
    client = make_client(FakeUseCase())

    response = client.post("/score", json={"sk_id_curr": "not-an-id"})

    assert response.status_code == 422
    app.dependency_overrides.clear()
