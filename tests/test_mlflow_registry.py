import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from home_credit_default_risk.adapters.mlflow_registry import MlflowModelRegistry


def make_synthetic_pipeline_and_data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(50, 3)
    y = (X[:, 0] > 0.5).astype(int)
    pipeline = Pipeline(steps=[("classify", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline, X


def make_registry(tmp_path, model_name="test_model"):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return MlflowModelRegistry(
        tracking_uri=tracking_uri, model_name=model_name, alias="production"
    )


def test_register_and_promote_returns_a_version_string(tmp_path):
    registry = make_registry(tmp_path)
    pipeline, _ = make_synthetic_pipeline_and_data()

    version = registry.register_and_promote(
        pipeline, metrics={"roc_auc": 0.77}, params={"random_state": 42}
    )

    assert version == "1"


def test_get_production_model_returns_probabilities_in_valid_range(tmp_path):
    registry = make_registry(tmp_path)
    pipeline, X = make_synthetic_pipeline_and_data()
    registry.register_and_promote(
        pipeline, metrics={"roc_auc": 0.77}, params={"random_state": 42}
    )

    production_model = registry.get_production_model()
    proba = production_model.predict(X)

    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_get_production_model_matches_the_original_pipelines_predictions(tmp_path):
    registry = make_registry(tmp_path)
    pipeline, X = make_synthetic_pipeline_and_data()
    registry.register_and_promote(
        pipeline, metrics={"roc_auc": 0.77}, params={"random_state": 42}
    )

    production_model = registry.get_production_model()
    expected = pipeline.predict_proba(X)[:, 1]

    np.testing.assert_allclose(production_model.predict(X), expected)


def test_re_registering_moves_the_production_alias_to_the_new_version(tmp_path):
    registry = make_registry(tmp_path, model_name="reregister_test")
    pipeline_v1, X = make_synthetic_pipeline_and_data(seed=0)
    pipeline_v2, _ = make_synthetic_pipeline_and_data(seed=1)

    version_1 = registry.register_and_promote(
        pipeline_v1, metrics={"roc_auc": 0.70}, params={}
    )
    version_2 = registry.register_and_promote(
        pipeline_v2, metrics={"roc_auc": 0.80}, params={}
    )

    assert version_1 == "1"
    assert version_2 == "2"

    production_model = registry.get_production_model()
    expected = pipeline_v2.predict_proba(X)[:, 1]
    np.testing.assert_allclose(production_model.predict(X), expected)
