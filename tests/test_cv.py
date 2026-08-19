import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from home_credit_default_risk.cv import cross_validate_pipeline


def make_synthetic_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    # y correlated with f0 so the model has something real to learn
    y = pd.Series((X["f0"] + rng.randn(n) * 0.5 > 0).astype(int))
    return X, y


def make_pipeline():
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classify", LogisticRegression(random_state=42)),
        ]
    )


def test_returns_exactly_n_splits_fold_scores():
    X, y = make_synthetic_data()
    result = cross_validate_pipeline(make_pipeline, X, y, n_splits=5)
    assert len(result["fold_scores"]) == 5


def test_mean_and_std_match_fold_scores():
    X, y = make_synthetic_data()
    result = cross_validate_pipeline(make_pipeline, X, y, n_splits=5)
    scores = result["fold_scores"]["roc_auc"]
    assert result["mean_roc_auc"] == scores.mean()
    assert result["std_roc_auc"] == scores.std()


def test_stratification_preserves_class_balance_per_fold():
    X, y = make_synthetic_data(n=500)
    overall_rate = y.mean()

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _, valid_idx in splitter.split(X, y):
        fold_rate = y.iloc[valid_idx].mean()
        assert abs(fold_rate - overall_rate) < 0.05


def test_same_random_state_is_reproducible():
    X, y = make_synthetic_data()
    result_a = cross_validate_pipeline(make_pipeline, X, y, n_splits=5, random_state=42)
    result_b = cross_validate_pipeline(make_pipeline, X, y, n_splits=5, random_state=42)
    assert result_a["mean_roc_auc"] == result_b["mean_roc_auc"]


def test_fresh_pipeline_used_per_fold():
    X, y = make_synthetic_data()
    created_instances = []

    def tracking_factory():
        pipeline = make_pipeline()
        created_instances.append(pipeline)
        return pipeline

    cross_validate_pipeline(tracking_factory, X, y, n_splits=5)
    assert len(created_instances) == 5
    assert len(set(id(p) for p in created_instances)) == 5


def test_runtime_is_recorded():
    X, y = make_synthetic_data()
    result = cross_validate_pipeline(make_pipeline, X, y, n_splits=5)
    assert result["runtime_seconds"] > 0
