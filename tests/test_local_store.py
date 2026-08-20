import pandas as pd
import pytest

from home_credit_default_risk.adapters.local_store import LocalFeatureStore


def make_store(tmp_path):
    features = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "bureau_credit_count": [3, 0],
            "prev_refused_rate": [0.5, float("nan")],
        }
    )
    path = tmp_path / "historical_features.parquet"
    features.to_parquet(path, index=False)
    return LocalFeatureStore(path)


def test_get_online_features_returns_the_materialized_row(tmp_path):
    store = make_store(tmp_path)

    features = store.get_online_features(100001)

    assert features["bureau_credit_count"] == 3
    assert features["prev_refused_rate"] == 0.5


def test_get_online_features_preserves_a_true_missing_history_as_nan(tmp_path):
    store = make_store(tmp_path)

    features = store.get_online_features(100002)

    assert features["bureau_credit_count"] == 0
    assert pd.isna(features["prev_refused_rate"])


def test_get_online_features_raises_key_error_for_an_unknown_applicant(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(KeyError):
        store.get_online_features(999999)
