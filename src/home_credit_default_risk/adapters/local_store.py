"""`HC-M4-05` — local implementation of the `FeatureStore` port.

Feast (the originally planned adapter) was evaluated and rejected: every
released version (`0.20` through `0.65`, the full available range)
requires `pandas<3`, incompatible with this project's `pandas>=3.0.5` --
and unlike MLflow, there is no "skinny" Feast without pandas, since
pandas is used throughout its offline-store code, not just an optional
extra. `LocalFeatureStore` satisfies the exact same `FeatureStore` port
(Liskov Substitution -- nothing in `application/` will need to change
if a Feast-compatible pandas version ever ships) using a materialized
Parquet file (`scripts/materialize_features.py`) loaded once and indexed
in memory for O(1) point lookups -- the same "precompute once, look up
fast" shape a managed feature store's online store provides, without a
third-party service.
"""

from pathlib import Path

import pandas as pd

from home_credit_default_risk.domain.ports import FeatureStore


class LocalFeatureStore(FeatureStore):
    def __init__(self, parquet_path: Path) -> None:
        features = pd.read_parquet(parquet_path)
        self._features_by_id = features.set_index("SK_ID_CURR").to_dict(
            orient="index"
        )

    def get_online_features(self, sk_id_curr: int) -> dict:
        try:
            return self._features_by_id[sk_id_curr]
        except KeyError:
            raise KeyError(
                f"No materialized features for SK_ID_CURR={sk_id_curr}. "
                "Re-run scripts/materialize_features.py if this applicant "
                "should be covered."
            ) from None
