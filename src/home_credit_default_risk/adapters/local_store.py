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

from home_credit_default_risk.aggregations import COUNT_COLUMNS
from home_credit_default_risk.domain.ports import FeatureStore


class LocalFeatureStore(FeatureStore):
    def __init__(self, parquet_path: Path) -> None:
        features = pd.read_parquet(parquet_path)
        # Indexing is cheap; eagerly converting all rows to a dict-of-dicts
        # is not -- `to_dict(orient="index")` on the real ~356k-row store
        # measured at ~10.5s, most of a Lambda cold start's time budget for
        # a store that only ever needs one row per request. Row lookup is
        # deferred to `get_online_features` instead.
        self._features = features.set_index("SK_ID_CURR")
        self._default_features = {
            col: 0 if col in COUNT_COLUMNS else float("nan")
            for col in features.columns
            if col != "SK_ID_CURR"
        }

    def get_online_features(self, sk_id_curr: int) -> dict:
        try:
            return self._features.loc[sk_id_curr].to_dict()
        except KeyError:
            raise KeyError(
                f"No materialized features for SK_ID_CURR={sk_id_curr}. "
                "Re-run scripts/materialize_features.py if this applicant "
                "should be covered."
            ) from None

    def get_default_features(self) -> dict:
        return dict(self._default_features)
