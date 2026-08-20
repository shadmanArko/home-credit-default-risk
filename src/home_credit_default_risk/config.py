"""Centralized experiment configuration for Milestone 3 modeling.

Single source of truth for the random seed, holdout strategy, CV strategy,
and evaluation metric — imported by every notebook and script from
`HC-M3-02` onward, rather than each one hardcoding its own copy that can
silently drift out of sync with the others.

`RANDOM_STATE` and `TEST_SIZE` intentionally match the split already
created and reproducibility-verified in Milestone 1
(`data/interim/train_valid_split.csv`, `reports/reproducibility_check.md`)
— Milestone 3 reuses that split as its dev/holdout boundary rather than
drawing a new one, so there is exactly one holdout definition across the
whole project.
"""

from pathlib import Path

# Holdout split (matches the Milestone 1 split, reused as-is for M3)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cross-validation, applied within the development portion of the split
N_SPLITS = 5

# The one metric any model selection or hyperparameter search optimizes
# against — see docs/problem_definition.md §7 for why, and for the
# supporting metrics that are diagnostic-only and never the search target.
PRIMARY_METRIC = "roc_auc"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DB = REPORTS_DIR / "data_profile" / ".landing_cache.duckdb"
SPLIT_PATH = DATA_INTERIM_DIR / "train_valid_split.csv"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"

# Milestone 4 (`HC-M4-*`) — model registry. SQLite backend (not the
# default local file store) because the MLflow Model Registry's alias
# API requires a database-backed tracking store, not just run logging.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_MODEL_NAME = "home_credit_lightgbm"
MLFLOW_PRODUCTION_ALIAS = "production"

# Milestone 4 (`HC-M4-04`..`07`) — feature store. A materialized Parquet
# file (offline store) that `LocalFeatureStore` loads for fast online
# lookups -- Feast was evaluated and rejected (every released version
# requires `pandas<3`, the same class of conflict as `ydata-profiling`
# and the full `mlflow` package; unlike MLflow, Feast has no "skinny"
# escape hatch since pandas is load-bearing in its core, not optional).
FEATURE_STORE_PATH = DATA_PROCESSED_DIR / "historical_features.parquet"
