from home_credit_default_risk import config


def test_random_state_is_fixed():
    assert config.RANDOM_STATE == 42


def test_holdout_fraction_matches_milestone_1_split():
    assert config.TEST_SIZE == 0.2


def test_cv_and_metric_are_defined():
    assert config.N_SPLITS == 5
    assert config.PRIMARY_METRIC == "roc_auc"


def test_paths_resolve_under_project_root():
    assert config.SPLIT_PATH == config.DATA_INTERIM_DIR / "train_valid_split.csv"
    assert config.DATA_RAW_DIR.parent == config.PROJECT_ROOT / "data"
