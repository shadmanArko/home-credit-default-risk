"""Stratified K-Fold cross-validation harness (`HC-M3-11`).

A single train/valid split (used throughout Milestone 1 and for the
`HC-M3-09` smoke test) gives one noisy ROC-AUC estimate per model — not
enough to trust a small difference between two candidates. This harness
scores a model across `config.N_SPLITS` stratified folds and reports
mean +/- std, which is what `HC-M3-08`, `HC-M3-12`-`15` actually compare
on.
"""

import time

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from home_credit_default_risk import config


def cross_validate_pipeline(
    pipeline_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    random_state: int | None = None,
) -> dict:
    """Stratified K-fold CV for a pipeline factory.

    `pipeline_factory` must be a zero-argument callable returning a
    *fresh, unfitted* `Pipeline` each call — reusing one fitted pipeline
    across folds would let fold N's fit leak into fold N+1's evaluation.

    Returns a dict with per-fold ROC-AUC/PR-AUC (`fold_scores`), their
    mean/std (`mean_roc_auc`, `std_roc_auc`, `mean_pr_auc`), and wall-clock
    `runtime_seconds` for the whole CV run — `HC-M3-13`/`14` record this
    alongside the metrics, not as an afterthought.
    """
    n_splits = n_splits if n_splits is not None else config.N_SPLITS
    random_state = random_state if random_state is not None else config.RANDOM_STATE

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    fold_rows = []
    start = time.time()
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline = pipeline_factory()
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_valid)[:, 1]

        fold_rows.append(
            {
                "fold": fold_index,
                "roc_auc": roc_auc_score(y_valid, proba),
                "pr_auc": average_precision_score(y_valid, proba),
            }
        )
    runtime_seconds = time.time() - start

    fold_scores = pd.DataFrame(fold_rows)
    return {
        "fold_scores": fold_scores,
        "mean_roc_auc": fold_scores["roc_auc"].mean(),
        "std_roc_auc": fold_scores["roc_auc"].std(),
        "mean_pr_auc": fold_scores["pr_auc"].mean(),
        "runtime_seconds": runtime_seconds,
    }
