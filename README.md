# Home Credit Default Risk

Machine learning project for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition.

The goal is to develop a reproducible machine learning workflow for predicting the probability that a loan applicant will default.

## Project Status

✅ Milestone 1 (data understanding + baseline modeling) complete. 🚧 Feature engineering next.

Current stage:

- [x] Python environment
- [x] Dependency management with uv
- [x] Project structure
- [x] Package configuration
- [x] Linting with Ruff
- [x] Automated testing with pytest
- [x] Kaggle data pipeline
- [x] Data profiling & dictionary
- [x] Exploratory data analysis
- [x] Baseline model
- [x] Model evaluation
- [x] Model comparison
- [ ] Data validation
- [ ] Feature engineering
- [ ] Final Kaggle submission

## Milestone 1 Summary

| | |
|---|---|
| **Problem** | Predict `TARGET` — whether a loan applicant will have payment difficulty on the loan — from application-time data. Full framing, cost asymmetry, and metric justification: [`docs/problem_definition.md`](docs/problem_definition.md). |
| **Target** | Binary, ~11.4:1 imbalanced (~8.07% positive). Only present in `application_train`. |
| **Metric** | ROC-AUC — chosen over accuracy for the imbalance, and over other classification metrics because this stage optimizes ranking, not a final approve/reject threshold (see `docs/problem_definition.md` §4–5). |
| **Data understanding** | 9 tables profiled end-to-end with an out-of-core DuckDB pipeline (scales past what fits in memory) — [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb), [`reports/data_dictionary.md`](reports/data_dictionary.md), [`reports/data_quality_summary.md`](reports/data_quality_summary.md). |
| **Baseline modeling** | Reproducible stratified split + dummy baseline + `Pipeline`-wrapped logistic regression — [`notebooks/02_baseline_model.ipynb`](notebooks/02_baseline_model.ipynb), results in [`reports/experiments.csv`](reports/experiments.csv). |
| **Headline result** | **ROC-AUC 0.7489** (logistic regression, `application` features only) vs. **0.5000** (dummy floor) — the number future feature engineering and model upgrades have to beat. |
| **Reproducibility** | Verified from a full clean-room rebuild, not assumed — [`reports/reproducibility_check.md`](reports/reproducibility_check.md). |
| **Known limitations** | No cross-validation or hyperparameter tuning yet (deliberately deferred until a feature set beyond a single linear baseline exists to tune — see the notebook's own reasoning); `DAYS_EMPLOYED` sentinel and `bureau_balance`/`SK_ID_PREV` orphan rates identified but not yet fixed (feature engineering scope). |

## Milestone 3 Progress (in progress — feature engineering + model selection)

| | |
|---|---|
| **Feature engineering** | Ratio/age features + historical aggregations from `bureau`, `previous_application`, and payment-behavior tables — [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb), strategy in [`docs/feature_engineering_strategy.md`](docs/feature_engineering_strategy.md), leakage audit in [`docs/feature_leakage_audit.md`](docs/feature_leakage_audit.md). |
| **Reusable pipeline code** | Feature engineering, preprocessing, and CV moved into a real importable package — `src/home_credit_default_risk/{features,aggregations,pipeline,cv}.py` — with 39 unit tests, not left as notebook-only logic. |
| **Feature-set decision** | Engineered features beat application-only baseline by **+0.0106 ROC-AUC** under 5-fold CV, holding the model fixed (logistic regression) — [`notebooks/04_modeling.ipynb`](notebooks/04_modeling.ipynb). |
| **Best candidate** | **LightGBM, ROC-AUC 0.7747 ± 0.0020** (5-fold CV) on engineered features — beats XGBoost (0.7579, same features/CV/setup), logistic regression on the same features (0.7550), and the Milestone 1 baseline (0.7489). |
| **Hyperparameter tuning result** | `RandomizedSearchCV` (30 candidates × 5-fold CV) found a configuration scoring **+0.0032 CV ROC-AUC** over the untuned defaults — but with **more than double the train-vs-CV overfitting gap** (~0.0995 vs. ~0.0418). **Tuning was rejected**: the untuned LightGBM defaults are the final model. A genuine "don't tune just to say you tuned" result, not a foregone conclusion — see `notebooks/04_modeling.ipynb`, `HC-M3-18`. |
| **Final holdout result** | **ROC-AUC 0.7791** on the holdout set (61,503 rows, evaluated exactly once) — **+0.0044 above** the dev-pool CV estimate, the favorable direction, confirming no overfitting. PR-AUC 0.2758. See `notebooks/04_modeling.ipynb`, `HC-M3-19`–`21`. |
| **Error analysis** | False positive/negative characteristics (`HC-M3-23`/`24`), a leakage-safe threshold analysis on out-of-fold predictions choosing **0.485** against a 70% recall floor (`HC-M3-25`), gain-based feature importance (`EXT_SOURCE_*` ≈ 48% of total gain, `HC-M3-26`), and documented limitations including a measured recall gap by `CODE_GENDER` (`HC-M3-27`) — see `notebooks/04_modeling.ipynb`. |
| **Experiment summary** | 8 experiments recorded end-to-end (2 Milestone 1 baselines + 6 Milestone 3 candidates, including the rejected tuned configuration) — [`reports/experiments.csv`](reports/experiments.csv), `HC-M3-28`. |
| **What's next** | `HC-M3-29` — Milestone 3 technical review (Definition of Done checklist), closing Milestone 3. |

## Tech Stack

- Python 3.12
- uv
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- JupyterLab
- KaggleHub
- DuckDB
- Ruff
- pytest

## Data Profiling

The raw dataset is ~2.9GB across 8 tables and will only grow in future
projects, so it is never loaded into pandas wholesale just to answer "what
does this data look like." Instead:

1. `uv run python scripts/profile_data.py` lands each raw CSV once into a
   local DuckDB table (a single text parse) and computes full-table
   aggregates — row/column counts, per-column null rate, cardinality,
   numeric stats, top categorical values, primary-key uniqueness, and
   foreign-key orphan rates against the `SK_ID_CURR` spine — using
   DuckDB's out-of-core vectorized engine, which never requires the full
   file to fit in memory. Output: one JSON per table in
   `reports/data_profile/`.
2. `uv run python scripts/build_data_dictionary.py` merges those JSON
   profiles with the official `HomeCredit_columns_description.csv` into a
   single `reports/data_dictionary.md` — a compact, versioned reference
   with real statistics and business descriptions side by side.
3. `uv run python scripts/generate_data_quality_summary.py` consolidates
   every table's full-row duplicate count and every declared foreign-key
   relationship's orphan rate — not just single-column FK-to-spine checks,
   but arbitrary table-to-table relationships (e.g. `bureau_balance ->
   bureau`, `POS_CASH_balance.SK_ID_PREV -> previous_application`) — into
   one scorecard, `reports/data_quality_summary.md`. This is the answer to
   "how much duplicate/bad data is in all my tables": one file, one row
   per table, no digging through notebook cells.

Both the profile JSONs and the data dictionary are committed to git. They
are the reference for understanding the dataset going forward — for
developers and for AI assistants working on this repo — rather than
re-reading or re-summarizing the raw CSVs each time. This is the same
pattern real data platforms use for data catalogs (compute stats once with
a scalable engine, store as metadata, read the metadata everywhere else),
and it scales unchanged from this dataset's ~1GB tables to files far
larger than available RAM.

### Visual EDA report (Sweetviz)

`uv run python scripts/generate_eda_report.py` generates
`reports/eda_train_vs_test_compare.html`, a Sweetviz side-by-side
comparison of `application_train` vs. `application_test` — per-column
distributions, missingness, and cardinality plotted for both sets at once.

Two deliberate scoping decisions, not defaults:

- **`ydata-profiling` was evaluated and rejected**: its pins (`pandas
  <=1.1` / `==1.4.0`) conflict with this project's `pandas>=3.0.5` and
  there's no way to satisfy both — documented here instead of silently
  worked around.
- Only `application_train`/`application_test` are profiled this way
  (307k / 49k rows fit safely in pandas); the report intentionally
  **excludes `TARGET`** and skips pairwise feature associations. Per this
  project's EDA discipline (see `notebooks/01_data_understanding.ipynb`),
  feature-vs-target analysis is deferred until after the stratified
  train/valid split and computed on the training fold only — this report
  is train-vs-test drift only, not target correlation.

The generated HTML (~8MB, mostly embedded plots) is not committed to git;
regenerate it locally when needed.

## Baseline Modeling & Experiment Tracking

`notebooks/02_baseline_model.ipynb` builds a reproducible stratified
train/validation split (`random_state=42`, saved to
`data/interim/train_valid_split.csv`), then a dummy baseline and a
`Pipeline`-wrapped logistic regression baseline on `application` features.
Every result is appended to `reports/experiments.csv` — the durable,
cross-notebook experiment log every future model has to beat:

| id | model | features | roc_auc |
|---|---|---|---|
| B0 | DummyClassifier(strategy='prior') | none | 0.5000 |
| B1 | LogisticRegression(class_weight='balanced') | application | 0.7489 |

The write is an upsert by `id`, not an overwrite — a teammate's or a later
milestone's experiment rows survive regardless of which notebook runs last.
See `reports/reproducibility_check.md` for a from-scratch verification that
these numbers reproduce exactly, plus a documented, immaterial exception
(parallel floating-point `AVG()` in the data profiler).

## Project Structure

```text
home-credit-default-risk/
│
├── data/
│   ├── raw/                          # gitignored — download_data.py fetches it
│   ├── interim/                      # train_valid_split.csv (regenerable)
│   └── processed/
│
├── docs/
│   └── problem_definition.md         # HC-M1-01
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # HC-M1-02/03/04
│   └── 02_baseline_model.ipynb       # HC-M1-05..10
│
├── src/
│   └── home_credit_default_risk/
│
├── scripts/
│   ├── download_data.py
│   ├── profile_data.py
│   ├── build_data_dictionary.py
│   ├── generate_data_quality_summary.py
│   └── generate_eda_report.py
│
├── tests/
│
├── models/
│
├── reports/
│   ├── data_profile/*.json           # gitignored cache excluded, JSONs committed
│   ├── data_dictionary.md
│   ├── data_quality_summary.md
│   ├── reproducibility_check.md
│   └── experiments.csv
│
├── .github/
│   └── workflows/
│
├── pyproject.toml
├── uv.lock
└── README.md
```