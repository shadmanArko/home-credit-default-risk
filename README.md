# Home Credit Default Risk

Machine learning project for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition.

The goal is to develop a reproducible machine learning workflow for predicting the probability that a loan applicant will default.

## Project Status

🚧 Project setup in progress.

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
- [ ] Data validation
- [ ] Feature engineering
- [ ] Baseline model
- [ ] Model evaluation
- [ ] Model comparison
- [ ] Final Kaggle submission

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

## Project Structure

```text
home-credit-default-risk/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│
├── notebooks/
│
├── src/
│   └── home_credit_default_risk/
│
├── scripts/
│
├── tests/
│
├── models/
│
├── reports/
│
├── .github/
│   └── workflows/
│
├── pyproject.toml
├── uv.lock
└── README.md