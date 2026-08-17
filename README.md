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
- [ ] Data validation
- [ ] Exploratory data analysis
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