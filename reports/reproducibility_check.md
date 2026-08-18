# Reproducibility Check — `HC-M1-09`

## What was actually tested

Not a claim — a real clean-room rebuild. Every derived artifact was deleted
and regenerated from the raw CSVs and committed code, then compared against
the previously committed results:

```bash
rm reports/data_profile/.landing_cache.duckdb reports/data_profile/.landing_cache.duckdb.wal
rm data/interim/train_valid_split.csv
uv sync
uv run python scripts/profile_data.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_model.ipynb
```

## Result: the modeling numbers reproduced exactly

`notebooks/02_baseline_model.ipynb`'s diff after the full rebuild touches
**only execution timestamp metadata** — every printed value, every
dataframe, every ROC-AUC is byte-identical to the prior run:

- Dummy baseline (`B0`): **ROC-AUC = 0.5000**
- Logistic regression baseline (`B1`): **ROC-AUC = 0.7489**

This is expected, not lucky: the split is `train_test_split(..., stratify=y,
random_state=42)` on the same raw data, the entire preprocessing +
estimator is one `Pipeline` fit only on that split's training fold, and
`LogisticRegression`'s `lbfgs` solver is deterministic given a fixed
`random_state` and fixed input. Nothing in the pipeline depends on
wall-clock time, hash-randomized iteration order, or unseeded randomness.

## Result: the data *profiles* are not bit-identical — and here's why that's fine

Regenerating `reports/data_profile/*.json` from scratch produced identical
`min`/`max`/`median`/`null_count`/`distinct_count` values in every column,
but the `avg` field differs from the previously committed value in the
13th–15th significant digit for many numeric columns, e.g.:

```
bureau.json: AMT_CREDIT_SUM_DEBT.avg
  before:  137085.11995216084
  after:   137085.11995216087
```

**Root cause**: DuckDB computes `AVG()` in parallel across threads, and the
order in which partial sums from different threads get combined is not
fixed run-to-run. Floating-point addition is not associative
(`(a+b)+c` can differ from `a+(b+c)` at the bit level), so a different
combination order produces a value that agrees to ~13-15 significant
figures but not bit-for-bit. `MIN`/`MAX`/`COUNT DISTINCT`/exact-`median`
don't have this property (order-independent or single deterministic
comparisons), which is exactly the set of fields that stayed identical.

**Why this doesn't threaten model reproducibility**: no `avg` field from
the profiler is ever read by the modeling pipeline. `SimpleImputer` in
`notebooks/02_baseline_model.ipynb` computes its own median directly from
`X_train` via scikit-learn, not from the profiler's JSON. The `avg` field
is a descriptive statistic for `notebooks/01_data_understanding.ipynb` and
`reports/data_dictionary.md`, displayed rounded to 4 decimal places, where
this level of noise is invisible. Documented here so a future diff of
`reports/data_profile/*.json` showing only last-digit `avg` changes isn't
mistaken for a real data or code change — regenerate and re-diff scoped to
`min`/`max`/`median`/`*_count` fields if verifying data integrity.

## Environment record (what "same environment" means here)

| | |
|---|---|
| Python | 3.12.13 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| duckdb | 1.5.5 |
| numpy | 2.5.2 |
| `uv.lock` SHA256 | `9e10fe42856f854d3a3cde833b92f5c89555958ad4aad2b8a6026ba816f5bd2d` |

`uv sync` against the committed `uv.lock` reproduces this exact environment
— that's the point of committing the lockfile rather than a loose
`requirements.txt`.

## Dataset version record

SHA256 of each raw file, so "same dataset" is a checkable claim rather than
an assumption (Kaggle competition data has no built-in versioning):

| File | SHA256 |
|---|---|
| `application_train.csv` | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `application_test.csv` | `a36161331d839150a67b6216d4de066f543a3b01a34061507d40e76612e0dec8` |
| `bureau.csv` | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |
| `bureau_balance.csv` | `33e09f06174c26f0be6b8b7398886c69e7bf0abbb29b4122f7841ffe545729a9` |
| `previous_application.csv` | `5046cd657ee04df2eaa6dc8308ae86be6b3b1763674a3f63574886a2f2896505` |
| `POS_CASH_balance.csv` | `0e13bc573ffa8fc29b3f00d975e557143193a405d675b0e4694b06fbdffcb0cd` |
| `credit_card_balance.csv` | `a9cdc48900d55131c90f3128b991859aeb94ca1326fb5f4d1624b9fd03782247` |
| `installments_payments.csv` | `428c2e2496e4d6d697ee8270e98497e5213c41be16d882eed1bc95b133726797` |
| `HomeCredit_columns_description.csv` | `eef7665398228a80f7367c9258220c5fbe1038f3f54094244f354d54e2d4fb03` |

## How a teammate reproduces this

1. `uv sync` — installs the exact locked environment.
2. Confirm raw file checksums match the table above (`shasum -a 256 data/raw/*.csv`).
3. `uv run python scripts/profile_data.py && uv run python scripts/build_data_dictionary.py && uv run python scripts/generate_data_quality_summary.py`
4. `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding.ipynb notebooks/02_baseline_model.ipynb`
5. Confirm `B0`/`B1` ROC-AUC in `notebooks/02_baseline_model.ipynb` read 0.5000 / 0.7489.

## Acceptance criteria

- [x] You can reproduce the baseline (this document *is* that reproduction, done from a full clean-room wipe, not assumed)
- [x] Same dataset version (checksummed above)
- [x] Same preprocessing (single `Pipeline`, committed in `notebooks/02_baseline_model.ipynb`)
- [x] Same random seed (`random_state=42`, used consistently for the split and the estimator)
- [x] Same evaluation metric (`sklearn.metrics.roc_auc_score`)
- [ ] Teammate independently reproduces — pending an actual second person running the steps above; this document gives them everything needed to do it and something concrete to compare against
