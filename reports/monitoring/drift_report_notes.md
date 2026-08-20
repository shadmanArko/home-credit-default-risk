# Drift monitoring — `HC-M4-12`/`13`

The generated report itself (`drift_report.html`, ~11 MB, Evidently's
own HTML/JS bundle) is not committed — regenerate it with
`uv run python scripts/generate_drift_report.py`. This file is the
durable, versioned write-up of what it found and what it means, in the
same spirit as `reports/reproducibility_check.md`.

## What was compared

**Reference**: the development pool (246,008 rows) — everything the
model was trained and cross-validated on. **Current**: `application_test`
(48,744 rows) — the one batch in this entire dataset that has never been
used for training, model selection, tuning, or the `HC-M3-21` holdout
evaluation. It's the closest thing available to genuinely new production
data, which is why it's the drift baseline here rather than reusing the
already-spent holdout.

Both feature drift (149 engineered columns, `HC-M3-09`'s full feature
matrix) and prediction drift (the registered model's predicted
probability, added as one more column before running the same drift
preset) are checked together, using Evidently's `DataDriftPreset`.

## Real results

- **Dataset drift: not detected.** 15 of 150 columns (10%) show
  statistically detected drift — well under Evidently's 50% dataset-level
  threshold.
- **The drifted columns are concentrated in loan-amount fields and
  credit ratios**: `credit_to_annuity` (drift score 0.569),
  `AMT_REQ_CREDIT_BUREAU_QRT` (0.465), `credit_to_income` (0.295),
  `AMT_REQ_CREDIT_BUREAU_MON` (0.281), `bureau_balance_bad_month_count`
  (0.253), `AMT_GOODS_PRICE` (0.211), `AMT_CREDIT` (0.208), `AMT_ANNUITY`
  (0.161), `prev_application_count` (0.159), `AMT_REQ_CREDIT_BUREAU_WEEK`
  (0.153) round out the top ten. A plausible read: `application_test`
  and the development pool may have been drawn from slightly different
  origination periods or applicant mixes by Kaggle's own split — this
  wasn't investigated further here (out of scope for this chunk), but is
  exactly the kind of finding a real deployment would follow up on before
  dismissing it.
- **Prediction drift: not detected, and by a wide margin.** The
  predicted-probability Wasserstein distance is **0.031**, against
  Evidently's own default alert threshold of **0.1** — mean predicted
  probability moved from 0.392 (reference) to 0.386 (current). Despite
  10% of input columns showing detected drift, the model's actual risk
  output for the population barely moved. This is the reassuring
  finding: feature-level drift here isn't (yet) translating into a
  meaningfully different risk assessment.

## `HC-M4-13` — what would trigger retraining in a real deployment

- **Primary signal: prediction drift, not feature drift.** Feature drift
  is a leading indicator worth watching, but it's prediction drift that
  actually reflects "the model's real-world behavior changed" — that's
  the one tied to business impact. Recommended alert threshold: the same
  **0.1 Wasserstein distance** Evidently defaults to, checked against a
  rolling window of recently-scored applicants rather than a one-off
  batch.
- **Secondary signal: dataset drift share.** If the fraction of drifted
  columns climbs well past this report's 10% (e.g. crosses 30–40%) even
  while prediction drift stays low, that's worth a manual look before it
  becomes a prediction-drift problem — an early warning, not an
  auto-retrain trigger by itself.
- **Cadence**: for a lending model with the volume implied by this
  dataset, a **weekly** scheduled drift job comparing the trailing
  window of scored applicants against this same development-pool
  reference is a reasonable default — frequent enough to catch a real
  shift within days, not so frequent that it's reacting to noise.
- **What happens when it fires**: a new training run through the
  `HC-M4-01`–`03` MLflow registry infrastructure already built —
  logged, versioned, and evaluated on the same CV/holdout discipline as
  every model in this project, but **not auto-promoted**. Given the
  regulatory and financial stakes of a lending model, promoting a
  retrained model to the `production` alias should require a human
  reviewing the new model's CV/holdout metrics against the currently
  deployed version first — this closes the loop back to the registry
  without making retraining decisions fully automatic.
