# Feature Engineering Strategy — `HC-M3-04`

Candidate features documented as hypotheses **before** implementation, per
this project's discipline of not building hundreds of features and
figuring out afterward which ones make sense. Every source column below
is verified against `reports/data_profile/*.json` — the actual profiled
schema, not recalled from memory. Implementation happens in `HC-M3-05`
(basic derived features) and `HC-M3-06` (historical aggregations); a
dedicated leakage audit (`HC-M3-07`) happens before any of this is
trusted.

## A. Credit / income ratios

**Business rationale**: how large is the obligation relative to the
applicant's means? An applicant borrowing far more than their income
supports is inherently higher risk, independent of any single raw amount.

| Feature | Formula | Source columns |
|---|---|---|
| `credit_to_income` | `AMT_CREDIT / AMT_INCOME_TOTAL` | `application_train` |
| `annuity_to_income` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | `application_train` |
| `credit_to_annuity` | `AMT_CREDIT / AMT_ANNUITY` | `application_train` (a rough implied loan term) |
| `goods_price_to_credit` | `AMT_GOODS_PRICE / AMT_CREDIT` | `application_train` (down-payment proxy) |

- **Leakage risk**: none — all four source columns are static, known at
  application time, already used raw in the `HC-M1-07` baseline.
- **Priority**: High. Cheap, standard, and one of the most consistently
  useful engineered-feature families on this specific competition.

## B. Age & employment stability

**Business rationale**: age and employment tenure are classic
underwriting signals — younger applicants and less stable employment
histories correlate with higher risk in most consumer credit portfolios.

| Feature | Formula | Source columns |
|---|---|---|
| `age_years` | `-DAYS_BIRTH / 365.25` | `application_train` |
| `employment_years` | `-DAYS_EMPLOYED / 365.25`, **after** recoding the `365243` sentinel | `application_train` |
| `is_employment_unknown` | `DAYS_EMPLOYED == 365243` | `application_train` |
| `employment_to_age_ratio` | `employment_years / age_years` | derived |

- **Leakage risk**: none — static, application-time fields.
- **Known data-quality dependency**: `notebooks/01_data_understanding.ipynb`
  found `DAYS_EMPLOYED = 365243` is a sentinel for "not currently
  employed" (18.01% of applicants, ~99.96% of them `Pensioner`s) — flagged
  then as deferred to feature engineering. This is that moment: the
  sentinel must be recoded to missing (or flagged, not both silently
  treated as a real value) before computing `employment_years`, or the
  handful of sentinel rows would report a 1000-year employment history.
- **Priority**: High — directly resolves a documented data-quality issue,
  not purely speculative.

## C. Previous credit history (`bureau`, `bureau_balance`)

**Business rationale**: how has this applicant handled credit at *other*
institutions? External bureau history is one of the strongest classic
signals in consumer credit risk, and it's information Home Credit itself
wouldn't otherwise have.

| Feature | Aggregation | Source columns |
|---|---|---|
| `bureau_credit_count` | `COUNT(SK_ID_BUREAU)` per `SK_ID_CURR` | `bureau` |
| `bureau_active_credit_count` | count where `CREDIT_ACTIVE = 'Active'` | `bureau.CREDIT_ACTIVE` |
| `bureau_total_credit_sum` | `SUM(AMT_CREDIT_SUM)` | `bureau` |
| `bureau_total_debt_sum` | `SUM(AMT_CREDIT_SUM_DEBT)` | `bureau` |
| `bureau_debt_to_credit_ratio` | `bureau_total_debt_sum / bureau_total_credit_sum` | derived |
| `bureau_max_days_overdue` | `MAX(CREDIT_DAY_OVERDUE)` | `bureau` |
| `bureau_credit_prolong_count` | `SUM(CNT_CREDIT_PROLONG)` | `bureau` |
| `bureau_balance_bad_month_count` | count of `bureau_balance.STATUS` in `{'1','2','3','4','5'}` (days-past-due buckets), joined via `bureau_balance.SK_ID_BUREAU → bureau.SK_ID_BUREAU → bureau.SK_ID_CURR` | `bureau_balance`, `bureau` |

- **Leakage risk**: low. `bureau` records external credit that predates or
  runs alongside the current application by construction — it is not
  information about the outcome of the loan being predicted. One thing to
  verify in `HC-M3-07`, not assume: `DAYS_CREDIT_ENDDATE` can be positive
  (a future contractual end date for a *previous* credit) — that's a
  property of the other credit's terms, not a peek at this application's
  future, but it's exactly the kind of column worth a second look before
  trusting.
- **Known data-quality dependency**: `bureau_balance → bureau` has a
  documented 5.27% orphan rate (`reports/data_quality_summary.md`) — that
  slice of monthly-balance history is simply unattachable to any
  applicant and will be silently excluded from `bureau_balance`-derived
  aggregates. Expected, not a bug to chase.
- **Priority**: High — widely the single most valuable external table in
  this competition per public analysis, and grounded here in verified
  columns rather than assumption.

## D. Previous application behaviour (`previous_application`)

**Business rationale**: has this applicant applied to Home Credit before,
and how did that go? Prior refusals, and how much was requested vs.
actually granted, are direct signals Home Credit itself already has.

| Feature | Aggregation | Source columns |
|---|---|---|
| `prev_application_count` | `COUNT(SK_ID_PREV)` per `SK_ID_CURR` | `previous_application` |
| `prev_refused_count` / `prev_refused_rate` | count / rate where `NAME_CONTRACT_STATUS = 'Refused'` | `previous_application.NAME_CONTRACT_STATUS` |
| `prev_mean_credit` / `prev_max_credit` | `MEAN`/`MAX(AMT_CREDIT)` | `previous_application` |
| `prev_application_to_credit_ratio` | `MEAN(AMT_APPLICATION / AMT_CREDIT)` | `previous_application` (requested vs. granted) |

- **Leakage risk**: none — `previous_application` is Home Credit's own
  record of the client's *prior* applications; `DAYS_DECISION` is
  negative relative to the current application by the dataset's design.
- **Priority**: High.

## E. Payment behaviour (`installments_payments`, `POS_CASH_balance`, `credit_card_balance`)

**Business rationale**: the single most direct signal a lender has — did
this client pay previous installments on time and in full? This is
frequently the most predictive feature family in this competition.

| Feature | Aggregation | Source columns |
|---|---|---|
| `late_payment_count` / `late_payment_rate` | count/rate where `DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT` | `installments_payments` |
| `mean_payment_ratio` | `MEAN(AMT_PAYMENT / AMT_INSTALMENT)` | `installments_payments` |
| `pos_max_dpd` | `MAX(SK_DPD)` | `POS_CASH_balance` |
| `cc_max_dpd` | `MAX(SK_DPD)` | `credit_card_balance` |
| `cc_mean_utilization` | `MEAN(AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL)` | `credit_card_balance` |

- **Leakage risk**: **highest of any group — treat as the priority focus
  of `HC-M3-07`.** These tables are keyed by `SK_ID_PREV` (a *previous*
  Home Credit loan), never the current application being scored, so at
  face value there's no way for them to contain the outcome of the loan
  under prediction. Still the thing to actively verify, not assume,
  before trusting these features: that no row in these tables can ever
  correspond to the current (`application_train`) loan itself.
- **Known data-quality dependency**: `SK_ID_PREV → previous_application`
  orphan rates of 3.9–10.9% across these three tables
  (`reports/data_quality_summary.md`) don't block aggregating them
  directly to `SK_ID_CURR` (all three carry their own `SK_ID_CURR` with 0
  orphans) — they'd only matter if enriching these tables with
  `previous_application` attributes before aggregating, which isn't
  planned here.
- **Priority**: High.

## F. Already-present high-value features (no engineering needed)

`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` in `application_train` are
themselves normalized external credit-bureau scores — already the
single strongest predictors in most public analyses of this competition,
and already used raw in the `HC-M1-07` baseline. Noted here explicitly so
effort isn't wasted re-deriving something that already exists; the actual
opportunity is combining them (e.g. `mean`/`max` across the three, or
counting how many are missing) once modeling starts.

- **Priority**: Low-effort, worth a quick pass — not a research question.

## Prioritization summary

| Group | Priority | Rationale |
|---|---|---|
| A — Credit/income ratios | High | Cheap, standard, no leakage risk |
| B — Age/employment | High | Resolves a known data-quality issue (`DAYS_EMPLOYED` sentinel) |
| C — Bureau history | High | Historically the most valuable external table |
| D — Previous application | High | Direct signal Home Credit already owns |
| E — Payment behaviour | High | Most directly predictive family; highest leakage-audit priority |
| F — `EXT_SOURCE_*` combinations | Low | Already-strong raw features; minor additional engineering only |

## `HC-M3-04` acceptance criteria

- [x] Candidate features documented
- [x] Business rationale documented
- [x] Source columns identified (verified against `reports/data_profile/*.json`)
- [x] Potential leakage risks identified (per group above; Group E flagged as the `HC-M3-07` priority)
- [x] Features prioritized
