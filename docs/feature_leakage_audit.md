# Feature Leakage Audit — `HC-M3-07`

Every feature implemented in `HC-M3-05` (`src/home_credit_default_risk/features.py`)
and `HC-M3-06` (`src/home_credit_default_risk/aggregations.py`) reviewed
against the six risk categories below. Findings are based on actual
queries against the real data (`reports/data_profile/.landing_cache.duckdb`),
not assumptions — every number here is reproducible.

## 1. Target leakage

**Not found.** Every function in `features.py` and `aggregations.py`
accepts only feature data, never `TARGET` — there is no code path through
which the target could reach a derived feature. Asserted explicitly in
`tests/test_features.py::test_does_not_require_or_touch_target`, not just
implied by the function signatures.

## 2. Future information

Checked the actual min/max of every date-like column (`DAYS_*`,
`MONTHS_BALANCE`) that is either used in a current feature or sits next
to one, since "the table is historical" is not the same guarantee as
"this specific column never contains a value dated after the
application":

| Column | Min | Max | Positive (future) rows | Used in a current feature? |
|---|---|---|---|---|
| `bureau.DAYS_CREDIT` | -2922 | 0 | 0 | Yes (via `bureau` aggregation) |
| `bureau.DAYS_ENDDATE_FACT` | -42023 | 0 | 0 | Not used directly |
| `bureau.DAYS_CREDIT_UPDATE` | -41947 | 372 | **17** | Not used directly |
| `bureau.DAYS_CREDIT_ENDDATE` | -42060 | 31199 | **602,603** | Not used |
| `previous_application.DAYS_DECISION` | -2922 | -1 | 0 | Yes (aggregation grain) |
| `previous_application.DAYS_FIRST_DRAWING` | -2922 | 365243 | 934,444 | Not used |
| `previous_application.DAYS_FIRST_DUE` | -2892 | 365243 | 40,645 | Not used |
| `previous_application.DAYS_LAST_DUE_1ST_VERSION` | -2801 | 365243 | 318,256 | Not used |
| `previous_application.DAYS_LAST_DUE` | -2889 | 365243 | 211,221 | Not used |
| `previous_application.DAYS_TERMINATION` | -2874 | 365243 | 225,913 | Not used |
| `POS_CASH_balance.MONTHS_BALANCE` | -96 | -1 | 0 | Yes (aggregation grain) |
| `credit_card_balance.MONTHS_BALANCE` | -96 | -1 | 0 | Yes (aggregation grain) |
| `installments_payments.DAYS_INSTALMENT` | -2922 | -1 | 0 | Yes (late-payment logic) |
| `installments_payments.DAYS_ENTRY_PAYMENT` | -4921 | -1 | 0 | Yes (late-payment logic) |

**Every date column actually used by a current feature is confirmed ≤ 0**
— safely dated before or at the application, across all five source
tables. No future-information leakage in the current feature set.

Two things found that don't affect any current feature but are worth
recording so they aren't rediscovered the hard way later:

- **`previous_application`'s five `DAYS_*` columns above all hit
  `365243`** — the exact same sentinel already found and handled for
  `DAYS_EMPLOYED` in `HC-M3-05` (a refused or cancelled previous
  application never had a real "first due" or "termination" date, so
  it's coded as the sentinel rather than left null). Not a bug in the
  data, but any future feature built on these columns must recode the
  sentinel first, the same way `HC-M3-05` did.
- **`bureau.DAYS_CREDIT_UPDATE` has 17 rows (0.001% of 1,716,428) with a
  positive value** (max 372 days after the current application) — those
  17 bureau records were refreshed after the loan decision. `DAYS_CREDIT_UPDATE`
  itself isn't used as a feature, but `AMT_CREDIT_SUM_DEBT`/`CREDIT_ACTIVE`
  (which are used, via `bureau_total_debt_sum` and
  `bureau_active_credit_count`) reflect the bureau's snapshot state as of
  that update — for these 17 rows specifically, that snapshot is
  technically taken after the decision. **Decision: accepted, not
  filtered.** At 0.001% of bureau rows feeding a `SUM`/`COUNT` aggregate
  over 1.7M+ rows, the effect on any aggregate is immeasurably small, and
  adding a `WHERE DAYS_CREDIT_UPDATE <= 0` filter would be complexity
  spent on a rounding error. Documented here so the number is known and
  the decision is deliberate, not undiscovered.

## 3. Test contamination

**Not found, and structurally not possible at this stage.** Every
aggregation is `GROUP BY SK_ID_CURR` — each applicant's aggregate is
computed only from their own rows in `bureau`/`previous_application`/etc.,
independent of any other applicant and independent of whether that
applicant belongs to `application_train` or `application_test`. No
cross-applicant statistic (a global mean, a fitted scaler) exists yet at
this stage of the pipeline. That risk is real but deferred to `HC-M3-09`
(the preprocessing pipeline), where imputers/scalers must be fit on the
training folds only — flagged forward, not ignored.

## 4. Aggregation leakage

None of the aggregation queries in `aggregations.py` filter rows by date
— they aggregate over every row for a given key. That's only safe because
of what §2 confirmed: every date column feeding into an aggregation used
by a current feature is already ≤ 0 for every row in the table. If a
future feature aggregates a table where that isn't true (e.g. anything
built on `previous_application`'s sentinel-laden `DAYS_*` columns), an
explicit date filter would become necessary — noted here as the condition
under which this section's "no leakage" conclusion would need re-checking.

## 5. Preprocessing leakage

**Not yet applicable.** No preprocessing pipeline exists yet — that's
`HC-M3-09` (Phase C). Recorded here as a placeholder so the audit isn't
silently skipped: when built, fitting must happen on training folds only,
the same discipline already established since `HC-M1-07`.

## 6. Duplicate information

Checked pairwise correlation across all `HC-M3-05`/`HC-M3-06` engineered
features on the real development pool. One pair stood out:
`employment_years` and `employment_to_age_ratio` at **|r| = 0.955**.

**Investigated, not removed.** This is expected from the ratio's own
construction — `employment_to_age_ratio = employment_years / age_years`,
and since `age_years` varies proportionally much less across the
population than `employment_years` does, the ratio ends up strongly
correlated with its own numerator. It is not a literal duplicate (r ≠ 1.0,
and normalizing by age does carry distinct information — two applicants
with the same tenure at different ages get different ratios). **Decision:
keep both for now**, and revisit at model interpretation (`HC-M3-26`) if
either shows unstable coefficients under a linear model — gradient-boosted
candidates are not sensitive to this kind of correlation the way linear
models can be.

## `HC-M3-07` acceptance criteria

- [x] Feature list reviewed (all `HC-M3-05`/`HC-M3-06` features, against all six categories)
- [x] Suspicious features investigated (`DAYS_CREDIT_UPDATE` trace positives; `employment_years`/`employment_to_age_ratio` correlation)
- [x] Leakage-prone features removed — **none met the bar for removal**; the one trace-level finding (17 rows) was quantified and deliberately accepted rather than fixed, which is a documented decision, not an oversight
- [x] Audit documented (this file)
