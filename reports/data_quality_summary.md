# Data Quality Scorecard — Home Credit Default Risk

Generated from `reports/data_profile/*.json` by `scripts/generate_data_quality_summary.py`. Regenerate after re-running `scripts/profile_data.py`. One row per table, every number a full-table DuckDB scan — not a sample.

| Table | Rows | Duplicate rows | PK check | FK orphan checks | Worst missing column | Constant columns |
| --- | --- | --- | --- | --- | --- | --- |
| `POS_CASH_balance.csv` | 10,001,358 | 0 | — | SK_ID_CURR→spine.SK_ID_CURR: 0 orphans; SK_ID_PREV→previous_application.SK_ID_PREV: 37422 orphans | CNT_INSTALMENT_FUTURE (0.2608%) | 0 |
| `application_test.csv` | 48,744 | 0 | SK_ID_CURR: 0 dup keys | — | COMMONAREA_AVG (68.7161%) | 11 |
| `application_train.csv` | 307,511 | 0 | SK_ID_CURR: 0 dup keys | — | COMMONAREA_AVG (69.8723%) | 0 |
| `bureau.csv` | 1,716,428 | 0 | SK_ID_BUREAU: 0 dup keys | SK_ID_CURR→spine.SK_ID_CURR: 0 orphans | AMT_ANNUITY (71.4735%) | 0 |
| `bureau_balance.csv` | 27,299,925 | 0 | — | SK_ID_BUREAU→bureau.SK_ID_BUREAU: 43041 orphans | — | 0 |
| `credit_card_balance.csv` | 3,840,312 | 0 | — | SK_ID_CURR→spine.SK_ID_CURR: 0 orphans; SK_ID_PREV→previous_application.SK_ID_PREV: 11372 orphans | AMT_PAYMENT_CURRENT (19.9981%) | 0 |
| `installments_payments.csv` | 13,605,401 | 0 | — | SK_ID_CURR→spine.SK_ID_CURR: 0 orphans; SK_ID_PREV→previous_application.SK_ID_PREV: 38847 orphans | DAYS_ENTRY_PAYMENT (0.0214%) | 0 |
| `previous_application.csv` | 1,670,214 | 0 | SK_ID_PREV: 0 dup keys | SK_ID_CURR→spine.SK_ID_CURR: 0 orphans | RATE_INTEREST_PRIMARY (99.6437%) | 0 |
| `sample_submission.csv` | 48,744 | 0 | — | — | — | 1 |
