"""Historical aggregation features (`HC-M3-06`).

Aggregates `bureau`, `previous_application`, and the three event-level
history tables (`POS_CASH_balance`, `installments_payments`,
`credit_card_balance`) up from their native grain to one row per
`SK_ID_CURR` — application grain — via DuckDB SQL against already-landed
tables. Never pandas `.groupby()` on the full event tables: the same
scale reasoning as `scripts/profile_data.py` (`bureau_balance` alone is
27M rows).

**Aggregation keys**: every source table's rows are grouped by
`SK_ID_CURR` directly, except `bureau_balance`, which has no `SK_ID_CURR`
of its own — it's grouped by `SK_ID_BUREAU` and joined through `bureau`
to reach `SK_ID_CURR` (`bureau_balance.SK_ID_BUREAU -> bureau.SK_ID_BUREAU
-> bureau.SK_ID_CURR`), the same join path documented in
`notebooks/01_data_understanding.ipynb`.

**Missing history**: count-style aggregates (how many previous
applications, how many active bureau credits) are filled with 0 for an
applicant with no history in that table — a true zero, not missing.
Ratio/mean/max-style aggregates (mean payment ratio, max days past due)
are left as `NaN` — "no history" is not the same as "0", and collapsing
that distinction would fabricate a signal (e.g. `mean_payment_ratio = 0`
would look like "never paid anything" rather than "no previous credit to
pay"). Left for the modeling pipeline's imputer to handle explicitly,
same discipline as `HC-M3-05`.

**Leakage**: every function's docstring below states the specific reason
its table cannot describe the outcome of the loan under prediction. This
is the implementation-time check; the dedicated audit is `HC-M3-07`.

Each function accepts a `duckdb.DuckDBPyConnection` with the relevant
table(s) already present — either the real landing cache
(`scripts/profile_data.py`), or a small synthetic in-memory connection in
`tests/test_aggregations.py`.
"""

import duckdb
import pandas as pd

from home_credit_default_risk.utils import safe_divide

# bureau_balance.STATUS buckets that represent a delinquent month; 'C'
# (closed), 'X' (unknown/no history that month), and '0' (no DPD) are not
# counted. Verified against reports/data_profile/bureau_balance.json.
DELINQUENT_STATUS_CODES = ("1", "2", "3", "4", "5")

COUNT_COLUMNS = [
    "bureau_credit_count",
    "bureau_active_credit_count",
    "bureau_credit_prolong_count",
    "bureau_balance_bad_month_count",
    "prev_application_count",
    "prev_refused_count",
    "late_payment_count",
    "installment_count",
]


def aggregate_bureau_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group C — previous credit history (`bureau` + `bureau_balance`).

    Leakage note: `bureau` reports credit held at *other* institutions,
    dated relative to the current application (`DAYS_CREDIT` negative) —
    prior/external information, not a peek at this loan's outcome.
    `bureau_balance -> bureau` has a documented ~5.27% orphan rate
    (`reports/data_quality_summary.md`); that slice of monthly detail is
    unattachable and excluded here, not a bug.
    """
    bureau_agg = con.sql(
        """
        SELECT
            SK_ID_CURR,
            COUNT(*) AS bureau_credit_count,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)
                AS bureau_active_credit_count,
            SUM(AMT_CREDIT_SUM) AS bureau_total_credit_sum,
            SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_debt_sum,
            MAX(CREDIT_DAY_OVERDUE) AS bureau_max_days_overdue,
            SUM(CNT_CREDIT_PROLONG) AS bureau_credit_prolong_count
        FROM bureau
        GROUP BY SK_ID_CURR
        """
    ).df()

    bad_months = con.sql(
        f"""
        SELECT
            b.SK_ID_CURR,
            COUNT(*) AS bureau_balance_bad_month_count
        FROM bureau_balance bb
        JOIN bureau b ON bb.SK_ID_BUREAU = b.SK_ID_BUREAU
        WHERE bb.STATUS IN {DELINQUENT_STATUS_CODES}
        GROUP BY b.SK_ID_CURR
        """
    ).df()

    result = bureau_agg.merge(bad_months, on="SK_ID_CURR", how="left")
    result["bureau_balance_bad_month_count"] = result[
        "bureau_balance_bad_month_count"
    ].fillna(0)
    result["bureau_debt_to_credit_ratio"] = safe_divide(
        result["bureau_total_debt_sum"], result["bureau_total_credit_sum"]
    )
    return result


def aggregate_previous_application_features(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Group D — previous application behaviour (`previous_application`).

    Leakage note: `previous_application` is Home Credit's own record of
    this client's *prior* applications; `DAYS_DECISION` is negative
    relative to the current application by the dataset's design.
    """
    result = con.sql(
        """
        SELECT
            SK_ID_CURR,
            COUNT(*) AS prev_application_count,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)
                AS prev_refused_count,
            AVG(AMT_CREDIT) AS prev_mean_credit,
            MAX(AMT_CREDIT) AS prev_max_credit,
            AVG(AMT_APPLICATION / NULLIF(AMT_CREDIT, 0))
                AS prev_application_to_credit_ratio
        FROM previous_application
        GROUP BY SK_ID_CURR
        """
    ).df()
    result["prev_refused_rate"] = safe_divide(
        result["prev_refused_count"], result["prev_application_count"]
    )
    return result


def aggregate_payment_behavior_features(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Group E — payment behaviour (`installments_payments`,
    `POS_CASH_balance`, `credit_card_balance`).

    Leakage note — the highest-risk group, per
    `docs/feature_engineering_strategy.md`: all three tables are keyed by
    `SK_ID_PREV`, a *previous* Home Credit loan, never the current
    application being scored. By the dataset's design there is no row in
    these tables that can correspond to the loan under prediction, since
    it hasn't been disbursed/serviced at prediction time. Re-verified in
    `HC-M3-07`, not just asserted here.
    """
    installments = con.sql(
        """
        SELECT
            SK_ID_CURR,
            COUNT(*) AS installment_count,
            SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END)
                AS late_payment_count,
            AVG(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0)) AS mean_payment_ratio
        FROM installments_payments
        GROUP BY SK_ID_CURR
        """
    ).df()
    installments["late_payment_rate"] = safe_divide(
        installments["late_payment_count"], installments["installment_count"]
    )

    pos = con.sql(
        """
        SELECT SK_ID_CURR, MAX(SK_DPD) AS pos_max_dpd
        FROM POS_CASH_balance
        GROUP BY SK_ID_CURR
        """
    ).df()

    credit_card = con.sql(
        """
        SELECT
            SK_ID_CURR,
            MAX(SK_DPD) AS cc_max_dpd,
            AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))
                AS cc_mean_utilization
        FROM credit_card_balance
        GROUP BY SK_ID_CURR
        """
    ).df()

    result = installments.merge(pos, on="SK_ID_CURR", how="outer")
    result = result.merge(credit_card, on="SK_ID_CURR", how="outer")
    return result


def build_historical_features(
    con: duckdb.DuckDBPyConnection, spine_ids: pd.Series
) -> pd.DataFrame:
    """Left-joins every historical aggregate onto `spine_ids`.

    Guarantees exactly one row per input id, in input order is not
    required but count is: callers validate `len(result) == len(spine_ids)`
    (`HC-M3-06`'s "output row count validated" criterion) since a left
    join fanning out would silently duplicate application rows — the
    single most important structural check for any of these joins.
    """
    spine = pd.DataFrame({"SK_ID_CURR": spine_ids})

    result = spine.merge(aggregate_bureau_features(con), on="SK_ID_CURR", how="left")
    result = result.merge(
        aggregate_previous_application_features(con), on="SK_ID_CURR", how="left"
    )
    result = result.merge(
        aggregate_payment_behavior_features(con), on="SK_ID_CURR", how="left"
    )

    for col in COUNT_COLUMNS:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result
