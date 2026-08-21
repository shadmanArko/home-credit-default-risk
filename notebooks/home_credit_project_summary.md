# Home Credit Default Risk — Project Summary

*Combines `01_data_understanding.ipynb` → `04_modeling.ipynb` into one narrative.*

## 1. The Business Problem

Home Credit lends to people who often lack a traditional credit history. For every loan
application, the business needs to estimate how likely the applicant is to repay — with two
costly mistakes to avoid: approving someone who later defaults, or rejecting someone who
would have repaid fine. A model can rank applicants by risk; deciding what risk level to
accept is a business call, not a modeling one.

## 2. The Data

Nine linked tables, all keyed back to `SK_ID_CURR` (one row = one applicant):

| Table | Rows | What it captures |
|---|---|---|
| `application_train/test` | 307,511 / 48,744 | The application itself: income, family, housing, employment |
| `bureau` | 1,716,428 | Credit history with other lenders |
| `bureau_balance` | 27,299,925 | Monthly snapshots of those other-lender credits |
| `previous_application` | 1,670,214 | The applicant's past applications to Home Credit itself |
| `POS_CASH_balance` | 10,001,358 | Monthly status of past point-of-sale/cash loans |
| `credit_card_balance` | 3,840,312 | Monthly credit card statements |
| `installments_payments` | 13,605,401 | Actual repayment records on past credits |

**Only 8.07% of applicants defaulted** (an ~11.4:1 imbalance) — meaning a model that always
guesses "will repay" would be 92% "accurate" while being completely useless. This single fact
drove every later decision: the metric used (AUC-ROC, not accuracy), the stratified data split,
and the model's class weighting.

**Data quality issues found and handled:**
- `DAYS_EMPLOYED` contained a placeholder value of 365,243 days (\"employed for 1,000 years\")
  for 18% of applicants — almost entirely pensioners. Recoded as missing + flagged.
- Several `application` columns were 60–70% empty (building-metadata fields) — dropped.
- `bureau_balance` has no direct link to an applicant — it only connects via `bureau`, and about
  5.3% of that link doesn't resolve cleanly (flagged, not silently ignored).

## 3. Baseline (Milestone 1)

Using only the raw application form, two models were compared on an 80/20 stratified split:

| Model | Validation AUC |
|---|---|
| Dummy (always guesses the average) | 0.5000 |
| Logistic Regression | **0.7489** |

This 0.25-point gap over the dummy model is the proof the model is learning something real,
not just guessing — and it's the number every later model has to beat.

## 4. Feature Engineering (Milestone 3)

Two kinds of new information were added on top of the raw application form:

- **Simple ratios & flags**: credit-to-income, annuity-to-income, credit-to-annuity,
  goods-price-to-credit, applicant age in years, years employed, and a flag for
  "employment status unknown" (the pensioner group found above).
- **History aggregated per applicant** from the other five tables: how many past credits
  they have, how much they owe, whether they've missed payments, how they've used credit
  cards, and how their past Home Credit applications went.

Every new feature was checked for **leakage** — making sure nothing uses information that
wouldn't actually be available at the moment of application (all history dates were confirmed
to be safely in the past) — and for near-duplicate features (only one close pair was found and
kept deliberately, since it wasn't truly redundant).

**14.3% of applicants had no bureau credit history at all** — a concrete, measured confirmation
that Home Credit really does serve a "thin-file" population, not just a stated assumption.

## 5. Comparing Models (Milestone 3)

Five candidates were compared using 5-fold cross-validation, all on identical folds:

| Model | Features | AUC | Std |
|---|---|---|---|
| Dummy | — | 0.5000 | 0.0000 |
| Logistic Regression | application only | 0.7442 | 0.0016 |
| Logistic Regression | + engineered features | 0.7550 | 0.0026 |
| **LightGBM** | + engineered features | **0.7747** | **0.0020** |
| XGBoost | + engineered features | 0.7579 | 0.0016 |

**LightGBM won clearly** — 1.7 points ahead of XGBoost and 2 points ahead of the best logistic
regression, well beyond the run-to-run noise (~0.002). Both boosting models were given an equally
fair setup (same features, same folds, same class-imbalance handling), so this is a genuine result,
not a stacked comparison.

A hyperparameter search was then run on LightGBM: it found a configuration that scored slightly
better (0.7778 vs. 0.7747) but fit the training data far more tightly — a sign it was starting to
memorize noise rather than learn a generalizable pattern. **The simpler, untuned LightGBM was kept**
as the final model, since the tiny accuracy gain wasn't worth the added overfitting risk.

## 6. Final Model & Evaluation (Milestone 3)

The chosen model — LightGBM, default settings, application + engineered history features — was
retrained on the full development set and evaluated **exactly once** on a holdout set it had never
touched during any of the steps above:

| Metric | Value |
|---|---|
| Holdout ROC-AUC | **0.7791** |
| PR-AUC | 0.2758 |
| Development (cross-validation) estimate | 0.7747 |

The holdout score coming in *above* the cross-validation estimate is a good sign — it means the
model generalizes to genuinely unseen applicants at least as well as expected, not worse.

**What drives the model's decisions:** the three external credit-bureau scores Home Credit already
has access to account for nearly half (48%) of the model's total decision weight. Beyond those,
the engineered features earn real, meaningful weight — credit-to-annuity ratio, late payment
rate, credit card utilization, and employment history all rank in the top 12 features.

**Threshold trade-off:** at a naive 50% cutoff, the model catches about 69% of real defaulters,
but only 18% of applicants it flags as risky actually default — for every genuine defaulter
correctly caught, roughly four to five good applicants get flagged too. A specific operating
threshold (0.485) was chosen based on this trade-off, but the real cutoff should ultimately be
set by the business, based on which mistake costs more.

**Known limitations, documented rather than hidden:**
- The model relies heavily on external bureau scores it doesn't fully control or explain.
- A gender subgroup check found a modest difference in the flagging rate between men and
  women (43% vs. 27%) — noted for stakeholder awareness, not something the model was
  re-tuned to correct.
- Trained on one lender's historical data — performance under different economic conditions
  or a different population isn't guaranteed.
- The hyperparameter search itself was found to be somewhat unstable between runs — a caveat
  about the tuning process, not the final chosen model.

## 7. Bottom Line

The final model reaches a holdout AUC of about **0.78** — a real, evidence-backed improvement
over the 0.75 baseline and the 0.5 "no information" floor. It relies heavily on third-party
credit scores but also draws genuine, measurable value from custom-built features. The remaining
open decision for the business: **what's the acceptable trade-off between catching more real
defaulters and rejecting more good applicants?**
