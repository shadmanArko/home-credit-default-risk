# Home Credit Default Risk — Complete Project Workflow

*From business question to final model — written in plain language, for both technical and non-technical readers.*

---

## 1. The Business Problem

Home Credit is a lender that serves people who often don't have a traditional credit history — the kind of applicant a typical bank might turn away simply because there isn't enough data on them. The company's challenge is deciding, for each loan application, whether the person is likely to repay or likely to default.

Two kinds of mistakes are possible, and both cost the business:

- **Approving someone who later struggles to repay** → financial loss, collections costs.
- **Rejecting someone who would have repaid fine** → lost revenue, and a customer turned away who didn't deserve to be.

A machine learning model can help by estimating, from the information available at application time, how risky each applicant is — but it can only ever be a decision-support tool. The final call on what risk level is acceptable is a business decision, not a modeling one.

> **My take:** I think this framing matters a lot. It's tempting to present a model as "the answer," but really it's a ranking tool — it tells you who's riskier than whom, not what to actually do about it. Keeping this distinction visible builds trust, because it's honest about what the model can and can't decide on its own.

---

## 2. Understanding the Data

The dataset comes from a public Kaggle competition and includes one main table (one row per loan application) plus several related tables capturing an applicant's history with other lenders and with Home Credit itself.

| Table | Rows | What it captures |
|---|---|---|
| `application_train.csv` | 307,511 | One row per applicant: demographics, income, loan details, and the outcome (defaulted or not) |
| `bureau.csv` | 1,716,428 | Applicant's credit history with other financial institutions |
| `installments_payments.csv` | 13,605,401 | History of actual installment payments on past credits — used to check punctuality |

Only about **8% of applicants** in the training data actually defaulted. This imbalance shaped almost every later decision — a model that always guesses "will repay" would already be right 92% of the time, which sounds good but is useless. It's why plain accuracy was never used as a metric in this project.

> **My take:** This is usually the first surprise for a non-technical audience — a 92%-accurate model can be worthless. I'd lead with this fact early in any presentation, because it explains why every later decision (the metric chosen, how the data was split, how the model was weighted) traces back to this one number.

---

## 3. Our Approach

We followed the standard data science lifecycle: understand and clean the data, build a simple baseline model, analyze where it goes wrong, improve it, and compare alternatives.

### 3.1 Choosing a metric: AUC-ROC

Because of the imbalance above, we used **AUC-ROC** instead of accuracy. In plain terms, AUC measures how well the model separates "will default" applicants from "will repay" applicants, across every possible risk cutoff — not just one fixed guess. A score of 0.50 means no better than a coin flip; 1.00 would mean perfect separation. This also matches the metric used in the original Kaggle competition.

### 3.2 Cleaning decisions — and why they matter

Several columns were missing for a large share of applicants (some over 50%). The easy option is to drop anything too incomplete. We mostly did that — **except** for `EXT_SOURCE_1` (an external credit score), missing for 56% of applicants but one of the three strongest predictors in the entire dataset. Dropping it by a blind rule would have thrown away real signal.

> **My take:** A "50% missing, so drop it" rule is a reasonable default, but it's still just a heuristic — and heuristics can be wrong. This is a good, concrete example of why a human should double-check automated rules rather than trusting a threshold blindly.

We also found a subtler issue: `DAYS_EMPLOYED` contained a placeholder value of **365,243 days** (over 1,000 years) for about **18% of applicants** — almost certainly a hidden code meaning "not currently employed" (likely retirees). Left as-is, this distorted the model's internal reasoning, even though it didn't change the model's overall accuracy.

> **My take:** My favorite finding to explain to a non-technical audience — "the computer thought some people had jobs for a thousand years." A good illustration of why data quality checks matter even when the headline number looks fine.

---

## 4. Baseline Results (Milestone 1)

Two simple models established a starting point:

| Model | AUC Score | Notes |
|---|---|---|
| Logistic Regression | 0.7472 | A simple, interpretable linear model |
| LightGBM | 0.7619 | A tree-based model well suited to this kind of data |

The strongest predictors by far were the three "external source" scores — third-party credit scores Home Credit has access to — followed by loan size, age, and employment status.

> **My take:** For a non-technical audience, I'd translate 0.76 as: "if you picked one random defaulter and one random non-defaulter from our data, the model correctly identifies the riskier one about 76% of the time."

---

## 5. Error Analysis (Milestone 3)

A single accuracy-style number hides a lot. So we asked: when the model is wrong, who is it wrong about, and in which direction?

### 5.1 The two ways the model can be wrong

- **Missed defaulter:** the model says "safe to approve," but the applicant actually defaults. Usually the costlier mistake for a lender.
- **Wrongly rejected:** the model says "risky," but the applicant would actually have repaid. Costs lost revenue and a bad customer experience.

At a standard 50/50 cutoff, the model catches about **67% of actual defaulters** (recall), but only **17%** of applicants flagged as risky actually default (precision) — for every genuine defaulter correctly caught, roughly five good applicants get flagged as risky too.

> **My take:** This trade-off is the single most important thing for stakeholders to understand and weigh in on. The model doesn't have one "correct" setting — it can be tuned to catch more defaulters at the cost of rejecting more good customers, or vice versa. That's a business decision, not a data science one.

### 5.2 The most important finding: a reliability gap by age

When we broke down errors by applicant age, a clear pattern emerged: **older applicants default less often overall** (which the model correctly learns), but **when an older applicant does default, the model is far more likely to miss it** — the miss rate climbs from about 20% (ages 20-30) to about 66% (ages 60-70).

The likely explanation: the model has learned "older = safe" as a strong general rule (which is mostly true), and becomes overconfident about the rare exceptions. A very similar pattern shows up for applicants who are not currently employed (largely overlapping with the older group).

> **My take:** This is not evidence the model is "biased against" any group in the sense of hurting them — if anything, older applicants are treated more leniently overall. The real issue is the opposite: the rare older applicant who is genuinely risky is more likely to slip through undetected. I'd frame this as a **reliability gap** for a specific group, not a fairness complaint, and be precise about that distinction if asked.

We also checked gender and education level. Gender showed a difference in the trade-off between the two error types between groups, worth a neutral, transparent mention. Education broadly followed expected patterns.

---

## 6. Feature Engineering — Adding More History

The baseline model only used the application form itself. We added two more sources of information:

- **Credit bureau history:** how many past credits an applicant has had elsewhere, how much they owe, and whether they've ever been overdue.
- **Repayment behavior:** from Home Credit's own past installment records — did the applicant pay on time, pay less than owed, or pay late — aggregated per applicant.

Adding both improved LightGBM's AUC from **0.7619 to ~0.7687**. Several of the new features — particularly average payment ratio and average days late — ranked among the model's most important signals, just behind the original external credit scores.

> **My take:** A clean, satisfying result: "we gave the model more relevant history, and it got measurably better at its job." It's also honest to say the improvement was incremental, not transformative — the external credit scores remain by far the strongest signal.

---

## 7. Comparing Three Algorithms (Milestone 3)

The project called for comparing at least three modeling approaches, each properly validated:

| Model | Features Used | AUC |
|---|---|---|
| Logistic Regression | Application data only | 0.7472 |
| Random Forest (tuned, cross-validated) | + bureau + repayment history | 0.7519 |
| LightGBM (baseline) | Application data only | 0.7619 |
| **LightGBM (final)** | **+ bureau + repayment history** | **~0.7687** |

Random Forest was tuned using cross-validation — testing each candidate setting on several different slices of the data before picking the best one, to avoid accidentally choosing settings that only look good by chance on one particular split.

> **My take:** Random Forest didn't win here, and that's fine — an honest three-way comparison is more convincing than quietly dropping the model that didn't come out on top. It also gives a natural talking point: gradient boosting models like LightGBM tend to handle this kind of messy, tabular data with lots of missing values better than Random Forest does.

---

## 8. Final Model & Deliverables (Milestone 4)

The winning pipeline — LightGBM with application, bureau, and repayment history features — was retrained on the full training set and finalized:

- **Final validation AUC: ~0.7687**
- **Saved model artifact:** `models/final_lightgbm_model.pkl` (model + encoders + feature list, ready to reload)
- **Model card:** `reports/model_card.txt` — plain-language documentation of performance, intended use, and known limitations
- **Test-set predictions:** `reports/final_submission.csv` — generated for all 48,744 applicants in the official Kaggle test set

### Known limitations (documented in the model card)

- The model is systematically less reliable at catching rare defaulters among older (60+) and non-employed applicants, despite correctly identifying these groups as lower-risk overall.
- The 0.5 approval threshold used during evaluation is a placeholder. The real deployment threshold should be set based on the business's relative cost of a missed default vs. a wrongly rejected applicant.
- Trained on historical data only; does not account for economic conditions that may differ from the training period.

---

## 9. Key Takeaways & Open Questions for Stakeholders

- The final model reaches an AUC of about 0.77, a meaningful improvement over the simple baseline of 0.75.
- The model's ranking ability is solid, but the actual approve/reject cutoff is a business decision: **what's more costly — one missed default, or several wrongly rejected good applicants?** We need this input to set the right threshold.
- The model is currently less reliable at catching rare defaulters among older and non-employed applicants. Worth monitoring after deployment, and may warrant a closer look before rollout.
- Further gains are possible: more history tables were not used yet (e.g. previous applications, monthly balance records) and could add more signal.

> **My take:** If I had to leave stakeholders with one sentence, it would be: "the model tells you who is riskier, but you still have to tell us how much risk you're willing to accept." That's the crux of translating a technical result into a business decision.

---

## 10. Reproducing This Project

### Repo structure
```
home-credit-default-risk/
  data/
    raw/            <- original Kaggle CSVs
    processed/       <- engineered features (bureau_agg.csv, installments_agg.csv)
  notebooks/
    milestone1_baseline.ipynb          <- cleaning, baseline models
    installments_aggregation.ipynb     <- repayment history feature engineering
    milestone3_error_analysis.ipynb    <- error analysis, feature engineering, model comparison
    milestone4_final_model.ipynb       <- final model, model card, test predictions
  scripts/
    aggregate_installments.py          <- standalone script version of the installments aggregation
  models/
    final_lightgbm_model.pkl           <- saved final model
  reports/
    model_card.txt                     <- model documentation
    final_submission.csv               <- test-set predictions
    final_presentation.pptx            <- final slide deck
```

### To reproduce from scratch
```bash
# 1. Set up the environment
uv sync

# 2. Aggregate the large installments table (takes ~1 minute)
uv run python scripts/aggregate_installments.py

# 3. Run the notebooks in order (via VS Code or Jupyter Lab)
#    milestone1_baseline.ipynb -> milestone3_error_analysis.ipynb -> milestone4_final_model.ipynb
```

### To reload the final model later
```python
import joblib

bundle = joblib.load("models/final_lightgbm_model.pkl")
model = bundle['model']
preds = model.predict(new_data[bundle['feature_names']], num_iteration=bundle['best_iteration'])
```

---

## Appendix: Glossary (Plain-Language)

| Term | Plain-language meaning |
|---|---|
| **AUC-ROC** | A score from 0.5 (no better than guessing) to 1.0 (perfect) measuring how well the model tells risky and safe applicants apart, across all possible cutoffs. |
| **Recall** | Of all the applicants who actually defaulted, what percentage did the model correctly flag as risky? |
| **Precision** | Of all the applicants the model flagged as risky, what percentage actually defaulted? |
| **False Negative** | A missed defaulter — the model said "safe," but they defaulted. Usually the costlier mistake. |
| **False Positive** | A wrongly rejected good applicant — the model said "risky," but they would have repaid. |
| **Cross-validation** | Testing a model's settings on multiple different slices of the data, to make sure good results aren't just luck from one split. |
| **Feature importance** | A ranking of which pieces of information the model relies on most heavily to make its predictions. |
