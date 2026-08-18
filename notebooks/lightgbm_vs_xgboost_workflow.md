# Comparing Two Prediction Engines for Spotting Risky Loan Applicants

## The Goal

We already have a model (**LightGBM**) predicting which loan applicants are likely to default. We wanted to check: is there a better tool out there? So we tested a well-known alternative called **XGBoost**, side-by-side, on the exact same data.

## Step 1 — Load the Data

We used the same customer application data from the start of the project — over 300,000 past applicants, with dozens of pieces of information about each one (income, credit history, family situation, etc.).

## Step 2 — Clean It Up

Same cleanup steps as before:
- Removed columns that were mostly empty (except one we know is genuinely useful despite being incomplete)
- Filled in blanks for missing information
- Fixed a known data glitch where "not currently employed" had been mistakenly recorded as "employed for 1,000 years"

## Step 3 — Split the Data into Two Piles

- **80%** went into a "practice" pile the models learn from.
- **20%** was set aside as a "test" pile the models never saw during training.

This checks whether a model actually understands the patterns, rather than just memorizing the practice data.

## Step 4 — Train Three Models on the Identical Practice Pile

| Model | Role |
|---|---|
| Logistic Regression | Simple, transparent baseline — a sanity check |
| LightGBM | Our current model |
| XGBoost | The new challenger |

Using the exact same practice and test piles for all three is the key fairness step: any difference in performance reflects the model itself, not one model getting an easier test.

## Step 5 — Score All Three on the Test Pile

Each model gives every test applicant a risk score. We measure how well each model separates "will default" from "won't default" using one number (**AUC**) — closer to 1.0 means better at telling risky and safe applicants apart.

## Step 6 — Compare Results Side-by-Side

A final table shows all three models' accuracy scores and how long each took to train, so we can see who came out ahead, by how much, and whether the winner is worth any extra time or complexity.

## Bottom Line

> We're not replacing our approach — we're stress-testing it. We put a respected alternative (XGBoost) up against our current model (LightGBM) on identical data. Whichever wins, we now have solid evidence backing the choice, not just an assumption.

---

*Results table to be added once the comparison notebook is run on the full dataset.*
