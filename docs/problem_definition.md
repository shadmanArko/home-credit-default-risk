# Problem Definition — Home Credit Default Risk

## 1. Business problem

Home Credit extends loans to clients who often have thin or no formal credit
history, which makes traditional bureau-score underwriting unreliable for a
large share of applicants. At the moment an applicant applies, the business
must decide whether to approve or reject the loan, and that decision has to
be made from information available at application time — application form
data, prior Home Credit history, and external bureau records — before any
repayment behavior on this loan exists.

The cost of the two ways this decision can go wrong is not symmetric:

- **False negative** (model says low risk, client actually defaults):
  Home Credit disburses the loan and loses some or all of the principal,
  plus collection costs. This is the expensive error.
- **False positive** (model says high risk, client would actually have
  repaid): Home Credit forgoes the interest income on a loan it could
  safely have made, and a creditworthy client is turned away. This is a
  real cost (lost revenue, reputational), but it does not put existing
  capital at risk the way a default does.

Because a missed default is materially more expensive than a missed good
client, the model should be tuned to be conservative about risk once a
threshold is chosen — but that threshold decision is separate from model
selection (see §5).

## 2. Target variable

- **Table**: `application_train.csv`, column `TARGET`.
- **Grain**: one row per loan application (`SK_ID_CURR`).
- **TARGET = 0**: client had no late payment beyond the defined delinquency
  window on the first set of installments of this loan — i.e. a "good"
  client for this loan.
- **TARGET = 1**: client had at least one payment more than the defined
  number of days late on the first set of installments — i.e. a client who
  showed payment difficulty ("bad" client / default proxy).
- `TARGET` exists only in `application_train.csv`. `application_test.csv`
  is unlabeled — it is the Kaggle held-out set used for competition
  scoring, not a validation set we control.

## 3. Prediction objective and output

- **Prediction objective**: given everything known about an applicant at
  the moment of application, estimate the probability that this applicant
  will default per the `TARGET` definition above.
- **Model output**: a probability score in `[0, 1]` per `SK_ID_CURR`
  (`predict_proba`), not a hard class label. The label is a downstream
  business decision (§6), not something the model should decide by
  hard-coding a 0.5 cutoff.
- **Why this is binary classification**: `TARGET` is a discrete outcome
  with exactly two mutually exclusive states (defaulted on this loan /
  did not), known only after the fact for historical loans, and the task
  is to predict which state a new applicant falls into. There is no
  ordinal or continuous structure to `TARGET` itself — the continuous
  quantity we produce (the probability) is an intermediate output, not
  the target.

## 4. Primary metric

- **Primary metric: ROC-AUC**, matching the Kaggle competition's own
  leaderboard metric, evaluated on a held-out validation split.
- **Why ROC-AUC and not accuracy**: `TARGET` is heavily imbalanced (see
  `notebooks/01_data_understanding.ipynb` for the exact ratio). A model
  that predicts "no default" for every applicant would score very high on
  accuracy while being business-useless — it would approve every
  applicant, including all the defaulters. Accuracy rewards the majority
  class by construction and does not reflect this.
- **Why ROC-AUC specifically (over PR-AUC, F1, etc.) as the primary
  metric here**: the model's job in this phase is not to make the final
  approve/reject call — it's to *rank* applicants by relative risk so
  that a later, separately-chosen threshold can decide where to cut. ROC-AUC
  measures exactly that: how well the model orders positives above
  negatives across all thresholds, independent of any one operating
  point. It also matches the competition metric, which keeps our
  validation score comparable to the public leaderboard as a sanity
  check.
- **Supporting metrics** (not primary, but tracked alongside once we have
  a working baseline): Precision-Recall AUC (more informative than
  ROC-AUC under heavy imbalance, worth watching as a secondary signal),
  and — once a threshold is chosen — recall on the default class,
  precision, and a confusion matrix expressed in business terms
  (defaults caught vs. good clients wrongly rejected).
- **Floor / stopping condition**: a `DummyClassifier(strategy="prior")`
  baseline is expected to score ROC-AUC ≈ 0.50 (no ranking ability). A
  logistic regression baseline on `application_*` features alone that
  fails to clear roughly **0.70** would indicate something is
  structurally wrong (a leak, a broken pipeline, or a data issue) rather
  than "the model needs more tuning" — published baselines using only the
  application table on this dataset typically land in the ~0.74–0.76
  range, so a result well below that is a signal to debug, not to keep
  tuning hyperparameters.

## 5. Threshold vs. ranking — why they're kept separate

ROC-AUC and PR-AUC evaluate the model's ranking quality and are what we
optimize for during model selection in this milestone. The actual
approve/reject cutoff is a business decision that trades off the
asymmetric costs in §1, and will be chosen later (in a subsequent
milestone) by inspecting the precision-recall curve on validation
predictions and picking the point that satisfies a recall floor on
defaulters that the business is willing to commit to. That threshold
work is out of scope for Milestone 1.

## 6. Deliverable status

This document is the Milestone 1 deliverable for `HC-M1-01`.

- [x] Business problem documented
- [x] TARGET documented
- [x] Prediction objective documented
- [x] ROC-AUC justified
- [x] Supporting metrics identified
- [ ] Team agrees on definition — pending review
