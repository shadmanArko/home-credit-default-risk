## README update — paste this over your current "Project Status" section

Replace your existing checklist block with this (checked items reflect what's actually done):

```markdown
## Project Status

✅ Project setup in progress → **complete**

Current stage:

- [x] Python environment
- [x] Dependency management with uv
- [x] Project structure
- [x] Package configuration
- [x] Linting with Ruff
- [x] Automated testing with pytest
- [x] Kaggle data pipeline
- [x] Data validation
- [x] Exploratory data analysis
- [x] Feature engineering (bureau + installment payment history)
- [x] Baseline model (Logistic Regression, LightGBM)
- [x] Model evaluation (error analysis, threshold trade-offs, segment analysis)
- [x] Model comparison (Logistic Regression, Random Forest, LightGBM)
- [x] Final model selection and artifact (`models/final_lightgbm_model.pkl`)
- [x] Final presentation (`reports/final_presentation.pptx`)

## Key Results

| Model | Features | Validation AUC |
|---|---|---|
| Logistic Regression | Application data only | 0.7472 |
| Random Forest (tuned, cross-validated) | + bureau + installment history | 0.7519 |
| LightGBM (baseline) | Application data only | 0.7619 |
| **LightGBM (final)** | **+ bureau + installment history** | **0.7691** |

See `reports/model_card.txt` for full model documentation, known limitations, and
deployment considerations.

## Notebooks

- `notebooks/milestone1_baseline.ipynb` — data cleaning, baseline models
- `notebooks/installments_aggregation.ipynb` — repayment history feature engineering
- `notebooks/milestone3_error_analysis.ipynb` — error analysis, feature engineering, model comparison
- `notebooks/milestone4_final_model.ipynb` — final model training, model card, test predictions
```
