# Home Credit Default Risk

Machine learning project for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition.

The goal is to develop a reproducible machine learning workflow for predicting the probability that a loan applicant will default.

## Project Status

🚧 Project setup in progress.

Current stage:

- [x] Python environment
- [x] Dependency management with uv
- [x] Project structure
- [x] Package configuration
- [x] Linting with Ruff
- [x] Automated testing with pytest
- [ ] Kaggle data pipeline
- [ ] Data validation
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline model
- [ ] Model evaluation
- [ ] Model comparison
- [ ] Final Kaggle submission

## Tech Stack

- Python 3.12
- uv
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- JupyterLab
- KaggleHub
- Ruff
- pytest

## Project Structure

```text
home-credit-default-risk/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── src/
│   └── home_credit_default_risk/
│
├── scripts/
│
├── tests/
│
├── models/
│
├── reports/
│
├── .github/
│   └── workflows/
│
├── pyproject.toml
├── uv.lock
└── README.md