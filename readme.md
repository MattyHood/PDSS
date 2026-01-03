# PDSS  
## Python Data Science Summary Tools

PDSS (Python Data Science Summary Tools) is a personal machine learning toolkit designed to streamline common modelling workflows in Python.

It provides reusable utilities for regression and classification, with a focus on clean workflows, sensible defaults, and decision focused evaluation for real world data science problems.

PDSS is built to simplify experimentation, improve reproducibility, and provide readable, consistent tooling for rapid data science development.

---

## Core Capabilities

PDSS includes utilities for:

- Regression modelling
- Classification modelling
- Cross validation
- Hyperparameter optimisation
- ROC AUC evaluation
- Residual diagnostics
- Variance Inflation Factor (VIF) checks
- Model comparison across multiple algorithms

---

## Classification Tools (`classification.py`)

The classification module is designed for exploratory modelling, benchmarking, and decision focused evaluation, particularly for imbalanced problems such as churn prediction or DNA (Did Not Attend) risk.

### Key Features

- Train a single classifier or benchmark all default models
- Supported algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - K Nearest Neighbours
  - Support Vector Classifier (SVC)
- Stratified train test splitting by default
- Optional RandomizedSearchCV hyperparameter optimisation
- Cross validation with stability interpretation
- Confusion matrix visualisation
- ROC AUC scoring and optional ROC curve plotting
- Probability based threshold tuning using a validation split
- Threshold objectives such as recall, F1, precision, or cost based optimisation
- Automatic handling of probability based predictions when available
- Built in guidance via `classification_help()`

### Design Philosophy

- Recall and confusion matrices drive decision making
- ROC AUC is treated as a ranking diagnostic, not a decision metric
- Threshold selection is a first class step for imbalanced classification
- Simple, readable workflows are prioritised over complex abstractions

---

## Regression Tools (`regression.py`)

The regression module provides a consistent interface for training, comparing, and diagnosing regression models.

### Key Features

- Train a single regression model or benchmark multiple supported models
- Supported algorithms include:
  - Linear Regression
  - Lasso
  - Ridge
  - ElasticNet
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - Huber Regressor and others
- RandomizedSearchCV hyperparameter optimisation
- Variance Inflation Factor (VIF) calculation to diagnose multicollinearity
- Full regression diagnostics suite:
  - Residuals vs predicted plot
  - Histogram of residuals
  - Q Q normality plot
- Built in guidance via `regression_help()`

---

## Typical Workflow

A common PDSS workflow looks like:

1. Prepare features `X` and target `y`
2. Run a quick benchmark using all supported models
3. Select a promising model based on appropriate metrics
4. Optionally optimise hyperparameters
5. Tune decision thresholds for classification problems
6. Validate performance using cross validation and diagnostics

PDSS is designed to support fast iteration while encouraging good modelling hygiene.

---

## Intended Use

PDSS is intended for:

- Exploratory data analysis
- Early stage modelling and benchmarking
- Imbalanced classification problems
- Portfolio projects and reproducible experiments
- Personal and professional data science workflows

It is not intended to replace full production pipelines, but to provide a reliable and consistent modelling toolkit.

---

## Author

Matty Hood

---

## License

MIT
