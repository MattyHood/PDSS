# Overview

## PDSS (Python Data Science Summary Tools) is a personal machine-learning toolkit designed to streamline common modelling workflows in Python.
It includes reusable utilities for:

- Regression modelling

- Classification modelling

- Cross-validation

- Hyperparameter optimisation

- ROC-AUC evaluation

- Residual diagnostics

- Variance Inflation Factor (VIF) checks

- Model comparison for multiple algorithms

- PDSS is built to simplify experimentation, improve reproducibility, and provide clean, readable workflows for rapid data-science development.

## Features
### Classification Tools (classification.py)

- Train a single classifier or all default models

- Supported algorithms: Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, KNN, SVC

- Optional randomised hyperparameter optimisation

- Confusion matrix visualisation

- ROC-AUC scoring + plotting

- Cross-validation stability scoring

- Automatic positive-class detection

- Built-in guidance via classification_help()

### Regression Tools (regression.py)

- Train a single regression model or all supported models

- Supported algorithms: Linear Regression, Lasso, Ridge, ElasticNet, Random Forest, Gradient Boosting, SVR, Huber and more

- RandomizedSearchCV hyperparameter optimisation

- VIF calculation to diagnose multicollinearity

- Full diagnostics suite:

- Residuals vs predicted

- Histogram of residuals

- Q–Q normality plot

- Built-in help via regression_help()
