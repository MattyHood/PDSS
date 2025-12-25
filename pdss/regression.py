"""
Python Data Science Summary Tools
regression.py v1.3

1/12/2025 Changelog:
Changed function naming from "Train_Regression" to "train_regression" due to better Python practice
Added verbose control for train_regression to allow users to customise responses
Added drop N/A flag for VIF to inform users and verbose control to allow users to understand what is being done
Help text updated to match additional control

03/12/2025 Changelog:
Remodelled train_regression to make it more universal and added parameters model_name = "LinearRegression", all_models = False, optimise = False, n_iter = 10, n_jobs = 4
Added list of generic regression models to use in the all_models and model_name parameters
Added list of RandomSearchCV parameter grid for generic optimisation
Separated train_regression into train_regression and evaluate_model similar to classification.py to allow for all_models and easier model evaluation
Added version (v1.3)

A collection of simple and useful functions for performing
regression modelling, diagnostics, and evaluation within PDSS.


Features:

- regression_help():
    - Overview of how to prepare data for regression
    - Steps for train/test splitting
    - Description of supported models (Linear Regression)
    - Guidance on recommended checks:
        • Pearson correlations
        • Variance Inflation Factor (VIF)
        • Residual diagnostics (homoscedasticity, normality)

- train_regression(X, y, model_name = "LinearRegression", all_models = False, optimise = False, test_size=0.2, random_state=2025, n_iter = 10, n_jobs = 4, verbose = True):
    - Fits a regression model
    - Returns model, predictions, and evaluation metrics
    - Prints MAE, MSE, RMSE, and R² if verbose = True
    - Can be run on all regression models provided in the script
    - Supports basic optimisation using RandomisedSearchCV hyperparameter optimisation
    - n_iter and n_jobs allows for greater customisation over the optimisation using RandomSearch

- calculate_vif(df, dropna = True, verbose = True):
    - Computes Variance Inflation Factor for numeric features
    - Helps identify multicollinearity issues
    - Drops missing values to allow VIF calculation numeric features

- plot_regression_diagnostics(model, X_test, y_test):
    - Generates:
        • Residuals vs predicted plot
        • Histogram of residuals
        • Q–Q normality plot
    - Helps visually assess assumptions of linear regression

Author: Matty Hood
Created: 26/11/2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

# HELP TEXT

def regression_help():
    """
    Help Guide for Regression Tools in PDSS

    Provides step-by-step instructions for preparing data,
    splitting into training/testing sets, training models,
    and checking regression assumptions.

    Usage:
        regression_help()
    """

    text = """
Regression Models
-----------------
Steps for using regression tools in PDSS

1. Prepare your dataframe
   - Ensure numeric dtypes for all feature columns
   - Handle missing values (drop, fill, interpolate)
   - Select numeric columns for X if appropriate:
       Example:
           X = df.select_dtypes(include="number")

2. Train/test split
   Example:
       from sklearn.model_selection import train_test_split
       result = train_test_split(X, y)

3. Recommended Pre-Checks
   - Pearson correlation:
       Helps identify which features relate to the target.
   - Multicollinearity (VIF):
       VIF > 5 = moderate collinearity
       VIF > 10 = strong collinearity (bad)

    Example:
      calculate_vif(df)
   - Inspect correlation heatmap if needed.

4. Fit the model and interpret coefficients
    Example:
      model, metrics, (X_train, X_test, y_train, y_test) = train_regression(X, y)
   - Coefficients indicate how each feature impacts the target
   - Unexpected or very large coefficients may suggest:
       • Collinearity
       • Outliers
       • Incorrect preprocessing

5. Check model assumptions
   - Homoscedasticity:
       Residuals should show constant spread.
   - Normality of residuals:
       Use histogram or Q–Q plot.
   - Linearity:
       Residual plot should not show patterns.
  Example:
    plot_regression_diagnostics(model, X_test, y_test)

Model Supported:
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- RandomForestRegressor
- GradientBoostingRegressor
- KNeighborsRegressor
- SVR

Evaluation metrics:
- MAE
- MSE / RMSE
- R² Score
"""
    print(text)

regressor_map = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "HuberRegressor": HuberRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "KNeighborsRegressor": KNeighborsRegressor,
    "SVR": SVR
}

regression_param_grids = {

    "LinearRegression": {"fit_intercept": [True, False], "copy_X": [True], "positive": [False]},
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100],"fit_intercept": [True, False],"solver": ["auto", "lbfgs", "sag", "saga"]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10],"fit_intercept": [True, False],"max_iter": [1000, 5000]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1],"l1_ratio": [0.2, 0.5, 0.8],"max_iter": [1000, 5000]},
    "RandomForestRegressor": {"n_estimators": [100, 200],"max_depth": [None, 5, 10],"min_samples_split": [2, 5],"min_samples_leaf": [1, 2],"max_features": ["sqrt", "log2", None]},
    "GradientBoostingRegressor": {"n_estimators": [100, 200],"learning_rate": [0.01, 0.05, 0.1],"max_depth": [2, 3, 5],"subsample": [0.8, 1.0]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5, 7],"weights": ["uniform", "distance"],"p": [1, 2]}, # Manhattan, Euclidean
    "SVR": {"C": [0.1, 1, 10],"kernel": ["linear", "rbf"],"gamma": ["scale", "auto"],"epsilon": [0.01, 0.1, 1.0]}
}

# VIF CALCULATOR

def calculate_vif(df, dropna=True, verbose=True):
    """
    Compute Variance Inflation Factor (VIF) for numeric features.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_numeric = df.select_dtypes(include=[np.number])

    if dropna:
        before = len(df_numeric)
        df_numeric = df_numeric.dropna()
        after = len(df_numeric)
        if verbose:
            print(f"Missing values dropped: {before - after} rows removed.")
    else:
        if verbose:
            print("Missing values NOT dropped (may cause errors if NA present).")

    vif_data = pd.DataFrame({
        "feature": df_numeric.columns,
        "VIF": [
            variance_inflation_factor(df_numeric.values, i)
            for i in range(len(df_numeric.columns))
        ]
    })

    return vif_data

# REGRESSION TRAINING FUNCTION

def train_regression(X, y, model_name = "LinearRegression", all_models = False, optimise = False, test_size=0.2, random_state=2025, n_iter = 10, n_jobs = 4, verbose = True):
    """
    Train and evaluate a Linear Regression model.

    Parameters:
        X : DataFrame
            Feature matrix.
        y : Series or array
            Target variable.
        test_size : float, default 0.2
            Size of test set.
        random_state : int
            Reproducibility seed.

    Returns:
        model : fitted LinearRegression model
        metrics : dict of evaluation metrics
        (X_train, X_test, y_train, y_test)
    """

    if model_name not in regressor_map:
        raise ValueError(f"Unknown model '{model_name}'. Valid options: {list(regressor_map.keys())}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if all_models:
        results = {}

        for name, reg in regressor_map.items():
            print(f"\nTraining {name} ...")
            model = reg()

            if optimise and name in regression_param_grids:
                param_grid = regression_param_grids[name]
                max_iter = int(np.prod([len(v) for v in param_grid.values()]))
                n_iter_eff = min(n_iter, max_iter)

                rs = RandomizedSearchCV(
                    model, param_distributions=param_grid,
                    n_iter=n_iter_eff, cv=3, random_state=random_state,
                    n_jobs=n_jobs
                )
                rs.fit(X_train, y_train)
                model = rs.best_estimator_

                if verbose == True:
                    print(f"Best params for {name}: {rs.best_params_}")

            else:
                model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
            results[name] = {"model": model, "metrics": metrics}

        return results

    # Train single model
    if model_name not in regressor_map:
        raise ValueError(f"Unknown model_name {model_name}.")

    model = regressor_map[model_name]()

    if optimise and model_name in regression_param_grids:
        param_grid = regression_param_grids[model_name]
        max_iter = int(np.prod([len(v) for v in param_grid.values()]))
        n_iter_eff = min(n_iter, max_iter)

        rs = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=n_iter_eff, cv=3, random_state=random_state,
            n_jobs=n_jobs
        )
        rs.fit(X_train, y_train)
        model = rs.best_estimator_
        if verbose == True:
            print(f"Best params for {model_name}: {rs.best_params_}")
    else:
        model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    return model, metrics, (X_train, X_test, y_train, y_test)

def evaluate_model(model, X_train, X_test, y_train, y_test, verbose=True):
    """Evaluate a regression model and return metrics."""

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    if verbose:
        print(f"\nRegression Model {model} Performance")
        print("----------------------------")
        print(f"MAE :  {mae:.4f}")
        print(f"MSE :  {mse:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"R²   :  {r2:.4f}")

    return metrics


def plot_regression_diagnostics(model, X_test, y_test):
    """
    Generate diagnostic plots for a regression model:
        - Residuals vs Predicted
        - Histogram of residuals
        - Q–Q plot of residuals
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Residuals vs Predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")

    # Histogram
    ax = axes[1]
    ax.hist(residuals, bins=20)
    ax.set_title("Residual Histogram")

    # Q–Q plot
    ax = axes[2]
    sm.qqplot(residuals, line="s", ax=ax)
    ax.set_title("Q–Q Plot")

    plt.tight_layout()
    plt.show()
