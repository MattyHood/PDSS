"""
Python Data Science Summary Tools
classification.py

26/11/2025 Changelog:
- Created initial classification module
- Added basic training, evaluation, and diagnostic plotting
- Supports multiple classifiers with optional automated hyperparameter tuning

01/12/2025 Changelog:
- Added validate_model to validate whether the model was useful (similar to R²)
- train_classifier instead of Train_Classifier for better Python practice
- Removed duplicate library imports and tidied unnecessary code
- Edited max_iter function to decrease the chance of code breaking errors based on user input

02/12/2025 Changelog:
- Removed duplicate libraries
- Added plot_roc_auc function and added comments explaining how it works
- Standardised formatting for module notes

A collection of simple and useful functions for performing
classification modelling, evaluation, and visualisation within PDSS.

Features:

- classification_help():
    - Overview of how to prepare data for classification
    - Steps for train/test splitting
    - Description of supported classifiers
    - Guidance on recommended checks:
        - Class balance
        - Feature preprocessing (numeric/categorical)
        - Optional scaling

- train_classifier(X, y, model_list=None, optimise=False, test_size=0.2, random_state=2025):
    - Fits one or more classification models
    - Returns models, evaluation metrics, and train/test sets
    - Prints accuracy, precision, recall and F1-score

- validate_model(model, X, y, folds=5, scoring='accuracy'):
    - Cross-validation on any fitted or unfitted classifiers
    - Returns 5 folds, and provides a mean and standard deviation for cross validation
    - Summarises results with print function

- plot_roc_auc(model, X_test, y_test, labels=None, plot=True):
    - Computes roc_auc
    - Option generation of ROC_AUC curve plot
    - Auto-detects positive class
    - Returns AUC score numerically.

Author: Matty Hood
Created: 26/11/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



import warnings
warnings.filterwarnings("ignore")

# HELP TEXT
def classification_help():
    """
    Help Guide for Classification Tools in PDSS

    Provides step-by-step instructions for preparing data,
    splitting into training/testing sets, training classifiers,
    and evaluating model performance.

    Usage:
        classification_help()
    """
    text = """
Classification Models
---------------------
Steps for using classification tools in PDSS

1. Prepare your dataframe
   - Identify features (X) and target (y)
   - Ensure correct column types (numeric for X, categorical for y)
   - Handle missing values (drop, fill, encode)
   - Optional: Scale numeric features for algorithms like SVM or KNN

2. Check class balance
   - Use y.value_counts() to inspect distribution
   - Consider balancing techniques if heavily skewed

3. Train/test split
   - Handled internally by train_classifier
   Example:
   result = train_classifier(X, y)
   model = result['model']
   metrics = result['metrics']
   X_train, X_test = result['X_train'], result['X_test']
   y_train, y_test = result['y_train'], result['y_test']

4. Choose models
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - AdaBoost
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)

5. Hyperparameter optimisation
   - optimise=False -> Cross-validation only
   - optimise=True  -> RandomizedSearchCV with sensible defaults

6. Evaluate models
   Metrics for each classifier:
   - Accuracy
   - Precision
   - Recall
   - F1-score

7. Optional visual diagnostics
   - Confusion matrix
   - ROC-AUC plot

"""
    print(text)


classifier_map = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC
}


default_param_grids = {
    "LogisticRegression": {'C':[0.01,0.1,1,10,100],'penalty':['l2'],'solver':['lbfgs'],'max_iter':[100,200]},
    "RandomForestClassifier": {'n_estimators':[100,200],'max_depth':[None,5,10],'min_samples_split':[2,5],'min_samples_leaf':[1,2],'max_features':['sqrt','log2',None]},
    "GradientBoostingClassifier": {'n_estimators':[100,200],'learning_rate':[0.01,0.05],'max_depth':[3,5],'subsample':[0.8,1.0]},
    "AdaBoostClassifier": {'n_estimators':[50,100],'learning_rate':[0.01,0.1,0.5]},
    "KNeighborsClassifier": {'n_neighbors':[3,5,7],'weights':['uniform','distance'],'p':[1,2]},
    "SVC": {'C':[0.1,1,10],'kernel':['linear','rbf'],'gamma':['scale','auto'],'probability':[True]}
}


def train_classifier(X, y, model_name="RandomForestClassifier", optimise=False, all_models=False, test_size=0.2, random_state=2025, n_jobs=4, labels=None, n_iter = 10):
    """
    Train and evaluate classification model(s) in PDSS.

    Parameters:
        X : DataFrame
        y : Series
        model_name : str
            Name of classifier to train (ignored if all_models=True)
        optimise : bool
            If True, runs RandomizedSearchCV for rough hyperparameter optimisation
        all_models : bool
            If True, trains all default classifiers and prints metrics for comparison
        test_size : float
            Fraction of data used for test set
        random_state : int
            Seed for reproducibility
        n_jobs : int
            Parallel jobs for RandomizedSearchCV
        labels : list, optional
            Labels for evaluation and confusion matrix. If None, inferred from y

    Returns:
        dict of trained models and their metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train ALL models
    if all_models:
        results = {}

        for name, cls in classifier_map.items():
            print(f"\nTraining {name} ...")
            model = cls()

            if optimise and name in default_param_grids:
                param_grid = default_param_grids[name]
                max_iter = int(np.prod([len(v) for v in param_grid.values()]))
                n_iter_eff = min(n_iter, max_iter)

                rs = RandomizedSearchCV(
                    model, param_distributions=param_grid,
                    n_iter=n_iter_eff, cv=3, random_state=random_state,
                    n_jobs=n_jobs
                )
                rs.fit(X_train, y_train)
                model = rs.best_estimator_

                print(f"Best params for {name}: {rs.best_params_}")

            else:
                model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, labels)
            results[name] = {"model": model, "metrics": metrics}

        return results

    # Train single model
    if model_name not in classifier_map:
        raise ValueError(f"Unknown model_name {model_name}.")

    model = classifier_map[model_name]()

    if optimise and model_name in default_param_grids:
        param_grid = default_param_grids[model_name]
        max_iter = int(np.prod([len(v) for v in param_grid.values()]))
        n_iter_eff = min(n_iter, max_iter)

        rs = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=n_iter_eff, cv=3, random_state=random_state,
            n_jobs=n_jobs
        )
        rs.fit(X_train, y_train)
        model = rs.best_estimator_

        print(f"Best params for {model_name}: {rs.best_params_}")
    else:
        model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, labels)

    return {
        "model": model,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

def evaluate_model(model, X_train, X_test, y_train, y_test, labels=None):
    """
    Compute metrics and plot confusion matrix for a trained classifier.
    """

    if labels is None:
        labels = sorted(y_test.unique())  # auto-detect labels

    binary_class = len(labels) == 2
    pos_label = labels[-1] if binary_class else None

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Basic metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)

    if binary_class:
        precision = precision_score(y_test, y_pred_test, pos_label=pos_label)
        recall    = recall_score(y_test, y_pred_test, pos_label=pos_label)
        f1        = f1_score(y_test, y_pred_test, pos_label=pos_label)
    else:
        precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        recall    = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        f1        = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

    # Print summary
    print("-"*30)
    print(f"Train Accuracy : {train_acc:.2%}")
    print(f"Test Accuracy  : {test_acc:.2%}")
    print(f"Precision ({labels[-1]}) : {precision:.2%}")
    print(f"Recall    ({labels[-1]}) : {recall:.2%}")
    print(f"F1-Score  ({labels[-1]}) : {f1:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model.__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def validate_model(model, X, y, folds=5, scoring='accuracy'):
    """
    Performs cross-validation on any fitted or unfitted classifier.

    Parameters:
        model : sklearn estimator
        X : DataFrame or array-like
        y : Series or array-like
        folds : int (default 5)
            Number of CV folds.
        scoring : str (default 'accuracy')
            Any valid sklearn scoring string.

    Returns:
        dict with cross-validation mean, std, and raw fold scores.
    """

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=folds, scoring=scoring)

    acc_mean = scores.mean()
    acc_std = scores.std()

    # Interpret mean
    if acc_mean < 0.5:
        mean_interpretation = 'Unusable'
    elif acc_mean <= 0.7:
        mean_interpretation = 'Possibly usable, can be improved'
    else:
        mean_interpretation = 'Good model fit'

    # Interpret variance
    if acc_std < 0.05:
        std_interpretation = 'Good stability'
    elif acc_std < 0.1:
        std_interpretation = 'Moderate variation — investigate'
    else:
        std_interpretation = 'High variation — possible instability'

    # Print results
    print("Cross-validation fold scores:", scores)
    print(f"Mean: {acc_mean:.3f} — {mean_interpretation}")
    print(f"Std:  {acc_std:.3f} — {std_interpretation}")

    return {
        "cv_scores": scores,
        "mean": acc_mean,
        "std": acc_std,
        "mean_interpretation": mean_interpretation,
        "std_interpretation": std_interpretation
    }

def plot_roc_auc(model, X_test, y_test, labels=None, plot=True):
    """
    Compute and optionally plot ROC-AUC for a classifier.

    Parameters:
        model : trained classifier
        X_test : test feature data
        y_test : true labels
        labels : list or None
            If None, inferred from y_test
        plot : bool (default True)
            Whether to plot the ROC curve

    Returns:
        float : ROC-AUC score
    """

    # Infer labels
    if labels is None:
        labels = sorted(np.unique(y_test))

    # ROC-AUC requires binary classification
    if len(labels) != 2:
        raise ValueError("ROC-AUC is only defined for binary classification problems.")

    pos_label = labels[-1]

    # --- Get model probability or decision scores ---
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, -1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if scores.ndim > 1:
            scores = scores[:, -1]
        y_scores = scores
    else:
        raise ValueError(
            f"Model {model.__class__.__name__} does not support probability estimation or decision scores."
        )

    # --- Compute ROC-AUC ---
    auc = roc_auc_score(y_test, y_scores)

    # --- Optional plotting ---
    if plot:
        fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=pos_label)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model.__class__.__name__}")
        plt.legend()
        plt.show()

    return auc
