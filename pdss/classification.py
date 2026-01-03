"""
Python Data Science Summary Tools (PDSS)
classification.py

A collection of simple, practical utilities for classification modelling,
evaluation, and visual diagnostics. Designed to support exploratory analysis,
model benchmarking, and decision-focused evaluation for real-world problems
such as churn prediction and DNA (Did Not Attend) risk.

Core design principles:
- Simple, readable workflows
- Sensible defaults for classification
- Emphasis on recall and decision thresholds for imbalanced problems
- Diagnostic metrics separated from operational decision metrics

Features
--------

classification_help()
    - Step-by-step guidance for classification workflows in PDSS
    - Data preparation and class balance checks
    - Supported models and when to use them
    - Explanation of stratification, optimisation, threshold tuning, and ROC-AUC

train_classifier(X, y, ...)
    - Train a single classifier or benchmark multiple classifiers
    - Stratified train test splitting by default
    - Optional hyperparameter optimisation using RandomizedSearchCV
    - Optional threshold tuning using a validation split
    - Prints accuracy, precision, recall, F1-score
    - Prints ROC-AUC as a ranking diagnostic for binary problems
    - Returns trained model, metrics, and train test splits

evaluate_model(model, X_train, X_test, y_train, y_test, ...)
    - Evaluates a trained classifier at a chosen probability threshold
    - Supports probability based predictions when available
    - Computes accuracy, precision, recall, F1-score
    - Optionally prints ROC-AUC for binary classification
    - Displays confusion matrix for interpretability

validate_model(model, X, y, ...)
    - Performs cross-validation on fitted or unfitted classifiers
    - Supports stratified folds for classification
    - Returns mean and standard deviation of scores
    - Provides simple interpretability guidance for stability

plot_roc_auc(model, X_test, y_test, ...)
    - Computes ROC-AUC for binary classification
    - Optionally plots the ROC curve
    - Intended as a diagnostic and comparison tool
    - Not intended for threshold selection

Intended usage
--------------
PDSS classification tools are designed for workflows where:
- Class imbalance is common
- Missing positive cases is more costly than false positives
- Threshold selection is a key decision step
- Model ranking quality (ROC-AUC) is used as a diagnostic, not an objective

Author: Matty Hood
Created: 26/11/2025
Last updated: 03/01/2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
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
PDSS Classification Models
--------------------------
Steps for using classification tools in PDSS

1. Prepare your data
   - Identify features (X) and target (y)
   - X should contain numeric columns and any encoded categorical columns
   - y should be the target labels (binary is most common for churn, DNA, etc.)
   - Handle missing values (drop, fill, encode)
   - Optional: Scale numeric features for algorithms like SVC or KNN

2. Check class balance
   - Use y.value_counts() to inspect distribution
   - If heavily imbalanced, accuracy can be misleading
   - For churn or DNA problems, recall is often more important than accuracy

3. Train/test split (stratification)
   - train_classifier can stratify splits so both sets keep similar class balance
   - Recommended for classification, especially imbalanced data
   - Use stratify=True (recommended default)

4. Train models (single or all models)
   Example single model:
       result = train_classifier(X, y, model_name="RandomForestClassifier")

   Example quick benchmark:
       results = train_classifier(X, y, all_models=True)

   Supported models:
   - LogisticRegression
   - RandomForestClassifier
   - GradientBoostingClassifier
   - AdaBoostClassifier
   - KNeighborsClassifier
   - SVC

5. Hyperparameter optimisation (RandomizedSearchCV)
   - optimise=False -> Fit with default parameters
   - optimise=True  -> RandomizedSearchCV using sensible default grids

   Tip:
   - Choose a scoring metric that matches your goal
   - For imbalanced problems, consider recall or f1 rather than accuracy

6. Threshold tuning (recommended for binary problems)
   - Some models output probabilities (predict_proba) or decision scores
   - By default, model.predict uses an internal cutoff (often similar to 0.5)
   - Threshold tuning chooses a better cutoff for your goal

   PDSS workflow:
   - Fit (and optionally optimise) model on training data
   - Tune threshold on a validation split taken from training data
   - Evaluate once on the test set using the tuned threshold

   Why this matters:
   - High ROC-AUC with low recall often means the model ranks well but needs a different threshold

   Example:
       result = train_classifier(
           X, y,
           model_name="AdaBoostClassifier",
           optimise=True,
           stratify=True,
           threshold_tune=True,
           threshold_objective="recall"
       )

   Objectives for threshold tuning:
   - "recall"     -> catch as many positives as possible
   - "f1"         -> balance precision and recall
   - "precision"  -> reduce false positives
   - "min_cost"   -> minimise weighted FP/FN cost (advanced)

7. Model validation (cross-validation)
   - validate_model runs cross-validation for a chosen scoring metric
   Example:
       validate_model(model, X, y, folds=5, scoring="recall")

8. ROC-AUC (interpretation)
   - ROC-AUC measures ranking quality, not thresholded performance
   - It answers: "Do positives tend to receive higher scores than negatives?"
   - ROC-AUC does not change when you change the threshold
   - Use ROC-AUC as a diagnostic and comparison metric, not a threshold decision tool

   ROC-AUC rough guide:
   - 0.50 -> random ranking
   - 0.60 to 0.70 -> weak but possibly useful
   - 0.70 to 0.85 -> good ranking signal
   - 0.90+ -> very strong signal (or potential leakage)

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

def get_scores(model, X):
    """
    Returns a continuous score for the positive class.
    Prefers predict_proba, falls back to decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, -1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if hasattr(scores, "ndim") and scores.ndim > 1:
            scores = scores[:, -1]
        # Map to 0 to 1 range using logistic transform for thresholding convenience
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError(f"Model {model.__class__.__name__} does not support predict_proba or decision_function.")


def threshold_optimiser(
    y_true,  # true binary labels
    y_score, # continuous scores for positive class (get_scores(model, X))
    objective: str = "recall",          # "recall" | "f1" | "precision" | "min_cost"
    step: float = 0.01,
    constraint: dict | None = None,     # example: {"precision_min": 0.8, "recall_min": 0.5}
    costs: dict | None = None           # example: {"fp": 1.0, "fn": 5.0}
):
    """
    Sweep thresholds and return best threshold plus a table you can turn into a DataFrame.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thresholds = np.arange(0.0, 1.0 + step, step)
    rows = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        cost = None
        if costs is not None:
            cost = float(costs.get("fp", 1.0)) * fp + float(costs.get("fn", 1.0)) * fn

        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "cost": None if cost is None else float(cost),
        })

    filtered = rows
    if constraint:
        if "precision_min" in constraint:
            pmin = float(constraint["precision_min"])
            filtered = [r for r in filtered if r["precision"] >= pmin]
        if "recall_min" in constraint:
            rmin = float(constraint["recall_min"])
            filtered = [r for r in filtered if r["recall"] >= rmin]

    if not filtered:
        return 0.5, rows, {"reason": "No thresholds satisfied constraints"}

    if objective == "recall":
        best = max(filtered, key=lambda r: r["recall"])
    elif objective == "f1":
        best = max(filtered, key=lambda r: r["f1"])
    elif objective == "precision":
        best = max(filtered, key=lambda r: r["precision"])
    elif objective == "min_cost":
        if costs is None:
            raise ValueError("objective='min_cost' requires costs, eg {'fp': 1.0, 'fn': 5.0}")
        best = min(filtered, key=lambda r: r["cost"])
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return float(best["threshold"]), rows, best


def train_classifier(
    X, y,
    model_name="RandomForestClassifier",
    optimise=False, # whether to do basic hyperparameter optimisation
    all_models=False, # whether to train all default classifiers
    test_size=0.2, # fraction of data used for test set
    random_state=2025, # seed for reproducibility
    n_jobs=4, # parallel jobs for RandomizedSearchCV
    labels=None, # list of labels for evaluation
    n_iter=10, # number of iterations for RandomizedSearchCV
    stratify=True, # stratified splitting
    threshold_tune=False, # whether to do threshold tuning
    threshold_objective="recall", # "recall", "precision", "f1", "min_cost"
    threshold_step=0.01, 
    threshold_val_size=0.2, # fraction of train set for threshold tuning
    threshold_constraint=None, # e.g., {"precision_min": 0.8, "recall_min": 0.5}
    threshold_costs=None # e.g., {"fp": 1.0, "fn": 5.0}
):
    """
    Train and evaluate classification model(s) in PDSS.
    Adds stratified splitting by default and optional threshold tuning.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    # Optional internal validation split for threshold tuning
    if threshold_tune:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train,
            test_size=threshold_val_size,
            random_state=random_state,
            stratify=y_train if stratify else None
        )
    else:
        X_fit, y_fit = X_train, y_train
        X_val = y_val = None
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def _fit_one(name, cls):
        print(f"\nTraining {name} ...")
        model = cls()

        best_params = None

        if optimise and name in default_param_grids:
            param_grid = default_param_grids[name]
            max_iter = int(np.prod([len(v) for v in param_grid.values()]))
            n_iter_eff = min(n_iter, max_iter)

            rs = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=n_iter_eff,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs
            )
            rs.fit(X_fit, y_fit)
            model = rs.best_estimator_
            best_params = rs.best_params_
            print(f"Best params for {name}: {best_params}")
        else:
            model.fit(X_fit, y_fit)

        # Threshold tuning after optimisation
        best_threshold = 0.5
        threshold_table = None

        if threshold_tune and (y_val is not None):
            # Only valid for binary problems
            unique_labels = sorted(np.unique(y_val))
            if unique_labels == [0, 1] and (hasattr(model, "predict_proba") or hasattr(model, "decision_function")):
                y_val_score = get_scores(model, X_val)
                best_threshold, threshold_table, best_row = threshold_optimiser(
                    y_true=y_val,
                    y_score=y_val_score,
                    objective=threshold_objective,
                    step=threshold_step,
                    constraint=threshold_constraint,
                    costs=threshold_costs
                )
                print(f"[PDSS] Best threshold (objective={threshold_objective}) : {best_threshold:.3f}")
            else:
                print("[PDSS] Threshold tuning skipped (requires binary labels and probability scores).")

        # Evaluate on test at chosen threshold
        metrics = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            labels=labels,
            threshold=best_threshold
        )

        return {
            "model": model,
            "metrics": metrics,
            "best_params": best_params,
            "best_threshold": best_threshold,
            "threshold_table": threshold_table,
        }

    # Train ALL models
    if all_models:
        results = {}
        for name, cls in classifier_map.items():
            results[name] = _fit_one(name, cls)
        return results

    # Train single model
    if model_name not in classifier_map:
        raise ValueError(f"Unknown model_name {model_name}.")

    return_obj = _fit_one(model_name, classifier_map[model_name])

    # Keep your original return keys too
    return {
        **return_obj,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }



def evaluate_model(model, X_train, X_test, y_train, y_test, labels=None, threshold=0.5):
    """
    Compute metrics and plot confusion matrix for a trained classifier.
    Supports custom threshold when probabilities are available.
    """

    if labels is None:
        labels = sorted(y_test.unique())

    binary_class = len(labels) == 2
    pos_label = labels[-1] if binary_class else None 
    
    # Predictions
    used_threshold = False
    auc = None

    if binary_class and (hasattr(model, "predict_proba") or hasattr(model, "decision_function")):
        y_train_score = get_scores(model, X_train)
        y_test_score = get_scores(model, X_test)

        y_pred_train = (y_train_score >= threshold).astype(int)
        y_pred_test = (y_test_score >= threshold).astype(int)

        used_threshold = True
        try:
            auc = roc_auc_score(y_test, y_test_score)
        except Exception:
            auc = None
    else:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_test_score = None

    # Basic metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)

    if binary_class:
        precision = precision_score(y_test, y_pred_test, pos_label=pos_label, zero_division=0)
        recall    = recall_score(y_test, y_pred_test, pos_label=pos_label, zero_division=0)
        f1        = f1_score(y_test, y_pred_test, pos_label=pos_label, zero_division=0)
    else:
        precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        recall    = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        f1        = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

    # Print summary
    print("-"*30)
    if used_threshold:
        print(f"Threshold      : {threshold:.2f}")
    print(f"Train Accuracy : {train_acc:.2%}")
    print(f"Test Accuracy  : {test_acc:.2%}")
    print(f"Precision ({labels[-1]}) : {precision:.2%}")
    print(f"Recall    ({labels[-1]}) : {recall:.2%}")
    print(f"F1-Score  ({labels[-1]}) : {f1:.2%}")
    if auc is not None:
        print(f"ROC-AUC        : {auc:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model.__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    out = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if auc is not None:
        out["roc_auc"] = auc

    if used_threshold:
        out["threshold"] = threshold

    return out

def validate_model(model, X, y, folds=5, scoring='accuracy', stratify=True, random_state=2025):
    """
    Performs cross-validation on any fitted or unfitted classifier.
    """

    if stratify:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        cv = folds

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    acc_mean = scores.mean()
    acc_std = scores.std()

    if acc_mean < 0.5:
        mean_interpretation = 'Unusable'
    elif acc_mean <= 0.7:
        mean_interpretation = 'Possibly usable, can be improved'
    else:
        mean_interpretation = 'Good model fit'

    if acc_std < 0.05:
        std_interpretation = 'Good stability'
    elif acc_std < 0.1:
        std_interpretation = 'Moderate variation, investigate'
    else:
        std_interpretation = 'High variation, possible instability'

    print("Cross-validation fold scores:", scores)
    print(f"Mean: {acc_mean:.3f} ({mean_interpretation})")
    print(f"Std:  {acc_std:.3f} ({std_interpretation})")

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
