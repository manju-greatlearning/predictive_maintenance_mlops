"""Model training with experimentation tracking.

Loads train/test splits (produced by prepare_data.py / downloaded as a
workflow artifact), tunes several candidate algorithms with GridSearchCV,
logs every tuned parameter and metric to models/metrics.json, and saves the
best model (by cross-validation F1) to models/best_model.joblib.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
TARGET = "Machine_failure"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "str"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

candidates = {
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [3, 5, 8, None], "model__min_samples_split": [2, 10]},
    ),
    "bagging": (
        BaggingClassifier(random_state=42, n_jobs=-1),
        {"model__n_estimators": [50, 100], "model__max_samples": [0.7, 1.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [5, 10, None]},
    ),
    "adaboost": (
        AdaBoostClassifier(random_state=42),
        {"model__n_estimators": [50, 100], "model__learning_rate": [0.5, 1.0]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_depth": [3, 5],
         "model__learning_rate": [0.05, 0.1]},
    ),
    "xgboost": (
        XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [3, 5],
         "model__learning_rate": [0.05, 0.1]},
    ),
}

experiment_log = {}
best_name, best_cv_f1, best_pipeline = None, -1.0, None

for name, (model, param_grid) in candidates.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    search = GridSearchCV(
        pipe, param_grid, cv=5, scoring="f1", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    preds = search.predict(X_test)
    proba = search.predict_proba(X_test)[:, 1] if hasattr(search, "predict_proba") else None

    metrics = {
        "search_space": {k.replace("model__", ""): v for k, v in param_grid.items()},
        "cv_trials": [
            {"params": {k.replace("model__", ""): v for k, v in params.items()},
             "mean_f1": round(float(score), 4)}
            for params, score in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"]
            )
        ],
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "cv_best_f1": round(float(search.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_precision": round(float(precision_score(y_test, preds)), 4),
        "test_recall": round(float(recall_score(y_test, preds)), 4),
        "test_f1": round(float(f1_score(y_test, preds)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4) if proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    experiment_log[name] = metrics
    print(f"{name:18s} F1={metrics['test_f1']:.4f}  recall={metrics['test_recall']:.4f}  params={metrics['best_params']}")
    if metrics["cv_best_f1"] > best_cv_f1:
        best_name, best_cv_f1, best_pipeline = name, metrics["cv_best_f1"], search.best_estimator_

experiment_log["_best_model"] = best_name
(MODEL_DIR / "metrics.json").write_text(json.dumps(experiment_log, indent=2))
joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

print(f"\nBest model: {best_name} (CV F1={best_cv_f1:.4f}, test F1={experiment_log[best_name]['test_f1']:.4f})")
print(classification_report(y_test, best_pipeline.predict(X_test)))
print("Saved -> models/best_model.joblib, models/metrics.json")
