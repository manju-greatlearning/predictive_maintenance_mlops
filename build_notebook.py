"""Build the interim notebook programmatically with nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

md("""# Predictive Maintenance - Interim Report Notebook
**Capstone Project 3 | ManjunathaNS**

Dataset: **AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository (dataset id 601, https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). 10,000 synthetic-but-realistic milling-machine records with process telemetry (air/process temperature, rotational speed, torque, tool wear) and a binary `Machine failure` target plus per-mode failure flags (TWF, HDF, PWF, OSF, RNF).

Business goal: predict machine failure ahead of time so maintenance can be scheduled proactively, reducing unplanned downtime and repair cost.""")

md("## 1. Data Registration and Validation\nLoad the registered dataset from the repository `data/` folder, verify expected columns, and print a registration summary.")
code("""import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("../data")
df_raw = pd.read_csv(DATA_DIR / "ai4i2020.csv")

expected = ["UDI","Product ID","Type","Air temperature [K]","Process temperature [K]",
            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Machine failure",
            "TWF","HDF","PWF","OSF","RNF"]
missing = [c for c in expected if c not in df_raw.columns]
assert not missing, f"Missing columns: {missing}"
print(f"Registered dataset: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print("All expected columns present:", len(expected))
df_raw.head()""")

code("""df_raw.info()
df_raw.describe().round(2)""")

code("""# Registration summary: missing values, duplicates, class balance
print("Missing values per column:")
print(df_raw.isna().sum().to_string())
print("\\nDuplicate rows:", df_raw.duplicated().sum())
print("\\nTarget distribution:")
print(df_raw["Machine failure"].value_counts().to_string())
print("Failure rate: {:.2%}".format(df_raw["Machine failure"].mean()))""")

md("""## 2. Exploratory Data Analysis

### 2.1 Univariate analysis
Distribution of each numeric feature, the categorical `Type`, and the target.""")
code("""num_cols = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]",
            "Torque [Nm]","Tool wear [min]"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df_raw[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(col)
sns.countplot(x="Type", data=df_raw, ax=axes.flat[5], palette="viridis")
axes.flat[5].set_title("Machine Type count")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_univariate.png", dpi=120)
plt.show()""")

code("""fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="Machine failure", data=df_raw, ax=ax, palette=["#4c72b0","#dd8452"])
ax.set_title("Target: Machine failure (3.39% positive)")
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.savefig(FIG_DIR / "fig_target.png", dpi=120)
plt.show()""")

md("""### 2.2 Bivariate analysis
Feature behaviour vs `Machine failure` (boxplots) and failure rate by machine `Type`.""")
code("""fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for ax, col in zip(axes.flat, num_cols):
    sns.boxplot(x="Machine failure", y=col, data=df_raw, ax=ax, palette="Set2")
    ax.set_title(col.split(" [")[0])
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_bivariate_box.png", dpi=120)
plt.show()""")

code("""fail_by_type = df_raw.groupby("Type")["Machine failure"].agg(["mean","sum","count"])
fail_by_type["mean"] = (fail_by_type["mean"]*100).round(2)
fail_by_type = fail_by_type.rename(columns={"mean":"failure_rate_%","sum":"failures","count":"machines"})
print(fail_by_type)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=fail_by_type.index, y=fail_by_type["failure_rate_%"], palette="viridis", ax=ax)
ax.set_ylabel("Failure rate (%)"); ax.set_title("Failure rate by machine Type")
plt.savefig(FIG_DIR / "fig_failure_by_type.png", dpi=120)
plt.show()""")

code("""# Failure mode counts
modes = ["TWF","HDF","PWF","OSF","RNF"]
mode_counts = df_raw[modes].sum().sort_values(ascending=False)
print(mode_counts.to_string())
fig, ax = plt.subplots(figsize=(7,4))
sns.barplot(x=mode_counts.index, y=mode_counts.values, palette="rocket", ax=ax)
ax.set_title("Failure events by mode (TWF/HDF/PWF/OSF/RNF)")
plt.savefig(FIG_DIR / "fig_failure_modes.png", dpi=120)
plt.show()""")

md("### 2.3 Multivariate analysis\nCorrelation heatmap and torque-vs-wear relationship coloured by failure.")
code("""plt.figure(figsize=(9,6))
corr = df_raw[num_cols + ["Machine failure"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_corr_heatmap.png", dpi=120)
plt.show()""")

code("""fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x="Torque [Nm]", y="Tool wear [min]", hue="Machine failure",
                data=df_raw, ax=ax, alpha=0.5, palette=["#4c72b0","#dd8452"])
ax.set_title("Torque vs Tool wear coloured by failure")
plt.savefig(FIG_DIR / "fig_torque_wear.png", dpi=120)
plt.show()""")

md("""### 2.4 Key EDA insights
- **Severe class imbalance**: only 3.39% of records are failures -> stratified split and recall/F1 metrics are essential; accuracy alone is misleading.
- **Mechanical indicators**: inspect the torque and tool-wear plots jointly rather than treating either feature as a deterministic failure rule; they can help prioritise inspection.
- **Temperature pair**: process temperature closely tracks air temperature; correlated predictors warrant caution when interpreting marginal effects.
- **Rotational speed + torque**: their interaction may contain more signal than a single-variable correlation; failure-mode labels are descriptive, not valid model inputs.
- **Machine Type**: L machines account for most observations and have a 3.92% failure rate versus 2.77% for M and 2.09% for H; differing base rates warrant monitoring by type.
- No missing values or duplicates -> cleaning effort is minimal; main preparation step is dropping identifiers and leakage columns.""")

md("""## 3. Data Preparation
Methodology: drop identifier columns (`UDI`, `Product ID`) and leakage columns (`TWF`..`RNF`, which encode the failure mode and are only known after failure), normalise column names, remove duplicates/missing rows, then perform a stratified 80/20 train/test split saved to `data/train.csv` and `data/test.csv`. In the CI pipeline these files are passed to the training job as a workflow artifact (`actions/upload-artifact` / `download-artifact` in `.github/workflows/pipeline.yml`).""")
code("""from sklearn.model_selection import train_test_split

df = df_raw.drop_duplicates().dropna()
df = df.drop(columns=["UDI","Product ID","TWF","HDF","PWF","OSF","RNF"])
df.columns = [c.split(" [")[0].replace(" ","_") for c in df.columns]
df.to_csv(DATA_DIR / "cleaned.csv", index=False)
print("Cleaned shape:", df.shape)
df.head()""")

code("""TARGET = "Machine_failure"
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
train.to_csv(DATA_DIR / "train.csv", index=False)
test.to_csv(DATA_DIR / "test.csv", index=False)
print(f"train {train.shape} fail-rate {train[TARGET].mean():.2%} | test {test.shape} fail-rate {test[TARGET].mean():.2%}")""")

md("""## 4. Model Building with Experimentation Tracking
Methodology: a `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for `Type`) feeds six candidate algorithms - Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost. Each is tuned with 5-fold `GridSearchCV` (scoring=F1 to respect class imbalance). All tuned parameters and test metrics are logged to `models/metrics.json` and the best pipeline is persisted to `models/best_model.joblib` for the pipeline to commit.""")
code("""from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), X_train.select_dtypes(include=[np.number]).columns.tolist()),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]),  # only categorical feature
])

candidates = {
    "decision_tree": (DecisionTreeClassifier(random_state=42),
        {"model__max_depth":[3,5,8,None],"model__min_samples_split":[2,10]}),
    "bagging": (BaggingClassifier(random_state=42, n_jobs=-1),
        {"model__n_estimators":[50,100],"model__max_samples":[0.7,1.0]}),
    "random_forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
        {"model__n_estimators":[100,200],"model__max_depth":[5,10,None]}),
    "adaboost": (AdaBoostClassifier(random_state=42),
        {"model__n_estimators":[50,100],"model__learning_rate":[0.5,1.0]}),
    "gradient_boosting": (GradientBoostingClassifier(random_state=42),
        {"model__n_estimators":[100,200],"model__max_depth":[3,5],"model__learning_rate":[0.05,0.1]}),
    "xgboost": (XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        {"model__n_estimators":[100,200],"model__max_depth":[3,5],"model__learning_rate":[0.05,0.1]}),
}

experiment_log = {}
best_name, best_f1, best_model = None, -1, None
for name, (model, grid) in candidates.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    preds = gs.predict(X_test)
    proba = gs.predict_proba(X_test)[:,1]
    rec = {"search_space":{k.replace("model__",""):v for k,v in grid.items()},
           "cv_trials":[{"params":{k.replace("model__",""):v for k,v in params.items()},
                         "mean_f1":round(float(score),4)}
                        for params,score in zip(gs.cv_results_["params"],gs.cv_results_["mean_test_score"])],
           "best_params":{k.replace("model__",""):v for k,v in gs.best_params_.items()},
           "cv_best_f1":round(gs.best_score_,4),
           "test_accuracy":round(accuracy_score(y_test,preds),4),
           "test_precision":round(precision_score(y_test,preds),4),
           "test_recall":round(recall_score(y_test,preds),4),
           "test_f1":round(f1_score(y_test,preds),4),
           "test_roc_auc":round(roc_auc_score(y_test,proba),4),
           "confusion_matrix":confusion_matrix(y_test,preds).tolist()}
    experiment_log[name]=rec
    print(f"{name:18s} F1={rec['test_f1']:.4f} recall={rec['test_recall']:.4f} params={rec['best_params']}")
    if rec["cv_best_f1"]>best_f1:
        best_name,best_f1,best_model=name,rec["cv_best_f1"],gs.best_estimator_
experiment_log["_best_model"]=best_name""")

code("""# Experimentation tracking: persist the run log and the best model
MODEL_DIR = Path("../models"); MODEL_DIR.mkdir(exist_ok=True)
(MODEL_DIR/"metrics.json").write_text(json.dumps(experiment_log, indent=2))
joblib.dump(best_model, MODEL_DIR/"best_model.joblib")

results = pd.DataFrame(experiment_log).T.drop(index="_best_model")
results[["test_accuracy","test_precision","test_recall","test_f1","test_roc_auc"]].sort_values("test_f1", ascending=False)""")

code("""fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=results["test_f1"].astype(float).values, y=results.index, palette="mako", ax=ax)
ax.set_xlabel("Test F1 score"); ax.set_title("Model comparison (test F1)")
plt.savefig(FIG_DIR / "fig_model_comparison.png", dpi=120)
plt.show()""")

code("""from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay.from_predictions(y_test, best_model.predict(X_test), ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix - {best_name}")
plt.savefig(FIG_DIR / "fig_confusion_matrix.png", dpi=120)
plt.show()
print(classification_report(y_test, best_model.predict(X_test)))""")

code("""# Model-agnostic importance on held-out data also works for Bagging.
from sklearn.inspection import permutation_importance
importance = permutation_importance(best_model, X_test, y_test, scoring="f1",
                                    n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(importance.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(7,4))
imp.plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Mean decrease in F1 on permutation")
ax.set_title(f"Permutation importance - {best_name}")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_feature_importance.png", dpi=120)
plt.show()""")

md("""## 5. Interim Findings and Next Steps
- Best model (by cross-validation F1) is saved to `models/best_model.joblib` and experiments are logged in `models/metrics.json`; the CI workflow passes train/test as an artifact, uploads the trained model artifact and conditionally commits the model to `main` when hosted write permissions are enabled.
- Review the permutation-importance chart for measured feature rankings; correlated features can share importance.
- Recall for the minority failure class is the business-critical metric: missed failures = unplanned downtime. Threshold tuning and cost-sensitive learning are candidates for the final phase.
- Next steps: deployment packaging (Gradio/HF Space app mirroring the previous capstone), monitoring, and final report.""")

from pathlib import Path
nbf.write(nb, Path(__file__).resolve().parent / "notebooks" / "Predictive_Maintenance_Notebook.ipynb")
print("notebook written")
