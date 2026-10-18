"""Render the interim business report using executed notebook figures and metrics."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "ManjunathaNS_PredictiveMaintenance_InterimReport.pdf"
FIGURES = ROOT / "reports" / "figures"
metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
raw = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=20, leading=25, textColor=colors.HexColor("#163752"), spaceAfter=17))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"], fontSize=9.5, leading=14,
    spaceAfter=10))
styles.add(ParagraphStyle(
    name="CaptionCustom", parent=styles["BodyText"], fontSize=8.5, leading=11,
    alignment=TA_CENTER, textColor=colors.HexColor("#264963"), spaceAfter=11))
styles.add(ParagraphStyle(
    name="SubCustom", parent=styles["Heading2"], textColor=colors.HexColor("#167478"),
    spaceBefore=13, spaceAfter=8))

story = []


def title(text):
    story.append(Paragraph(escape(text), styles["TitleCustom"]))


def section(text):
    story.append(Paragraph(escape(text), styles["Heading1"]))


def sub(text):
    story.append(Paragraph(escape(text), styles["SubCustom"]))


def body(text):
    story.append(Paragraph(text, styles["BodyCustom"]))


def figure(name, number, caption, width=6.3):
    from reportlab.lib.utils import ImageReader
    path = FIGURES / name
    iw, ih = ImageReader(str(path)).getSize()
    image = Image(str(path), width=width * inch, height=width * inch * ih / iw)
    image.hAlign = "CENTER"
    story.append(KeepTogether([
        image,
        Paragraph(f"Figure {number}. {escape(caption)}", styles["CaptionCustom"]),
    ]))


def table(number, caption, rows, widths=None):
    t = Table([[str(v) for v in row] for row in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17485a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.HexColor("#e9f2f0")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), .25, colors.lightgrey),
    ]))
    story.append(KeepTogether([
        t, Spacer(1, 5),
        Paragraph(f"Table {number}. {escape(caption)}", styles["CaptionCustom"]),
    ]))


title("Predictive Maintenance | Interim Project Report")
body("<b>Capstone Project 3 | Learner:</b> ManjunathaNS<br/>"
     "<b>Scope:</b> Data registration through experiment tracking<br/>"
     "<b>Dataset:</b> AI4I 2020 Predictive Maintenance, UCI Machine Learning Repository")
section("Executive overview")
body("Unexpected equipment breakdowns disrupt production and create avoidable repair "
     "costs. This interim study builds a binary classifier for machine failure using "
     "operational sensor readings. The source dataset is synthetic, so measured "
     "performance is a research benchmark, not evidence of field deployment readiness.")
best = metrics[metrics["_best_model"]]
body(f"Of six tuned algorithms, Bagging achieved the best five-fold training cross-validation "
     f"F1 ({best['cv_best_f1']:.3f}). On the held-out test set it reached F1 "
     f"{best['test_f1']:.3f}, precision {best['test_precision']:.3f} and recall "
     f"{best['test_recall']:.3f}. It missed {best['confusion_matrix'][1][0]} of "
     f"68 failures; operators should treat this as a prioritisation aid, not an "
     "automatic shutdown decision.")

section("1. Data Registration")
sub("Methodology")
body("Registered the UCI AI4I 2020 CSV in the repository data folder and checked for "
     "all 14 expected columns using the validation script. Recorded shape, types, "
     "missing-value counts, duplicate count and target distribution before cleaning.")
table(1, "Dataset registration and quality summary", [
    ["Measure", "Result"], ["Observations", "10,000"],
    ["Columns", "14 (5 sensors, Type, 2 IDs, target, 5 failure flags)"],
    ["Missing values / duplicate rows", "0 / 0"],
    ["Machine failures / non-failures", "339 / 9,661"],
    ["Failure prevalence", "3.39%"],
], widths=[2.45*inch, 3.95*inch])
body("Source: UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance "
     "Dataset (dataset 601): "
     "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset. "
     "The data describes simulated industrial milling measurements; failure-mode flags "
     "record labels rather than deployable real-time sensor inputs.")

section("2. Exploratory Data Analysis")
sub("Methodology: background and data overview")
body("Inspected numeric summaries, target frequencies, sensor distributions, differences "
     "across failure labels and machine types, pairwise correlations and two-sensor "
     "relationships. Analyses below are descriptive associations, not causal claims.")
figure("fig_target.png", 1, "Failure versus non-failure record counts.", 4.0)
body("Only 3.39% of observations are failures. A model that always predicts no "
     "failure would achieve 96.61% accuracy while missing every breakdown. "
     "Consequently, positive-class recall and F1 are central business measures.")
figure("fig_univariate.png", 2, "Univariate sensor measurements and machine type.", 6.0)
body("Machine type L has 6,000 of 10,000 records; M has 2,997 and H has 1,003. "
     "Sensor distributions span varied operating conditions; monitoring should "
     "check for drift in machine mix as well as in the numeric readings.")
story.append(PageBreak())
sub("Methodology: bivariate and multivariate comparisons")
body("Compared each numeric sensor across target groups; aggregated failure rates "
     "by machine type; inspected correlations and torque/tool-wear interactions. "
     "Failure-mode labels are presented only for context, never used as predictors.")
figure("fig_bivariate_box.png", 3, "Sensor distributions conditional on failure.", 6.4)
table(2, "Failure rate by machine type", [
    ["Type", "Records", "Failures", "Failure rate"],
    ["H", "1,003", "21", "2.09%"],
    ["L", "6,000", "235", "3.92%"],
    ["M", "2,997", "83", "2.77%"],
], widths=[1.4*inch]*4)
body("Type L has the highest observed failure rate (3.92%), but this alone does "
     "not justify attributing failures to machine grade: operating conditions "
     "and the synthetic generation process can confound comparisons.")
figure("fig_corr_heatmap.png", 4, "Correlations among sensors and the failure target.", 5.7)
body("Air and process temperatures covary strongly, so individual sensor effects "
     "must be interpreted cautiously. Pairwise correlation can also miss "
     "interactions relevant to overstrain or power-related failure.")
figure("fig_torque_wear.png", 5, "Torque and tool wear, coloured by failure.", 5.6)
body("The joint plot suggests operational context matters: no single torque or "
     "wear threshold is sufficiently reliable. Maintenance triage should use "
     "the full model score and prioritise validation on real production data.")

section("3. Data Preparation")
sub("Methodology")
body("Loaded the repository CSV directly. Removed 0 duplicate or missing rows, "
     "discarded UDI and Product ID identifiers, and removed TWF, HDF, PWF, OSF "
     "and RNF because they encode the outcome and would leak information into "
     "prediction. Normalised names and used a reproducible stratified 80/20 split "
     "(random seed 42). Standardisation and one-hot encoding occur within the "
     "model pipeline, fitted separately inside each training fold to avoid leakage.")
table(3, "Prepared data and local workflow artifacts", [
    ["Split", "Rows", "Failures", "Stored as"],
    ["Train", "8,000", "271", "data/train.csv"],
    ["Test", "2,000", "68", "data/test.csv"],
], widths=[1.2*inch, 1.0*inch, 1.0*inch, 3.2*inch])
body("In GitHub Actions, the first job validates/prepares data and uploads "
     "train.csv and test.csv as 'train-test-splits'; the second job downloads "
     "that artifact before tuning, then uploads and conditionally commits "
     "the selected model and metrics. This workflow is configured locally; "
     "a successful hosted run has not yet been verified.")

section("4. Model Building and Experimentation Tracking")
sub("Methodology")
body("Trained Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting "
     "and XGBoost with five-fold GridSearchCV optimising F1 on training data. "
     "The preprocessing transformer is part of each estimator pipeline. "
     "Selected the model by cross-validation F1, then reported held-out "
     "accuracy, precision, recall, F1 and ROC-AUC. Complete tuned parameters "
     "and confusion matrices are recorded in models/metrics.json.")
rows = [["Model", "CV F1", "Test precision", "Test recall", "Test F1", "ROC-AUC"]]
for name, m in metrics.items():
    if name.startswith("_"):
        continue
    rows.append([name.replace("_", " ").title(), *(f"{m[k]:.3f}" for k in (
        "cv_best_f1", "test_precision", "test_recall", "test_f1", "test_roc_auc"))])
table(4, "Five-fold tuning and held-out test results", rows,
      widths=[1.35*inch] + [1.03*inch]*5)
figure("fig_model_comparison.png", 6, "Held-out F1 comparison (not selection criterion).", 5.6)
figure("fig_confusion_matrix.png", 7, "Bagging model: held-out confusion matrix.", 4.4)
body("Of 68 actual failures, the selected model detected 49 and missed 19, "
     "with 8 false alarms among 1,932 non-failures. A missed failure may cost "
     "more than an inspection triggered by a false alarm, so threshold choice "
     "should follow a measured cost-of-downtime analysis. Accuracy alone "
     "does not reflect this trade-off.")
figure("fig_feature_importance.png", 8,
       "Permutation importance measured by change in F1 on held-out data.", 5.3)
body("Permutation importance shows association with predictive performance, "
     "not a causal effect; correlated sensors can divide their apparent "
     "contribution. Do not remove a low-ranked sensor without retraining.")

section("5. Interim Business Implications and Limitations")
body("The workflow establishes a reproducible benchmark and tracks all tuned "
     "settings, model metrics and a serialised model artifact. It does not "
     "demonstrate actual avoided downtime or operational cost savings. The "
     "dataset is synthetic and the split is random rather than chronological; "
     "validate on time-separated real equipment records before any deployment. "
     "Next phase: define an alert threshold and intervention costs, assess "
     "calibration and drift, run the hosted CI workflow and pilot with "
     "maintenance engineers.")

section("Appendix A. Reproducibility and Supporting Outputs")
body("The separately submitted executed HTML notebook contains the full Python "
     "code, charts, summary tables, tuned parameter printouts and model logs. "
     "Repository scripts: src/validate_dataset.py, src/prepare_data.py, "
     "src/train.py; CI job definitions: .github/workflows/pipeline.yml. "
     "Outputs: data/cleaned.csv, data/train.csv, data/test.csv, "
     "models/metrics.json and models/best_model.joblib. "
     "These artifacts provide the full evidence without raw code in the report body.")
for name, m in metrics.items():
    if name.startswith("_"):
        continue
    body(f"<b>{escape(name)}:</b> tuned parameters {escape(str(m['best_params']))}; "
         f"CV F1={m['cv_best_f1']:.4f}; test F1={m['test_f1']:.4f}; "
         f"confusion matrix={escape(str(m['confusion_matrix']))}.")
body("<b>Source and citation:</b> Matzka, S. (2020). AI4I 2020 Predictive "
     "Maintenance Dataset. UCI Machine Learning Repository. "
     "https://doi.org/10.24432/C5HS5C. All figures and tables in this "
     "report are generated from that dataset and this project's notebook.")


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#45666b"))
    canvas.drawString(44, 29, "ManjunathaNS | Predictive Maintenance Interim")
    canvas.drawRightString(568, 29, f"Page {document.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=(612, 792), rightMargin=44, leftMargin=44,
    topMargin=47, bottomMargin=48, title="Predictive Maintenance Interim Report",
    author="ManjunathaNS",
)
doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
print(f"Wrote {OUTPUT}")
