# Predictive Maintenance - Capstone 3 (Interim)

MLOps pipeline for predicting machine failure using the AI4I 2020 Predictive
Maintenance dataset (UCI ML Repository, dataset id 601).

## Layout
- `data/` - registered dataset (`ai4i2020.csv`), cleaned/train/test CSVs
- `src/validate_dataset.py` - dataset registration & validation summary
- `src/prepare_data.py` - cleaning + stratified train/test split
- `src/train.py` - GridSearchCV tuning across multiple algorithms; logs to
  `models/metrics.json`, saves `models/best_model.joblib`
- `.github/workflows/pipeline.yml` - CI: validate -> prepare -> split artifact ->
  train -> model artifact + conditional commit to `main`
- `build_notebook.py` and `build_report.py` - regenerate submission evidence
- `notebooks/Predictive_Maintenance_Notebook.ipynb` - EDA + full interim workflow
- `submission/` - final PDF report and executed HTML notebook for upload

## Run locally
```bash
pip install -r requirements.txt
python src/validate_dataset.py
python src/prepare_data.py
python src/train.py
```

On macOS, XGBoost may require the OpenMP runtime (`brew install libomp`).
The GitHub Actions runner uses Linux. To use the CI workflow, push this master
folder as the root of a GitHub repository on `main` and enable GitHub Actions
write permissions so the job can commit the model. The hosted workflow has
not yet been run.
