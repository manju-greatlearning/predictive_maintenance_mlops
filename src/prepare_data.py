"""Data preparation for the Predictive Maintenance pipeline.

Loads the raw CSV from data/, removes identifier columns and leakage columns
(per-failure-mode flags that encode the target), drops duplicates/missing
rows, then writes a stratified train/test split to data/train.csv and
data/test.csv so the training job can consume them as a workflow artifact.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = "Machine_failure"
# UDI / Product ID are identifiers; TWF..RNF are post-failure mode labels
# that leak the target and are not available at prediction time.
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def main() -> None:
    df = pd.read_csv(DATA_DIR / "ai4i2020.csv")
    print(f"Raw shape: {df.shape}")

    before = len(df)
    df = df.drop_duplicates().dropna()
    print(f"Dropped {before - len(df)} duplicate/missing rows")

    cleaned = df.drop(columns=DROP_COLS)
    cleaned.columns = [c.split(" [")[0].replace(" ", "_") for c in cleaned.columns]
    cleaned.to_csv(DATA_DIR / "cleaned.csv", index=False)
    print(f"Cleaned shape: {cleaned.shape}; saved to data/cleaned.csv")

    train, test = train_test_split(
        cleaned, test_size=0.2, random_state=42, stratify=cleaned[TARGET]
    )
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"train.csv: {train.shape}  (failure rate {train[TARGET].mean():.2%})")
    print(f"test.csv : {test.shape}  (failure rate {test[TARGET].mean():.2%})")


if __name__ == "__main__":
    main()
