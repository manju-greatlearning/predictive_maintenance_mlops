"""Dataset registration validation for the AI4I 2020 Predictive Maintenance data.

Checks that data/ai4i2020.csv exists, contains the expected columns with
sensible dtypes, and prints a registration summary (shape, missing values,
duplicates, class balance).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

EXPECTED_COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        return 1

    df = pd.read_csv(DATA_PATH)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        print(f"ERROR: missing expected columns: {missing_cols}")
        return 1
    if extra_cols:
        print(f"WARNING: unexpected extra columns: {extra_cols}")

    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY - AI4I 2020 Predictive Maintenance")
    print("=" * 60)
    print(f"Path               : {DATA_PATH}")
    print(f"Rows x Columns     : {df.shape[0]} x {df.shape[1]}")
    print(f"Duplicate rows     : {df.duplicated().sum()}")
    print("\nColumn dtypes and missing values:")
    summary = pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
         "unique": df.nunique()}
    )
    print(summary.to_string())
    print("\nTarget distribution (Machine failure):")
    print(df["Machine failure"].value_counts(normalize=False).to_string())
    print("Failure rate: {:.2%}".format(df["Machine failure"].mean()))
    print("\nMachine type distribution:")
    print(df["Type"].value_counts().to_string())
    print("\nValidation PASSED: all expected columns present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
