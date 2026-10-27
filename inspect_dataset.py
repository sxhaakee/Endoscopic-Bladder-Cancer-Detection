"""
inspect_dataset.py — Phase 2: Inspect 'bladder data.csv' before any training.
Run: python3 inspect_dataset.py
"""
import pandas as pd
import sys, json

CSV_PATH = "bladder data.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    sys.exit(f"ERROR: '{CSV_PATH}' not found. Place it in the same folder.")

print("=" * 60)
print("  DATASET INSPECTION REPORT")
print("=" * 60)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n  {list(df.columns)}")

print("\n--- Data Types ---")
print(df.dtypes.to_string())

print("\n--- Target column: Disease ---")
if "Disease" not in df.columns:
    print("ERROR: 'Disease' column not found!")
else:
    vc = df["Disease"].value_counts()
    print(vc.to_string())
    print(f"\nTotal labelled rows: {vc.sum()}")
    print(f"Classes: {sorted(df['Disease'].unique())}")
    null_target = df["Disease"].isnull().sum()
    print(f"Missing target values: {null_target}")

print("\n--- Missing values per column (non-zero only) ---")
miss = df.isnull().sum()
miss = miss[miss > 0]
if miss.empty:
    print("  No missing values found.")
else:
    print(miss.to_string())

print("\n--- Duplicate rows ---")
dups = df.duplicated().sum()
print(f"  Duplicate rows: {dups}")

if "Patient Number" in df.columns:
    pid_dups = df["Patient Number"].duplicated().sum()
    print(f"  Duplicate Patient Numbers: {pid_dups}")
    print(f"  Unique Patient Numbers: {df['Patient Number'].nunique()}")

print("\n--- Sample row (index 0) ---")
print(df.iloc[0].to_string())

print("\n--- Numeric column summary ---")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if "Patient Number" in numeric_cols:
    numeric_cols.remove("Patient Number")
print(df[numeric_cols].describe().to_string())

print("\n--- Categorical columns ---")
cat_cols = df.select_dtypes(include="object").columns.tolist()
if "Disease" in cat_cols:
    cat_cols.remove("Disease")
for c in cat_cols:
    print(f"  {c}: {sorted(df[c].dropna().unique())}")

print("\n" + "=" * 60)
print("  Inspection complete.")
print("=" * 60)
