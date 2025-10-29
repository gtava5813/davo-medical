# File: inspect_mimic_columns.py

import pandas as pd
from pathlib import Path

HOME = Path.home()
RAW_DIR = HOME / "ai-env" / "data" / "mimic-iii-clean" / "raw"

print("Inspecting MIMIC-III CHARTEVENTS columns...")
print("="*80)

# Load just the first few rows to see structure
chartevents = pd.read_csv(RAW_DIR / "CHARTEVENTS.csv", nrows=10)

print("\nColumn names:")
print(chartevents.columns.tolist())

print("\nFirst 3 rows:")
print(chartevents.head(3))

print("\nData types:")
print(chartevents.dtypes)

print("\n" + "="*80)
print("Also checking D_ITEMS (item definitions)...")
d_items = pd.read_csv(RAW_DIR / "D_ITEMS.csv", nrows=10)
print("\nD_ITEMS columns:")
print(d_items.columns.tolist())
print("\nFirst 3 rows:")
print(d_items.head(3))
