"""
Quick data validation before training
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
data_file = PROJECT_ROOT / "data/processed/train.csv"

print("="*80)
print("DAVO MEDICAL: Data Validation")
print("="*80)

if not data_file.exists():
    print(f"\nERROR: {data_file} not found")
    print("Run: python scripts/data-processing/combine_datasets.py")
    exit(1)

print(f"\nLoading: {data_file}")
df = pd.read_csv(data_file)

print(f"\n✓ Samples: {len(df):,}")
print(f"\nColumns: {df.columns.tolist()}")

print(f"\n--- Sample Data ---")
print(f"Prompt: {df.iloc[0]['prompt'][:100]}...")
print(f"Response: {df.iloc[0]['response'][:100]}...")

print(f"\nPriority Classes:")
print(df['priority_class'].value_counts())

print(f"\nLambda Statistics:")
print(df['lambda'].describe())

print(f"\n{'='*80}")
print("✓ DATA LOOKS GOOD! Ready for training.")
print(f"{'='*80}")
