"""
Combine MIMIC-III + synthetic data and balance classes
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent

print("="*80)
print("DAVO MEDICAL: Dataset Combination")
print("="*80)

# Load data
mimic_file = PROJECT_ROOT / "data/mimic-iii/mimic-iii-clean/vital_signs/mimic_davo_training.csv"
synth_file = PROJECT_ROOT / "data/synthetic/clinical_davo_synthetic.csv"
output_dir = PROJECT_ROOT / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[1/3] Loading datasets...")

if not mimic_file.exists():
    print(f"ERROR: MIMIC file not found: {mimic_file}")
    sys.exit(1)

df_mimic = pd.read_csv(mimic_file)
print(f"  MIMIC-III: {len(df_mimic):,} samples")

if synth_file.exists():
    df_synth = pd.read_csv(synth_file)
    print(f"  Synthetic: {len(df_synth):,} samples")
else:
    print(f"  WARNING: No synthetic data found. Using MIMIC only.")
    df_synth = pd.DataFrame()

print(f"\n[2/3] Combining and balancing...")

# Normalize column names
df_mimic.columns = df_mimic.columns.str.lower()
if not df_synth.empty:
    df_synth.columns = df_synth.columns.str.lower()

# Ensure priority_class column exists
if 'priority_class' not in df_mimic.columns:
    print("ERROR: priority_class column missing from MIMIC data")
    sys.exit(1)

# Sample and combine
n_mimic = min(len(df_mimic), 50000)
df_mimic_sample = df_mimic.sample(n=n_mimic, random_state=42)

if not df_synth.empty:
    n_synth = min(len(df_synth), 30000)
    df_synth_sample = df_synth.sample(n=n_synth, random_state=42)
    df_combined = pd.concat([df_mimic_sample, df_synth_sample], ignore_index=True)
else:
    df_combined = df_mimic_sample

print(f"  Combined: {len(df_combined):,} samples")

# Balance classes
critical = df_combined[df_combined['priority_class'] == 'CRITICAL']
high = df_combined[df_combined['priority_class'] == 'HIGH']
medium = df_combined[df_combined['priority_class'] == 'MEDIUM'].sample(n=min(len(high)*2, 15000), random_state=42)
low = df_combined[df_combined['priority_class'] == 'LOW'].sample(n=min(len(high)*2, 15000), random_state=42)

# Oversample critical/high
critical_over = pd.concat([critical] * 4, ignore_index=True)
high_over = pd.concat([high] * 2, ignore_index=True)

df_balanced = pd.concat([critical_over, high_over, medium, low], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Balanced: {len(df_balanced):,} samples")
print(f"\n  Priority distribution:")
print(df_balanced['priority_class'].value_counts())

print(f"\n[3/3] Saving datasets...")

# Save combined
output_file = output_dir / "davo_combined_balanced.csv"
df_balanced.to_csv(output_file, index=False)
print(f"  ✓ {output_file}")

# Train/val/test splits
train_size = int(0.8 * len(df_balanced))
val_size = int(0.1 * len(df_balanced))

df_train = df_balanced[:train_size]
df_val = df_balanced[train_size:train_size+val_size]
df_test = df_balanced[train_size+val_size:]

df_train.to_csv(output_dir / "train.csv", index=False)
df_val.to_csv(output_dir / "val.csv", index=False)
df_test.to_csv(output_dir / "test.csv", index=False)

print(f"  ✓ train.csv ({len(df_train):,} samples)")
print(f"  ✓ val.csv ({len(df_val):,} samples)")
print(f"  ✓ test.csv ({len(df_test):,} samples)")

print(f"\n{'='*80}")
print("✓ COMPLETE!")
print(f"{'='*80}")
print(f"\nNext: python scripts/training/finetune_maverick.py \\")
print(f"  --model_path ~/.llama/checkpoints/Llama-4-Maverick-17B-128E-Instruct \\")
print(f"  --data_path data/processed/train.csv \\")
print(f"  --output_dir models/fine-tuned/maverick_davo_v1")
