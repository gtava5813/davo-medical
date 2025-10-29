#!/usr/bin/env python3
"""Rebalance training data to prevent LOW priority bias"""

import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/train.csv")

print("Original distribution:")
print(df['priority_class'].value_counts())

# Balance classes by oversampling CRITICAL/HIGH
critical = df[df['priority_class'] == 'CRITICAL']
high = df[df['priority_class'] == 'HIGH']
medium = df[df['priority_class'] == 'MEDIUM']
low = df[df['priority_class'] == 'LOW']

# Oversample to match LOW count
target_size = len(low)

critical_resampled = critical.sample(n=target_size, replace=True, random_state=42)
high_resampled = high.sample(n=target_size, replace=True, random_state=42)
medium_resampled = medium.sample(n=target_size, replace=True, random_state=42)

# Combine
balanced_df = pd.concat([critical_resampled, high_resampled, medium_resampled, low])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced distribution:")
print(balanced_df['priority_class'].value_counts())

# Save
balanced_df.to_csv("data/processed/train_balanced.csv", index=False)
print(f"\n✓ Saved to data/processed/train_balanced.csv ({len(balanced_df)} samples)")
