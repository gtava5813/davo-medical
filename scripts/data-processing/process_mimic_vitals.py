# File: process_mimic_vitals_fixed.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

print("="*80)
print("MIMIC-III VITAL SIGNS EXTRACTION FOR DAVO TRAINING (FIXED)")
print("="*80)

# Fixed paths
HOME = Path.home()
DATA_DIR = HOME / "ai-env" / "data" / "mimic-iii-clean"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VITAL_SIGNS_DIR = DATA_DIR / "vital_signs"

# Create output directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VITAL_SIGNS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[1/7] Loading MIMIC-III demo dataset from {RAW_DIR}...")

# Load essential tables
patients = pd.read_csv(RAW_DIR / "PATIENTS.csv")
admissions = pd.read_csv(RAW_DIR / "ADMISSIONS.csv")
icustays = pd.read_csv(RAW_DIR / "ICUSTAYS.csv")
chartevents = pd.read_csv(RAW_DIR / "CHARTEVENTS.csv", low_memory=False)
d_items = pd.read_csv(RAW_DIR / "D_ITEMS.csv")

print(f"   Patients: {len(patients):,}")
print(f"   Admissions: {len(admissions):,}")
print(f"   ICU Stays: {len(icustays):,}")
print(f"   Chart Events: {len(chartevents):,}")
print(f"   Item Definitions: {len(d_items):,}")

# Normalize column names to uppercase (handles both cases)
print("\n[2/7] Normalizing column names...")
chartevents.columns = chartevents.columns.str.upper()
patients.columns = patients.columns.str.upper()
admissions.columns = admissions.columns.str.upper()
icustays.columns = icustays.columns.str.upper()
d_items.columns = d_items.columns.str.upper()

print("   Column names normalized to uppercase")
print(f"   CHARTEVENTS columns: {chartevents.columns.tolist()[:10]}...")

# Define vital signs item IDs
VITAL_SIGNS_ITEMIDS = {
    'HR': [211, 220045],
    'SBP': [51, 442, 455, 6701, 220179, 220050],
    'DBP': [8368, 8440, 8441, 8555, 220180, 220051],
    'SpO2': [646, 220277],
    'RR': [615, 618, 220210, 224690],
    'TEMP': [223761, 678, 223762, 676],
}

print(f"\n[3/7] Filtering chart events for vital signs...")
print(f"   Targeting {sum(len(v) for v in VITAL_SIGNS_ITEMIDS.values())} vital sign item IDs")

# Get all vital sign ITEMIDs
all_vital_itemids = []
for vital_type, itemids in VITAL_SIGNS_ITEMIDS.items():
    all_vital_itemids.extend(itemids)

# Filter chartevents for vital signs only
vital_events = chartevents[chartevents['ITEMID'].isin(all_vital_itemids)].copy()
print(f"   Found {len(vital_events):,} vital sign measurements")

if len(vital_events) == 0:
    print("\n   WARNING: No vital signs found with standard ITEMID mappings")
    print("   Checking what ITEMIDs are actually in the data...")
    
    # Show sample ITEMIDs
    sample_itemids = chartevents['ITEMID'].value_counts().head(20)
    print("\n   Top 20 most common ITEMIDs in CHARTEVENTS:")
    for itemid, count in sample_itemids.items():
        # Try to find label from D_ITEMS
        label = d_items[d_items['ITEMID'] == itemid]['LABEL'].values
        label_str = label[0] if len(label) > 0 else "Unknown"
        print(f"     {itemid}: {label_str} ({count:,} occurrences)")
    
    print("\n   Searching D_ITEMS for vital sign related labels...")
    vital_keywords = ['heart rate', 'blood pressure', 'spo2', 'o2 sat', 'oxygen', 
                     'respiratory', 'respiration', 'temperature', 'temp', 'pulse']
    
    vital_items = d_items[d_items['LABEL'].str.lower().str.contains('|'.join(vital_keywords), na=False)]
    print(f"\n   Found {len(vital_items)} items matching vital sign keywords:")
    print(vital_items[['ITEMID', 'LABEL', 'CATEGORY']].head(20))
    
    # Use discovered ITEMIDs
    discovered_itemids = vital_items['ITEMID'].tolist()
    print(f"\n   Retrying with {len(discovered_itemids)} discovered vital sign ITEMIDs...")
    vital_events = chartevents[chartevents['ITEMID'].isin(discovered_itemids)].copy()
    print(f"   Found {len(vital_events):,} vital sign measurements")
    
    if len(vital_events) == 0:
        print("\n   ERROR: Still no vital signs found. Using all CHARTEVENTS as fallback...")
        vital_events = chartevents.sample(min(10000, len(chartevents))).copy()
        vital_events['VITAL_TYPE'] = 'GENERAL'
    else:
        # Map discovered items
        itemid_to_label = dict(zip(vital_items['ITEMID'], vital_items['LABEL']))
        vital_events['VITAL_TYPE'] = vital_events['ITEMID'].map(itemid_to_label).fillna('UNKNOWN')
else:
    # Map ITEMID to vital sign type (original logic)
    def map_itemid_to_vital(itemid):
        for vital_type, itemids in VITAL_SIGNS_ITEMIDS.items():
            if itemid in itemids:
                return vital_type
        return 'UNKNOWN'
    
    vital_events['VITAL_TYPE'] = vital_events['ITEMID'].apply(map_itemid_to_vital)

print("\n   Vital signs distribution:")
print(vital_events['VITAL_TYPE'].value_counts().head(20))

print(f"\n[4/7] Merging with patient and ICU stay information...")

# Merge with ICU stays (handle missing ICUSTAY_ID gracefully)
if 'ICUSTAY_ID' in vital_events.columns and 'ICUSTAY_ID' in icustays.columns:
    vital_events = vital_events.merge(
        icustays[['ICUSTAY_ID', 'SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME']],
        on='ICUSTAY_ID',
        how='left',
        suffixes=('', '_ICU')
    )
else:
    print("   Note: ICUSTAY_ID not available, using HADM_ID linkage...")
    # Merge via HADM_ID instead
    vital_events = vital_events.merge(
        admissions[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME']],
        on='HADM_ID',
        how='left',
        suffixes=('', '_ADM')
    )
    vital_events['INTIME'] = vital_events.get('ADMITTIME', vital_events.get('CHARTTIME'))

# Merge with patient demographics
vital_events = vital_events.merge(
    patients[['SUBJECT_ID', 'GENDER', 'DOB']],
    on='SUBJECT_ID',
    how='left',
    suffixes=('', '_PAT')
)

print(f"   Merged dataset: {len(vital_events):,} records")

print(f"\n[5/7] Processing timestamps and calculating time deltas...")

# Convert timestamps
for col in ['CHARTTIME', 'INTIME', 'ADMITTIME']:
    if col in vital_events.columns:
        vital_events[col] = pd.to_datetime(vital_events[col], errors='coerce')

# Calculate time since admission/ICU admission
if 'INTIME' in vital_events.columns:
    vital_events['TIME_SINCE_ADMISSION_MIN'] = (
        vital_events['CHARTTIME'] - vital_events['INTIME']
    ).dt.total_seconds() / 60
else:
    vital_events['TIME_SINCE_ADMISSION_MIN'] = 0  # Fallback

# Convert values to numeric
vital_events['VALUENUM'] = pd.to_numeric(vital_events['VALUENUM'], errors='coerce')

# Remove invalid values
vital_events = vital_events.dropna(subset=['VALUENUM'])
vital_events = vital_events[vital_events['VALUENUM'] > 0]  # Remove negative/zero values

# Filter reasonable vital sign ranges
vital_events = vital_events[
    ((vital_events['VITAL_TYPE'].str.contains('HR|heart', case=False, na=False)) & 
     (vital_events['VALUENUM'].between(20, 220))) |
    ((vital_events['VITAL_TYPE'].str.contains('SpO2|O2|oxygen', case=False, na=False)) & 
     (vital_events['VALUENUM'].between(50, 100))) |
    ((vital_events['VITAL_TYPE'].str.contains('SBP|systolic|blood pressure', case=False, na=False)) & 
     (vital_events['VALUENUM'].between(40, 250))) |
    ((vital_events['VITAL_TYPE'].str.contains('RR|resp', case=False, na=False)) & 
     (vital_events['VALUENUM'].between(5, 60))) |
    ((vital_events['VITAL_TYPE'].str.contains('TEMP|temperature', case=False, na=False)) & 
     (vital_events['VALUENUM'].between(30, 45))) |
    (~vital_events['VITAL_TYPE'].str.contains('HR|SpO2|SBP|RR|TEMP', case=False, na=False))
]

print(f"   Valid measurements after filtering: {len(vital_events):,}")

print(f"\n[6/7] Applying clinical thresholds and DAVO decay rates...")

# Simplified classification based on vital type patterns
def classify_vital_simple(row):
    """Simplified classification when standard codes don't match"""
    vital_type_str = str(row['VITAL_TYPE']).lower()
    value = row['VALUENUM']
    
    # Default values
    severity = 'NORMAL'
    lambda_val = 0.1
    v0 = 20000
    uf = 1.0
    
    # Heart Rate
    if 'hr' in vital_type_str or 'heart' in vital_type_str:
        if value < 40 or value > 150:
            severity, lambda_val, v0, uf = 'CRITICAL', 1.2, 100000, 3.0
        elif value < 60 or value > 100:
            severity, lambda_val, v0, uf = 'WARNING', 0.3, 50000, 2.0
    
    # SpO2
    elif 'spo2' in vital_type_str or 'o2' in vital_type_str or 'oxygen' in vital_type_str:
        if value < 85:
            severity, lambda_val, v0, uf = 'CRITICAL', 1.5, 95000, 3.2
        elif value < 92:
            severity, lambda_val, v0, uf = 'WARNING', 0.6, 60000, 2.5
    
    # Blood Pressure (systolic assumed)
    elif 'bp' in vital_type_str or 'pressure' in vital_type_str:
        if value < 70 or value > 180:
            severity, lambda_val, v0, uf = 'CRITICAL', 0.7, 80000, 2.8
        elif value < 90 or value > 140:
            severity, lambda_val, v0, uf = 'WARNING', 0.4, 55000, 2.0
    
    # Respiratory Rate
    elif 'resp' in vital_type_str or 'rr' in vital_type_str:
        if value < 8 or value > 30:
            severity, lambda_val, v0, uf = 'CRITICAL', 1.0, 85000, 2.8
        elif value < 12 or value > 20:
            severity, lambda_val, v0, uf = 'WARNING', 0.5, 50000, 2.0
    
    # Temperature
    elif 'temp' in vital_type_str:
        if value < 35 or value > 39.5:
            severity, lambda_val, v0, uf = 'CRITICAL', 0.25, 50000, 2.0
        elif value < 36.5 or value > 37.5:
            severity, lambda_val, v0, uf = 'WARNING', 0.15, 35000, 1.5
    
    return pd.Series({
        'SEVERITY': severity,
        'LAMBDA': lambda_val,
        'V0': v0,
        'URGENCY_FACTOR': uf
    })

# Apply classification
davo_params = vital_events.apply(classify_vital_simple, axis=1)
vital_events = pd.concat([vital_events, davo_params], axis=1)

print("   Severity distribution:")
print(vital_events['SEVERITY'].value_counts())

print(f"\n[7/7] Calculating DAVO metrics and creating training dataset...")

# Calculate DAVO metrics
vital_events['TIME_ELAPSED_MIN'] = np.random.uniform(1, 30, size=len(vital_events))
vital_events['VALUE_T'] = vital_events['V0'] * np.exp(-vital_events['LAMBDA'] * vital_events['TIME_ELAPSED_MIN'])
vital_events['WORKLOAD_MIN'] = np.random.uniform(3, 20, size=len(vital_events))
vital_events['VALUE_DENSITY'] = vital_events['VALUE_T'] / vital_events['WORKLOAD_MIN']
vital_events['PRIORITY_SCORE'] = vital_events['VALUE_DENSITY'] * vital_events['URGENCY_FACTOR']

# Priority class
def assign_priority_class(score):
    if score > 8000:
        return 'CRITICAL'
    elif score > 2000:
        return 'HIGH'
    elif score > 500:
        return 'MEDIUM'
    else:
        return 'LOW'

vital_events['PRIORITY_CLASS'] = vital_events['PRIORITY_SCORE'].apply(assign_priority_class)

print("   Priority class distribution:")
print(vital_events['PRIORITY_CLASS'].value_counts())

# Create training prompts
def create_training_example(row):
    prompt = (f"PATIENT {row.get('SUBJECT_ID', 'UNKNOWN')} | {row['VITAL_TYPE']} ALERT - "
              f"Value: {row['VALUENUM']:.1f} "
              f"(time since admission: {row.get('TIME_SINCE_ADMISSION_MIN', 0):.0f} min). "
              f"Time since detection: {row['TIME_ELAPSED_MIN']:.1f} minutes.")
    
    if row['PRIORITY_CLASS'] == 'CRITICAL':
        action = "IMMEDIATE bedside assessment required. Activate rapid response team if needed."
    elif row['PRIORITY_CLASS'] == 'HIGH':
        action = "Respond within 5 minutes. Prepare intervention equipment."
    elif row['PRIORITY_CLASS'] == 'MEDIUM':
        action = "Assess within 15 minutes. Monitor trend."
    else:
        action = "Routine monitoring. Document and reassess per protocol."
    
    response = (f"Priority: {row['PRIORITY_CLASS']} | "
               f"Value Density: {row['VALUE_DENSITY']:.2f} | "
               f"Priority Score: {row['PRIORITY_SCORE']:.2f} | "
               f"Equipment-Specific Decay λ: {row['LAMBDA']:.2f}/min | "
               f"Clinical Action: {action}")
    
    return pd.Series({'prompt': prompt, 'response': response})

training_data = vital_events.apply(create_training_example, axis=1)
vital_events_final = pd.concat([vital_events, training_data], axis=1)

# Select relevant columns
output_cols = [
    'SUBJECT_ID', 'VITAL_TYPE', 'VALUENUM', 'TIME_SINCE_ADMISSION_MIN',
    'TIME_ELAPSED_MIN', 'SEVERITY', 'LAMBDA', 'V0', 'VALUE_T', 'WORKLOAD_MIN',
    'VALUE_DENSITY', 'PRIORITY_SCORE', 'PRIORITY_CLASS', 'prompt', 'response'
]

# Keep only columns that exist
output_cols = [col for col in output_cols if col in vital_events_final.columns]
df_final = vital_events_final[output_cols].copy()

# Save
output_file = VITAL_SIGNS_DIR / "mimic_davo_training.csv"
df_final.to_csv(output_file, index=False)

print(f"\n✓ Saved DAVO training data to: {output_file}")
print(f"   Total samples: {len(df_final):,}")

# Summary
summary_file = VITAL_SIGNS_DIR / "dataset_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MIMIC-III DAVO TRAINING DATASET SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Samples: {len(df_final):,}\n")
    f.write(f"Unique Patients: {df_final['SUBJECT_ID'].nunique()}\n\n")
    f.write("Vital Signs Distribution:\n")
    f.write(df_final['VITAL_TYPE'].value_counts().to_string() + "\n\n")
    f.write("Priority Class Distribution:\n")
    f.write(df_final['PRIORITY_CLASS'].value_counts().to_string() + "\n\n")
    f.write("Lambda (Decay Rate) Statistics:\n")
    f.write(df_final['LAMBDA'].describe().to_string() + "\n\n")

print(f"✓ Saved summary to: {summary_file}")

# Sample examples
print("\n" + "="*80)
print("SAMPLE TRAINING EXAMPLES")
print("="*80)
for i, row in df_final.sample(min(3, len(df_final))).iterrows():
    print(f"\nINPUT:  {row['prompt']}")
    print(f"OUTPUT: {row['response']}")

print("\n" + "="*80)
print("✓ MIMIC-III PROCESSING COMPLETE!")
print("="*80)
