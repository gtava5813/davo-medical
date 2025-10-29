
# DAVO Medical - Data Directory

## Overview

This directory contains scripts and structure for processing MIMIC-III clinical data. 
**Raw data is NOT included in this repository** due to:
- Size constraints (6GB+)
- PhysioNet license requirements
- Privacy/security best practices

## Data Structure
data/
├── mimic-iii/ # MIMIC-III data (downloaded on cloud)
│ └── .gitkeep
├── processed/ # Processed training data (generated)
│ └── .gitkeep
├── raw/ # Temporary raw data staging
│ └── .gitkeep
└── README.md # This file


## Getting MIMIC-III Data

### For Cloud Training (Recommended)

Data is automatically downloaded on the DigitalOcean H100 droplet during deployment:

On cloud - handled automatically by deploy script
wget --user YOUR_USER --password YOUR_PASS
https://physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz

### For Local Development

1. Request access: https://physionet.org/content/mimiciii/
2. Complete required training
3. Download essential files:
   - `CHARTEVENTS.csv.gz` (4GB - vital signs)
   - `ICUSTAYS.csv.gz` (patient ICU stays)
   - `D_ITEMS.csv.gz` (data dictionary)

4. Place in: `data/mimic-iii/mimic-iii-clean/raw/`

## Data Processing Pipeline

See `scripts/data-processing/` for:
- `process_mimic_vitals.py` - Extract vital signs
- `rebalance_training_data.py` - Balance priority classes
- Cloud processing scripts (Polars-based for speed)

## Privacy Notice

⚠️ **NEVER commit patient data to version control**

All MIMIC-III data is:
- De-identified per HIPAA Safe Harbor
- Subject to PhysioNet Data Use Agreement
- Must be kept secure and private
