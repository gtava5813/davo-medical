#!/bin/bash
# Final cleanup before GitHub push

echo "=================================="
echo "DAVO Medical: GitHub Preparation"
echo "=================================="

cd ~/davo-medical

# [1] Move large MIMIC data OUTSIDE project (preserve for local use)
echo "[1/5] Moving MIMIC raw data outside project..."
mkdir -p ~/mimic-data-archive

# Move the large CSV files ONLY
mv data/mimic-iii/mimic-iii-clean/raw ~/mimic-data-archive/mimic-iii-raw
echo "  ✓ Moved raw CSVs (6GB) to ~/mimic-data-archive/mimic-iii-raw"

# Keep dataset_summary.txt (it's just metadata)
mkdir -p data/mimic-iii/mimic-iii-clean/vital_signs
# dataset_summary.txt stays (small text file)

# Remove only the large CSV
rm -f data/mimic-iii/mimic-iii-clean/vital_signs/mimic_davo_training.csv

# [2] Clean processed data (keep only scripts, not outputs)
echo "[2/5] Cleaning processed data..."
rm -f data/processed/train.csv
rm -f data/train_balanced_backup.csv

# Keep directory structure
touch data/processed/.gitkeep
touch data/raw/.gitkeep

# [3] Clean models (empty directories)
echo "[3/5] Cleaning model directories..."
rm -rf models/fine-tuned/*
rm -rf models/checkpoints/*
touch models/fine-tuned/.gitkeep
touch models/checkpoints/.gitkeep
touch models/exports/.gitkeep

# [4] Remove any leftover logs/temp files
echo "[4/5] Removing logs and temp files..."
find . -name "*.log" -delete
find . -name "nohup.out" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# [5] Show what will be committed
echo "[5/5] Summary of clean repository:"
echo ""
echo "Data directory:"
du -sh data/
echo ""
echo "Kept files:"
find data/mimic-iii/mimic-iii-clean/vital_signs/ -type f 2>/dev/null || echo "  (directory empty)"
echo ""
echo "Total repo size:"
du -sh .
echo ""
echo "✓ Repository cleaned and ready for GitHub!"
echo ""
echo "Archived data location: ~/mimic-data-archive/"
