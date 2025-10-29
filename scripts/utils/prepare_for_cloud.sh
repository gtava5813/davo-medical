#!/bin/bash
# Clean and prepare for cloud deployment

echo "=================================="
echo "DAVO Medical: Cloud Prep Cleanup"
echo "=================================="

# 1. Remove failed/test models (save space)
echo "[1/5] Removing test models..."
rm -rf models/fine-tuned/llama32_1b_davo_test
rm -rf models/fine-tuned/llama32_davo_test
rm -rf models/fine-tuned/maverick_davo_v1
rm -rf models/fine-tuned/phi35_davo_test

# 2. Remove base model checkpoints (will download fresh on cloud)
echo "[2/5] Removing base checkpoints..."
rm -rf models/checkpoints/*

# 3. Keep only essential processed data
echo "[3/5] Cleaning data directories..."
cd data/processed
# Keep only the balanced training set
rm -f davo_combined_balanced.csv  # Old version
rm -f test.csv val.csv             # Will regenerate on cloud
mv train_balanced.csv ../train_balanced_backup.csv
cd ../..

# Remove synthetic (will use real MIMIC only)
rm -rf data/synthetic/

# 4. Clean logs and temp files
echo "[4/5] Removing logs and temp files..."
rm -f *.log nohup.out
rm -rf __pycache__ */__pycache__ */*/__pycache__
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# 5. Show what's left
echo "[5/5] Summary of cleaned structure:"
du -sh data/ models/ scripts/
echo ""
du -sh .

echo "✓ Cleanup complete! Ready for GitHub/zip."
