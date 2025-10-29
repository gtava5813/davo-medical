#!/bin/bash
# Clean up failed/unnecessary models and files

echo "=================================="
echo "DAVO Medical: Clean Slate Cleanup"
echo "=================================="

# Stop any running training
pkill -f finetune

# Remove failed/partial models
echo "Removing partial models..."
rm -rf models/fine-tuned/phi35_davo_v1
rm -rf models/fine-tuned/llama32_3b_davo_v1
rm -rf models/fine-tuned/llama32_3b_davo_v2
rm -rf models/fine-tuned/qwen3_4b_thinking_davo_v1

# Remove downloaded base models (we'll re-download on cloud)
echo "Removing local base models..."
rm -rf models/checkpoints/phi35-mini-instruct
rm -rf models/checkpoints/llama32-3b
rm -rf models/checkpoints/qwen3-4b-thinking
rm -rf models/checkpoints/qwen3-4b-thinking-fp8
rm -rf models/checkpoints/maverick-17b-fp8-hf

# Remove old training logs
echo "Removing old logs..."
rm -f training*.log
rm -f nohup.out

# Remove temporary files
rm -rf offload/
rm -rf __pycache__/
find . -name "*.pyc" -delete

# Keep only essential files
echo ""
echo "Keeping essential files:"
echo "  ✓ data/mimic-iii/ (raw MIMIC data)"
echo "  ✓ data/processed/train_balanced.csv (balanced training data)"
echo "  ✓ scripts/ (all scripts)"
echo "  ✓ README.md"
echo ""

# Show disk space saved
du -sh models/ 2>/dev/null || echo "No models directory"
du -sh . | tail -1

echo "✓ Cleanup complete! Ready for cloud training."
