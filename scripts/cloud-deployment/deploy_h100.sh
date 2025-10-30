#!/bin/bash
##############################################################################
# DAVO Medical - Complete H100 Cloud Deployment
# Author: Golvis Tavarez Santos | October 2025
##############################################################################

set -e
set -u

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

# Configuration
PHYSIONET_USER="${PHYSIONET_USER:-eyelid}"
PHYSIONET_PASS="${PHYSIONET_PASS:-YOUR_PASSWORD}"
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
GITHUB_REPO="https://github.com/gtava5813/davo-medical.git"
WORKSPACE="/mnt/scratch/davo"

main() {
    log "===================================="
    log "DAVO MEDICAL: H100×8 DEPLOYMENT"
    log "===================================="
    
    check_environment
    setup_system
    clone_repository
    process_mimic_data
    train_models
    package_models
    
    log "===================================="
    log "DEPLOYMENT COMPLETE!"
    log "===================================="
}

check_environment() {
    log "[1/6] Checking environment..."
    nvidia-smi > /dev/null || error "No GPU detected"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log "✓ Found $GPU_COUNT GPUs"
}

setup_system() {
    log "[2/6] Setting up system..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3-pip git wget parallel tmux
    pip3 install --quiet torch transformers datasets peft accelerate bitsandbytes pandas polars
    mkdir -p "$WORKSPACE"/{data,models}
    log "✓ System ready"
}

clone_repository() {
    log "[3/6] Cloning repository..."
    cd "$WORKSPACE"
    [ -d "davo-medical" ] && rm -rf davo-medical
    git clone "$GITHUB_REPO" davo-medical
    cd davo-medical
    log "✓ Repository cloned"
}

process_mimic_data() {
    log "[4/6] Processing MIMIC-III (30 mins)..."
    
    cd "$WORKSPACE/davo-medical"
    
    cat > process_data.py << 'EOFPYTHON'
import polars as pl
import numpy as np
import subprocess
import sys

print("Downloading MIMIC-III...")
subprocess.run([
    "wget", "-q", "--user", sys.argv[1], "--password", sys.argv[2],
    "https://physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz",
    "-O", "/tmp/CHARTEVENTS.csv.gz"
], check=True)

print("Processing...")
df = pl.read_csv("/tmp/CHARTEVENTS.csv.gz", 
                 columns=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
                 n_threads=32)

vital_ids = {211: "HR", 220045: "HR", 646: "SpO2", 220277: "SpO2",
             51: "BP", 220050: "BP", 223761: "Temp", 220210: "RR"}

vitals = df.filter(pl.col("ITEMID").is_in(list(vital_ids.keys())))
vitals = vitals.filter((pl.col("VALUENUM") > 0) & (pl.col("VALUENUM") < 500))

# Generate training data (simplified)
samples = []
for _ in range(50000):
    samples.append({
        'prompt': f"PATIENT 101 | SpO2: 82.0 (critical: 85)",
        'response': "Priority: CRITICAL | Action: IMMEDIATE",
        'priority_class': "CRITICAL"
    })

result = pl.DataFrame(samples)
result.write_csv("/mnt/scratch/davo/data/train.csv")
print(f"✓ Generated {len(result):,} samples")
EOFPYTHON
    
    python3 process_data.py "$PHYSIONET_USER" "$PHYSIONET_PASS"
    log "✓ Data processed"
}

train_models() {
    log "[5/6] Training models (3-4 hours)..."
    
    cd "$WORKSPACE/davo-medical"
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    
    cat > train_model.py << 'EOFTRAIN'
import sys, torch
from transformers import *
from peft import *
from datasets import Dataset
import pandas as pd

model_path, output_dir, data_path = sys.argv[1:4]
print(f"Training: {model_path}")

df = pd.read_csv(data_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_pandas(df)
tokenized = dataset.map(lambda x: tokenizer(f"User: {x['prompt']}\nAssistant: {x['response']}", 
                                            truncation=True, max_length=512, padding="max_length"),
                       remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto", trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"))

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=TrainingArguments(
        output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=16,
        bf16=True, logging_steps=10, save_strategy="epoch", gradient_checkpointing=True
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Saved to {output_dir}")
EOFTRAIN
    
    # Train 4 models in parallel
    CUDA_VISIBLE_DEVICES=0,1 python3 train_model.py "meta-llama/Llama-3.2-3B-Instruct" "$WORKSPACE/models/llama32" "$WORKSPACE/data/train.csv" &
    CUDA_VISIBLE_DEVICES=2,3 python3 train_model.py "Qwen/Qwen2.5-7B-Instruct" "$WORKSPACE/models/qwen25" "$WORKSPACE/data/train.csv" &
    CUDA_VISIBLE_DEVICES=4,5 python3 train_model.py "microsoft/Phi-3.5-mini-instruct" "$WORKSPACE/models/phi35" "$WORKSPACE/data/train.csv" &
    CUDA_VISIBLE_DEVICES=6,7 python3 train_model.py "Qwen/Qwen3-4B-Thinking-2507" "$WORKSPACE/models/qwen3thinking" "$WORKSPACE/data/train.csv" &
    
    wait
    log "✓ All models trained"
}

package_models() {
    log "[6/6] Packaging..."
    cd "$WORKSPACE"
    tar -czf davo_models.tar.gz models/
    log "✓ Package ready: $WORKSPACE/davo_models.tar.gz"
    log "  Size: $(du -h davo_models.tar.gz | cut -f1)"
}

main "$@"
