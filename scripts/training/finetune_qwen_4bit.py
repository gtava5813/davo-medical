#!/usr/bin/env python3
"""Train Qwen3-4B-Thinking with 4-bit quantization (FAST)"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import argparse

def main(args):
    print("="*80)
    print("DAVO MEDICAL: Qwen3-4B-Thinking (4-bit Optimized)")
    print("="*80)
    
    DATA_FILE = PROJECT_ROOT / args.data_path
    OUTPUT_DIR = PROJECT_ROOT / args.output_dir
    MODEL_PATH = args.model_path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Quantization: 4-bit NF4")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    # Load data
    print(f"\n[1/5] Loading training data...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Samples: {len(df):,}")
    
    # Load tokenizer
    print(f"\n[2/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    def tokenize_fn(row):
        text = f"System: DAVO Clinical AI\nUser: {row['prompt']}\nAssistant: {row['response']}"
        return tokenizer(text, truncation=True, max_length=512, padding="max_length")
    
    dataset = Dataset.from_pandas(df[['prompt', 'response']])
    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    tokenized = tokenized.train_test_split(test_size=0.1, seed=42)
    
    # Load with 4-bit quantization
    print("  Loading with 4-bit quantization (GPU-optimized)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("  ✓ Model loaded on GPU")
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA
    print(f"\n[3/5] Configuring LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ))
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")
    
    # Training
    print(f"\n[4/5] Starting training...")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=20,
            save_strategy="epoch",
            eval_strategy="no",
            warmup_steps=50,
            report_to="none",
            save_total_limit=1,
            gradient_checkpointing=True,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    
    # Save
    print("\n" + "="*80)
    print("Saving model...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print("\n✓ TRAINING COMPLETE!")
    print(f"Model: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
