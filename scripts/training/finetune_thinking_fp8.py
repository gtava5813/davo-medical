#!/usr/bin/env python3
"""Train Qwen3-4B-Thinking with Chain-of-Thought reasoning - FIXED"""

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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import argparse

def format_thinking_prompt(row):
    """Format with Chain-of-Thought reasoning for clinical decisions"""
    return f"""System: DAVO Clinical AI with reasoning capability

User: {row['prompt']}

Assistant (thinking): Let me analyze this step-by-step:
1. Equipment type and vital sign
2. Current value vs critical threshold
3. Time sensitivity (decay rate λ)
4. Value density calculation
5. Priority score computation
6. Final classification

Assistant (response): {row['response']}"""

def main(args):
    print("="*80)
    print("DAVO MEDICAL: Qwen3-4B-Thinking (Chain-of-Thought)")
    print("="*80)
    
    DATA_FILE = PROJECT_ROOT / args.data_path
    OUTPUT_DIR = PROJECT_ROOT / args.output_dir
    MODEL_PATH = args.model_path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Reasoning: Chain-of-Thought enabled")
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
    
    # Tokenize with thinking prompts
    def tokenize_fn(row):
        text = format_thinking_prompt(row)
        return tokenizer(text, truncation=True, max_length=768, padding="max_length")
    
    dataset = Dataset.from_pandas(df[['prompt', 'response']])
    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    tokenized = tokenized.train_test_split(test_size=0.1, seed=42)
    
    # Load model
    print("  Loading Thinking model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use BF16 instead of FP16
        low_cpu_mem_usage=True
    )
    
    # CRITICAL FIX: Enable gradient checkpointing BEFORE LoRA
    model.gradient_checkpointing_enable()
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    print("  ✓ Model loaded")
    
    # LoRA
    print(f"\n[3/5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable gradients for LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    
    # Training
    print(f"\n[4/5] Starting training with Chain-of-Thought...")
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
            bf16=True,  # Use BF16 instead of FP16
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="no",
            warmup_steps=100,
            report_to="none",
            save_total_limit=1,
            gradient_checkpointing=False,  # Already enabled above
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
