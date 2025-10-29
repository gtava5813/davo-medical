#!/usr/bin/env python3
"""Test the trained DAVO clinical AI model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_DIR = Path.home() / "davo-medical/models/fine-tuned/llama32_3b_davo_v1"

print("="*80)
print("DAVO MEDICAL: Testing Llama 3.2 3B Fine-Tuned Model")
print("="*80)

print(f"\nLoading model from: {MODEL_DIR}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()
print("✓ Model loaded\n")

# Test scenarios
test_alerts = [
    {
        "name": "CRITICAL: Severe Hypoxemia",
        "alert": "PATIENT 4523 | Pulse Oximeter ALERT - SpO2: 82.0 (normal: 95-100%, critical: 85). Time since detection: 3.2 minutes."
    },
    {
        "name": "HIGH: Severe Tachycardia",
        "alert": "PATIENT 7821 | ECG Monitor ALERT - Heart Rate: 168.0 (normal: 60-100 bpm, critical: 160). Time since detection: 5.5 minutes."
    },
    {
        "name": "MEDIUM: Moderate Hypertension",
        "alert": "PATIENT 1092 | Blood Pressure Monitor ALERT - Systolic BP: 165.0 (normal: 90-120 mmHg, critical: 190). Time since detection: 12.0 minutes."
    },
    {
        "name": "LOW: Mild Fever",
        "alert": "PATIENT 3456 | Temperature Monitor ALERT - Body Temperature: 38.2°C (normal: 36.5-37.5°C, critical: 39.8). Time since detection: 25.0 minutes."
    }
]

for i, test in enumerate(test_alerts, 1):
    print("="*80)
    print(f"TEST {i}: {test['name']}")
    print("="*80)
    
    prompt = f"System: DAVO Clinical AI\nUser: {test['alert']}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    
    print(f"\n🏥 INPUT:\n{test['alert']}\n")
    print(f"🤖 DAVO AI RESPONSE:\n{response}\n")

print("="*80)
print("✓ TESTING COMPLETE!")
print("="*80)
