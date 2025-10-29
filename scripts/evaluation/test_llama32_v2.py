#!/usr/bin/env python3
"""Test Llama 3.2 3B v2 with balanced training"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_DIR = Path.home() / "davo-medical/models/fine-tuned/llama32_3b_davo_v2"

print("="*80)
print("DAVO MEDICAL: Testing Llama 3.2 3B v2 (Balanced Training)")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR), device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model.eval()
print("✓ Model loaded\n")

test_alerts = [
    {"name": "CRITICAL: Severe Hypoxemia", "alert": "PATIENT 4523 | SpO2: 82.0 (critical: 85). Time: 3.2 min.", "expected": "CRITICAL"},
    {"name": "HIGH: Severe Tachycardia", "alert": "PATIENT 7821 | HR: 168 (critical: 160). Time: 5.5 min.", "expected": "HIGH"},
    {"name": "MEDIUM: Moderate Hypertension", "alert": "PATIENT 1092 | BP: 165 (critical: 190). Time: 12.0 min.", "expected": "MEDIUM"},
    {"name": "LOW: Mild Fever", "alert": "PATIENT 3456 | Temp: 38.2°C (critical: 39.8). Time: 25.0 min.", "expected": "LOW"}
]

correct = 0
for test in test_alerts:
    print("="*80)
    print(f"TEST: {test['name']}")
    print("="*80)
    
    prompt = f"System: DAVO Clinical AI\nUser: {test['alert']}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.3, use_cache=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    
    # Extract priority
    priority = "UNKNOWN"
    for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if p in response:
            priority = p
            break
    
    result = "✅" if priority == test["expected"] else "❌"
    if priority == test["expected"]:
        correct += 1
    
    print(f"\n🏥 INPUT: {test['alert']}")
    print(f"\n🤖 RESPONSE:\n{response[:200]}...")
    print(f"\n📊 PRIORITY: {priority} (Expected: {test['expected']}) {result}\n")

print("="*80)
print(f"FINAL ACCURACY: {correct}/4 ({correct/4*100:.1f}%)")
print("="*80)
