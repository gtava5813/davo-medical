#!/usr/bin/env python3
"""Compare all DAVO models (Phi-3.5 v1, Llama 3.2 v1, Llama 3.2 v2)"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pandas as pd

test_cases = [
    {"name": "CRITICAL: Hypoxemia", "alert": "PATIENT 4523 | SpO2: 82.0 (critical: 85). Time: 3.2 min.", "expected": "CRITICAL"},
    {"name": "HIGH: Tachycardia", "alert": "PATIENT 7821 | HR: 168 (critical: 160). Time: 5.5 min.", "expected": "HIGH"},
    {"name": "MEDIUM: Hypertension", "alert": "PATIENT 1092 | BP: 165 (critical: 190). Time: 12.0 min.", "expected": "MEDIUM"},
    {"name": "LOW: Mild Fever", "alert": "PATIENT 3456 | Temp: 38.2°C (critical: 39.8). Time: 25.0 min.", "expected": "LOW"}
]

models = {
    "Phi-3.5 v1 (4B)": "models/fine-tuned/phi35_davo_v1",
    "Llama 3.2 v2 (3B)": "models/fine-tuned/llama32_3b_davo_v2"
}

results = []

for model_name, model_path in models.items():
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    model_dir = Path.home() / "davo-medical" / model_path
    
    if not model_dir.exists():
        print(f"  ⚠️  Model not found, skipping...")
        continue
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    
    for test in test_cases:
        prompt = f"System: DAVO Clinical AI\nUser: {test['alert']}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, use_cache=False, temperature=0.3)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        
        priority = "UNKNOWN"
        for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if p in response:
                priority = p
                break
        
        correct = "✅" if priority == test["expected"] else "❌"
        results.append({"Model": model_name, "Test": test["name"], "Expected": test["expected"], "Got": priority, "Correct": correct})
        print(f"  {test['name'][:25]}: {priority:8} (Expected: {test['expected']:8}) {correct}")

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

df = pd.DataFrame(results)
print(df.to_string(index=False))

print("\n" + "="*80)
print("ACCURACY BY MODEL")
print("="*80)
for model_name in models.keys():
    model_results = df[df["Model"] == model_name]
    if len(model_results) > 0:
        accuracy = (model_results["Correct"] == "✅").sum() / len(model_results) * 100
        print(f"{model_name}: {accuracy:.1f}%")
