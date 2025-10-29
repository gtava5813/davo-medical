"""
Test the fine-tuned model with sample alerts
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import argparse

HOME = Path.home()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="models/fine-tuned/maverick_davo_v1")
args = parser.parse_args()

model_dir = HOME / "davo-medical" / args.model_path
base_model_path = HOME / ".llama/checkpoints/Llama-4-Maverick-17B-128E-Instruct"

print("="*80)
print("DAVO MEDICAL: Inference Test")
print("="*80)

print(f"\nLoading model from: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

base_model = AutoModelForCausalLM.from_pretrained(
    str(base_model_path),
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

model = PeftModel.from_pretrained(base_model, str(model_dir))
model.eval()

print("✓ Model loaded\n")

# Test alerts
tests = [
    "PATIENT 101 | ECG Monitor ALERT - Heart Rate: 165.0 (normal: 60-100 bpm, critical: 160). Time since detection: 2.5 minutes.",
    "PATIENT 102 | Pulse Oximeter ALERT - SpO2: 82.0 (normal: 95-100%, critical: 85). Time since detection: 4.0 minutes.",
    "PATIENT 103 | BP Monitor ALERT - Systolic BP: 78.0 (normal: 90-120 mmHg, critical: 75). Time since detection: 8.0 minutes.",
]

for i, alert in enumerate(tests, 1):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a DAVO clinical decision support AI.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{alert}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("assistant")[-1].strip()
    
    print(f"\n{'='*80}")
    print(f"TEST {i}")
    print(f"{'='*80}")
    print(f"INPUT:\n{alert}\n")
    print(f"OUTPUT:\n{response}")

print(f"\n{'='*80}")
print("✓ INFERENCE TEST COMPLETE!")
print(f"{'='*80}")
