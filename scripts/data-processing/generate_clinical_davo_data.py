"""
Generate synthetic clinical vital signs scenarios for DAVO training
Complements real MIMIC-III data with edge cases and diverse scenarios
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DAVO MEDICAL: Synthetic Clinical Data Generation")
print("="*80)

class ClinicalScenario:
    def __init__(self, equipment, measurement, normal, critical, v0, lam, uf):
        self.equipment = equipment
        self.measurement = measurement
        self.normal = normal
        self.critical = critical
        self.v0 = v0
        self.lam = lam
        self.uf = uf
    
    def value_at_t(self, t):
        return self.v0 * np.exp(-self.lam * t)
    
    def alert(self, pid, val, t):
        return f"PATIENT {pid} | {self.equipment} ALERT - {self.measurement}: {val:.1f} (normal: {self.normal}, critical: {self.critical}). Time since detection: {t:.1f} minutes."

scenarios = [
    ClinicalScenario("ECG Monitor", "Heart Rate", "60-100 bpm", 40, 100000, 1.5, 3.5),
    ClinicalScenario("ECG Monitor", "Heart Rate", "60-100 bpm", 160, 95000, 1.2, 3.2),
    ClinicalScenario("Pulse Oximeter", "SpO2", "95-100%", 85, 95000, 1.5, 3.3),
    ClinicalScenario("Pulse Oximeter", "SpO2", "95-100%", 90, 70000, 0.8, 2.7),
    ClinicalScenario("Ventilator", "Respiratory Rate", "12-20/min", 32, 85000, 1.0, 2.9),
    ClinicalScenario("BP Monitor", "Systolic BP", "90-120 mmHg", 190, 80000, 0.6, 2.5),
    ClinicalScenario("BP Monitor", "Systolic BP", "90-120 mmHg", 75, 85000, 0.7, 2.7),
    ClinicalScenario("Temperature", "Body Temp", "36.5-37.5°C", 39.8, 45000, 0.2, 1.7),
]

data = []
n_samples = 5000

print(f"\nGenerating {n_samples:,} synthetic samples...")

for i in range(n_samples):
    if (i + 1) % 1000 == 0:
        print(f"  {i+1:,}/{n_samples:,}...")
    
    scenario = np.random.choice(scenarios)
    pid = np.random.randint(2000, 2150)
    t = np.random.uniform(0.5, 45)
    
    # Generate realistic value
    if "Heart Rate" in scenario.measurement:
        val = np.random.randint(35, 180)
    elif "SpO2" in scenario.measurement:
        val = np.random.randint(80, 95)
    elif "BP" in scenario.measurement:
        val = np.random.randint(65, 210)
    elif "Respiratory" in scenario.measurement:
        val = np.random.randint(6, 40)
    else:
        val = round(np.random.uniform(34, 40.5), 1)
    
    v_t = scenario.value_at_t(t)
    workload = np.random.uniform(2, 25)
    vd = v_t / workload
    ps = vd * scenario.uf
    
    prompt = scenario.alert(pid, val, t)
    
    if ps > 8000:
        pc = "CRITICAL"
        action = "IMMEDIATE bedside assessment. Activate rapid response."
    elif ps > 2500:
        pc = "HIGH"
        action = "Respond within 5 minutes."
    elif ps > 600:
        pc = "MEDIUM"
        action = "Assess within 15 minutes."
    else:
        pc = "LOW"
        action = "Routine monitoring."
    
    response = f"Priority: {pc} | Value Density: {vd:.2f} | Priority Score: {ps:.2f} | Equipment-Specific Decay λ: {scenario.lam:.2f}/min | Clinical Action: {action}"
    
    data.append({
        'prompt': prompt,
        'response': response,
        'patient_id': pid,
        'equipment': scenario.equipment,
        'measurement': scenario.measurement,
        'value': val,
        'time_elapsed': t,
        'lambda': scenario.lam,
        'v0': scenario.v0,
        'value_t': v_t,
        'workload_min': workload,
        'value_density': vd,
        'priority_score': ps,
        'priority_class': pc
    })

df = pd.DataFrame(data)
output = DATA_DIR / "clinical_davo_synthetic.csv"
df.to_csv(output, index=False)

print(f"\n✓ Saved {len(df):,} samples to: {output}")
print(f"\nPriority distribution:")
print(df['priority_class'].value_counts())
print(f"\n{'='*80}")
print("✓ COMPLETE!")
print(f"{'='*80}")
