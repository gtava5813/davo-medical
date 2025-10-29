# DAVO Medical: Clinical AI for IoT Vital Signs Prioritization

<div align="center">

**Decay-Aware Value-Optimized AI for Real-Time ICU Alert Triage**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

</div>

---

## 🎯 Overview

DAVO Medical applies time-sensitive scheduling algorithms to prioritize vital signs alerts in multi-patient ICU monitoring systems. Traditional hospital alarm systems suffer from 90% false positive rates leading to alert fatigue. DAVO uses **equipment-specific decay rates (λ)** to model how quickly clinical value deteriorates.

### Key Innovation

| Equipment | Decay Rate (λ) | Critical Threshold | Priority Impact |
|-----------|----------------|-------------------|-----------------|
| SpO₂ | 1.5/min | 85% | Rapid hypoxia risk |
| Heart Rate | 1.2/min | 160 bpm | Arrhythmia progression |
| Blood Pressure | 0.6/min | 190 mmHg | Slower deterioration |
| Temperature | 0.2/min | 39.8°C | Gradual changes |

**Priority Score**: \( P = V_d \times e^{\lambda \times t} \)

Where:
- \( V_d \) = Value density (deviation from normal)
- \( \lambda \) = Equipment-specific decay rate
- \( t \) = Time elapsed since alert

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (16GB+ VRAM recommended)
- CUDA 11.8+
- PhysioNet MIMIC-III access

### Installation


Clone repository
git clone https://github.com/YOUR_USERNAME/davo-medical.git
cd davo-medical

Create virtual environment
python3 -m venv ai-env
source ai-env/bin/activate

Install dependencies
pip install -r requirements.txt

text

### Get MIMIC-III Data

1. Request access: https://physionet.org/content/mimiciii/
2. Complete CITI training
3. Download essential files (see `data/README.md`)

---

## 📊 Cloud Training (Recommended)

Train all models on DigitalOcean H100×8:

On local machine - prepare for cloud
bash scripts/utils/github_prep_final.sh

On cloud droplet
bash deploy_on_digitalocean_h100.sh

text

**Training Time**: 4 hours  
**Cost**: ~$270  
**Output**: 5 production-ready models

### Models Trained

| Model | Parameters | Accuracy | Use Case |
|-------|-----------|----------|----------|
| **Llama-4-Maverick** | 17B | 94-97% | Production (best) |
| **Qwen3-8B** | 8B | 91-94% | High accuracy alternative |
| **Qwen3-4B-Thinking** | 4B | 89-93% | Edge + explainability |
| **Llama-4-Scout** | 17B | 86-90% | Fast inference |
| **Mamba-2.8B (SSM)** | 2.8B | 88-92% | Validator |

---

## 🏥 Usage

### Inference

from transformers import AutoModelForCausalLM, AutoTokenizer

Load trained model
model = AutoModelForCausalLM.from_pretrained("models/maverick_davo")
tokenizer = AutoTokenizer.from_pretrained("models/maverick_davo")

Clinical alert
alert = "PATIENT 101 | SpO2: 82.0 (critical: 85). Time: 3.2 min."
inputs = tokenizer(alert, return_tensors="pt")

Generate priority assessment
outputs = model.generate(**inputs, max_new_tokens=150)
response = tokenizer.decode(outputs, skip_special_tokens=True)

print(response)

Output: Priority: CRITICAL | Value Density: 1169.72 |
Priority Score: 3249.24 | Equipment-Specific Decay λ: 1.50/min |
Clinical Action: IMMEDIATE bedside assessment required.
text

### Ensemble Validation (Production)

from davo_ensemble import DAVOEnsemble

ensemble = DAVOEnsemble(
primary="models/maverick_davo",
validator="models/mamba_davo"
)

prediction = ensemble.predict(alert)

Returns: {priority, confidence, requires_human_review}
text

---

## 📁 Project Structure

davo-medical/
├── data/
│ ├── mimic-iii/ # MIMIC data (not in repo)
│ └── processed/ # Generated training data
├── models/
│ ├── checkpoints/ # Base models (download on cloud)
│ └── fine-tuned/ # Trained models
├── scripts/
│ ├── data-processing/ # MIMIC processing pipeline
│ ├── training/ # Fine-tuning scripts
│ ├── evaluation/ # Model testing
│ └── utils/ # Deployment helpers
├── docs/ # Documentation
├── notebooks/ # Jupyter experiments
└── requirements.txt

text

---

## 🔬 Development

### Process MIMIC-III Data

Extract vital signs from MIMIC
python scripts/data-processing/process_mimic_vitals.py

Balance priority classes
python scripts/data-processing/rebalance_training_data.py

text

### Train Custom Model

python scripts/training/finetune_maverick.py
--model_path meta-llama/Llama-3.2-3B-Instruct
--data_path data/processed/train_balanced.csv
--output_dir models/custom_model
--epochs 3
--batch_size 4

text

### Evaluate

python scripts/evaluation/compare_all_models.py

text

---

## 📈 Results

### Model Performance (Full MIMIC-III Training)

| Priority Level | Phi-3.5 (4B) | **Maverick (17B)** | Qwen3-Thinking |
|----------------|--------------|-------------------|----------------|
| CRITICAL | 0% | **95%** ✅ | 92% |
| HIGH | 100% | **94%** ✅ | 88% |
| MEDIUM | 0% | **93%** ✅ | 90% |
| LOW | 100% | **98%** ✅ | 95% |
| **Overall** | 50% | **95%** ✅ | 91% |

---

## 🛠️ Technology Stack

- **Models**: Llama 4, Qwen3, Mamba (SSM)
- **Training**: PyTorch, Transformers, PEFT, LoRA
- **Data**: MIMIC-III Clinical Database
- **Processing**: Polars, Pandas, NumPy
- **Deployment**: DigitalOcean H100, Docker
- **Monitoring**: TensorBoard, Weights & Biases

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **MIMIC-III Database**: PhysioNet, MIT Lab for Computational Physiology
- **Models**: Meta AI (Llama), Alibaba (Qwen), Tri Dao (Mamba)
- **Inspiration**: Time-sensitive scheduling in real-time systems

---

## 📧 Contact

Questions? Open an issue or reach out!

---

## 🚧 Roadmap

- [ ] Real-time streaming inference API
- [ ] Edge deployment (Raspberry Pi, NVIDIA Jetson)
- [ ] Multi-modal integration (ECG waveforms, imaging)
- [ ] Clinical trial deployment
- [ ] FDA 510(k) submission

---

<div align="center">

**Built with ❤️ for critical care**

</div>
