#!/usr/bin/env python3
"""
Convert Meta Llama .pth format to Hugging Face format
Specifically handles FP8 quantized models
"""

import torch
import json
from pathlib import Path
from transformers import LlamaConfig
from safetensors.torch import save_file
import sys

print("="*80)
print("DAVO MEDICAL: Meta → HuggingFace Model Converter")
print("="*80)

# Paths
META_PATH = Path.home() / ".llama/checkpoints/Llama-4-Maverick-17B-128E-Instruct-fp8"
HF_PATH = Path.home() / "davo-medical/models/checkpoints/maverick-hf-converted"

print(f"\nInput:  {META_PATH}")
print(f"Output: {HF_PATH}")

HF_PATH.mkdir(parents=True, exist_ok=True)

# Load Meta params
print("\n[1/5] Reading Meta model parameters...")
with open(META_PATH / "params.json") as f:
    params = json.load(f)

print(f"  Parameters: {json.dumps(params, indent=2)}")

# Create HF config
print("\n[2/5] Creating HuggingFace config...")
config = LlamaConfig(
    vocab_size=params.get("vocab_size", 128256),
    hidden_size=params.get("dim", 5120),
    intermediate_size=params.get("hidden_dim", 13824),
    num_hidden_layers=params.get("n_layers", 32),
    num_attention_heads=params.get("n_heads", 40),
    num_key_value_heads=params.get("n_kv_heads", 8),
    max_position_embeddings=params.get("max_seq_len", 131072),
    rms_norm_eps=params.get("norm_eps", 1e-5),
    rope_theta=params.get("rope_theta", 500000.0),
)

config.save_pretrained(str(HF_PATH))
print(f"  ✓ Config saved")

# Convert tokenizer
print("\n[3/5] Converting tokenizer...")
import shutil
shutil.copy(META_PATH / "tokenizer.model", HF_PATH / "tokenizer.model")

tokenizer_config = {
    "tokenizer_class": "LlamaTokenizer",
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|end_of_text|>",
    "pad_token": "<|end_of_text|>",
    "model_max_length": 131072,
    "legacy": True,
}
with open(HF_PATH / "tokenizer_config.json", "w") as f:
    json.dump(tokenizer_config, f, indent=2)
print(f"  ✓ Tokenizer created")

# Load weights
print("\n[4/5] Converting model weights (.pth → .safetensors)...")
print("  This may take 15-30 minutes...")

checkpoint_files = sorted(META_PATH.glob("consolidated.*.pth"))
print(f"  Found {len(checkpoint_files)} checkpoint shards")

if len(checkpoint_files) == 0:
    print("  ERROR: No consolidated.*.pth files found!")
    sys.exit(1)

state_dict = {}
for i, ckpt_file in enumerate(checkpoint_files):
    print(f"    Loading shard {i+1}/{len(checkpoint_files)}: {ckpt_file.name}")
    shard = torch.load(ckpt_file, map_location="cpu")
    for key, value in shard.items():
        state_dict[key] = value
    print(f"      ✓ Loaded ({len(shard)} keys)")

print(f"  Total keys: {len(state_dict)}")

# Convert keys
print("\n  Converting parameter names...")
hf_state_dict = {}

key_map = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

for i in range(params.get("n_layers", 32)):
    key_map.update({
        f"layers.{i}.attention.wq.weight": f"model.layers.{i}.self_attn.q_proj.weight",
        f"layers.{i}.attention.wk.weight": f"model.layers.{i}.self_attn.k_proj.weight",
        f"layers.{i}.attention.wv.weight": f"model.layers.{i}.self_attn.v_proj.weight",
        f"layers.{i}.attention.wo.weight": f"model.layers.{i}.self_attn.o_proj.weight",
        f"layers.{i}.feed_forward.w1.weight": f"model.layers.{i}.mlp.gate_proj.weight",
        f"layers.{i}.feed_forward.w2.weight": f"model.layers.{i}.mlp.down_proj.weight",
        f"layers.{i}.feed_forward.w3.weight": f"model.layers.{i}.mlp.up_proj.weight",
        f"layers.{i}.attention_norm.weight": f"model.layers.{i}.input_layernorm.weight",
        f"layers.{i}.ffn_norm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
    })

mapped_count = 0
for meta_key, tensor in state_dict.items():
    if meta_key in key_map:
        hf_state_dict[key_map[meta_key]] = tensor
        mapped_count += 1

print(f"  Mapped {mapped_count}/{len(state_dict)} parameters")

# Save
print("\n[5/5] Saving in HuggingFace format...")
total_size = sum(t.numel() * t.element_size() for t in hf_state_dict.values()) / (1024**3)
print(f"  Total size: {total_size:.2f} GB")

MAX_SHARD = 5 * 1024**3
num_shards = int(total_size // 5) + 1

if num_shards == 1:
    save_file(hf_state_dict, str(HF_PATH / "model.safetensors"))
    print(f"  ✓ Saved model.safetensors")
else:
    keys = list(hf_state_dict.keys())
    keys_per_shard = len(keys) // num_shards + 1
    weight_map = {}
    
    for idx in range(num_shards):
        start = idx * keys_per_shard
        end = min((idx + 1) * keys_per_shard, len(keys))
        shard_keys = keys[start:end]
        shard_dict = {k: hf_state_dict[k] for k in shard_keys}
        fname = f"model-{idx+1:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard_dict, str(HF_PATH / fname))
        print(f"  ✓ Shard {idx+1}/{num_shards}: {fname}")
        for k in shard_keys:
            weight_map[k] = fname
    
    index = {"metadata": {"total_size": int(total_size * 1024**3)}, "weight_map": weight_map}
    with open(HF_PATH / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"  ✓ Index saved")

print("\n" + "="*80)
print("✓ CONVERSION COMPLETE!")
print("="*80)
print(f"\nModel ready at: {HF_PATH}")
print("\nTrain with:")
print(f"  python scripts/training/finetune_maverick.py \\")
print(f"    --model_path models/checkpoints/maverick-hf-converted \\")
print(f"    --data_path data/processed/train.csv \\")
print(f"    --output_dir models/fine-tuned/maverick_davo_v1")
