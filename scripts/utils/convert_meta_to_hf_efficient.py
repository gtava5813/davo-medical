#!/usr/bin/env python3
"""
Memory-efficient Meta → HF converter
Processes one shard at a time to avoid OOM
"""

import torch
import json
from pathlib import Path
from transformers import LlamaConfig
from safetensors.torch import save_file
import gc

print("="*80)
print("DAVO MEDICAL: Memory-Efficient Model Converter")
print("="*80)

META_PATH = Path.home() / ".llama/checkpoints/Llama-4-Maverick-17B-128E-Instruct-fp8"
HF_PATH = Path.home() / "davo-medical/models/checkpoints/maverick-hf-converted"

print(f"\nInput:  {META_PATH}")
print(f"Output: {HF_PATH}")
HF_PATH.mkdir(parents=True, exist_ok=True)

# Load params
print("\n[1/4] Reading model parameters...")
with open(META_PATH / "params.json") as f:
    params = json.load(f)

# Key parameters
n_layers = params["n_layers"]
vocab_size = params["vocab_size"]
print(f"  Layers: {n_layers}, Vocab: {vocab_size}")

# Create config
print("\n[2/4] Creating config...")
config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=params["dim"],
    intermediate_size=int(params["dim"] * params.get("ffn_exp", 4.0)),
    num_hidden_layers=n_layers,
    num_attention_heads=params["n_heads"],
    num_key_value_heads=params["n_kv_heads"],
    max_position_embeddings=131072,
    rms_norm_eps=params["norm_eps"],
    rope_theta=params["rope_theta"],
)
config.save_pretrained(str(HF_PATH))
print("  ✓ Config saved")

# Tokenizer
print("\n[3/4] Converting tokenizer...")
import shutil
shutil.copy(META_PATH / "tokenizer.model", HF_PATH / "tokenizer.model")
tokenizer_config = {
    "tokenizer_class": "LlamaTokenizer",
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|end_of_text|>",
    "pad_token": "<|end_of_text|>",
    "model_max_length": 131072,
}
with open(HF_PATH / "tokenizer_config.json", "w") as f:
    json.dump(tokenizer_config, f, indent=2)
print("  ✓ Tokenizer saved")

# Key mapping
def get_hf_key(meta_key, n_layers):
    """Convert Meta key to HF key"""
    if meta_key == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    elif meta_key == "norm.weight":
        return "model.norm.weight"
    elif meta_key == "output.weight":
        return "lm_head.weight"
    
    for i in range(n_layers):
        mappings = {
            f"layers.{i}.attention.wq.weight": f"model.layers.{i}.self_attn.q_proj.weight",
            f"layers.{i}.attention.wk.weight": f"model.layers.{i}.self_attn.k_proj.weight",
            f"layers.{i}.attention.wv.weight": f"model.layers.{i}.self_attn.v_proj.weight",
            f"layers.{i}.attention.wo.weight": f"model.layers.{i}.self_attn.o_proj.weight",
            f"layers.{i}.feed_forward.w1.weight": f"model.layers.{i}.mlp.gate_proj.weight",
            f"layers.{i}.feed_forward.w2.weight": f"model.layers.{i}.mlp.down_proj.weight",
            f"layers.{i}.feed_forward.w3.weight": f"model.layers.{i}.mlp.up_proj.weight",
            f"layers.{i}.attention_norm.weight": f"model.layers.{i}.input_layernorm.weight",
            f"layers.{i}.ffn_norm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
        }
        if meta_key in mappings:
            return mappings[meta_key]
    
    return None

# Process shards one at a time
print("\n[4/4] Converting weights (one shard at a time to save memory)...")
checkpoint_files = sorted(META_PATH.glob("consolidated.*.pth"))
print(f"  Found {len(checkpoint_files)} shards")

all_hf_keys = []
shard_mappings = {}  # Map HF key → shard file

for shard_idx, ckpt_file in enumerate(checkpoint_files):
    print(f"\n  Processing shard {shard_idx+1}/{len(checkpoint_files)}: {ckpt_file.name}")
    
    # Load shard
    print(f"    Loading...")
    state_dict = torch.load(ckpt_file, map_location="cpu")
    print(f"    Loaded {len(state_dict)} keys")
    
    # Convert keys
    hf_shard = {}
    for meta_key, tensor in state_dict.items():
        hf_key = get_hf_key(meta_key, n_layers)
        if hf_key:
            hf_shard[hf_key] = tensor
            all_hf_keys.append(hf_key)
            shard_mappings[hf_key] = shard_idx
    
    # Save this shard immediately
    shard_file = HF_PATH / f"model-{shard_idx+1:05d}-of-{len(checkpoint_files):05d}.safetensors"
    save_file(hf_shard, str(shard_file))
    print(f"    ✓ Saved {len(hf_shard)} params to {shard_file.name}")
    
    # Free memory
    del state_dict
    del hf_shard
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Create index
print(f"\n  Creating index...")
weight_map = {key: f"model-{shard_mappings[key]+1:05d}-of-{len(checkpoint_files):05d}.safetensors" 
              for key in all_hf_keys}

index = {
    "metadata": {"total_size": 400 * 1024**3},  # Approximate
    "weight_map": weight_map
}

with open(HF_PATH / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"  ✓ Index saved ({len(weight_map)} parameters)")

print("\n" + "="*80)
print("✓ CONVERSION COMPLETE!")
print("="*80)
print(f"\nModel: {HF_PATH}")
print(f"Total parameters: {len(all_hf_keys):,}")
print("\nTrain with:")
print(f"  python scripts/training/finetune_maverick.py \\")
print(f"    --model_path models/checkpoints/maverick-hf-converted \\")
print(f"    --data_path data/processed/train.csv \\")
print(f"    --output_dir models/fine-tuned/maverick_davo_v1")
