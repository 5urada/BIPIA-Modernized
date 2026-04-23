# experiment/load_data.py
# Step 1: Load BIPIA email dataset and sample 30 prompts
# Saves sampled_prompts.json for use by all other scripts

from bipia.data import AutoPIABuilder
from datasets import Dataset
import json
import random

# ============================================================
# Configuration
# ============================================================
SEED = 42
BIPIA_SEED = 2023
NUM_SAMPLES = 150
DATASET_NAME = "email"
CONTEXT_DATA_FILE = "./benchmark/email/test.jsonl"
ATTACK_DATA_FILE = "./benchmark/text_attack_test.json"
OUTPUT_FILE = "./experiment/sampled_prompts.json"
TARGET_ATTACK_TYPES = [
    "Task Automation",
    "Business Intelligence",
    "Conversational Agent",
    "Research Assistance",
    "Sentiment Analysis",
    "Substitution Ciphers",
    "Base Encoding",
    "Reverse Text",
    "Emoji Substitution",
    "Language Translation",
    "Information Dissemination",
    "Marketing & Advertising",
    "Entertainment",
    "Scams & Fraud",
    "Misinformation & Propaganda",
]
# We focus on targeted attacks since they have the highest ASR
# per the BIPIA paper (Scams & Fraud, Marketing & Advertising, Misinformation)
TARGET_ATTACKS = [f"{a}-{i}" for a in TARGET_ATTACK_TYPES for i in range(5)]

# ============================================================
# Load dataset
# ============================================================
print("Loading BIPIA email dataset...")

pia_builder = AutoPIABuilder.from_name(DATASET_NAME)(BIPIA_SEED)
pia_samples = pia_builder(
    CONTEXT_DATA_FILE,
    ATTACK_DATA_FILE,
    enable_stealth=False,
)
pia_dataset = Dataset.from_pandas(pia_samples)
print(f"Full dataset size: {len(pia_dataset)}")

# ============================================================
# Filter to targeted attacks only
# ============================================================
filtered = pia_dataset.filter(lambda x: x["attack_name"] in TARGET_ATTACKS)
print(f"Filtered to targeted attacks: {len(filtered)} samples")

# ============================================================
# Sample 30 prompts
# ============================================================
random.seed(SEED)
indices = random.sample(range(len(filtered)), NUM_SAMPLES)
sampled = filtered.select(indices)
print(f"Sampled: {len(sampled)} prompts")

# ============================================================
# Save to JSON
# ============================================================
output = []
for example in sampled:
    output.append({
        "attack_name": example["attack_name"],
        "attack_str":  example["attack_str"],
        "context":     example["context"],
        "question":    example["question"],
        "ideal":       example["ideal"],
        "position":    example["position"],
        "task_name":   example["task_name"],
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(output)} samples to {OUTPUT_FILE}")
print("\n--- Sample 0 preview ---")
print(f"attack_name : {output[0]['attack_name']}")
print(f"attack_str  : {output[0]['attack_str']}")
print(f"question    : {output[0]['question']}")
print(f"position    : {output[0]['position']}")
print(f"context     : {output[0]['context'][:200]}...")
