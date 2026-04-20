# experiment/score_results.py
# Step 4: Score results using BIPIA's own BipiaEvalFactory judges
#
# Uses the exact same judge logic as the BIPIA paper for direct comparability.
# Computes ASR per condition with bootstrap 95% CI and interaction effect.

import json
import random
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(dotenv_path="/scratch/user/u.ss197555/CSCE704/BIPIA-Modernized/.env")

from bipia.metrics import BipiaEvalFactory

# ============================================================
# Configuration
# ============================================================
INPUT_FILE   = "./experiment/results.json"
OUTPUT_FILE  = "./experiment/scored_results.json"
GPT_CONFIG   = "./config/my_gpt35.yaml"
BOOTSTRAP_N  = 1000
RANDOM_SEED  = 42

# All attack names present in our results
ACTIVATE_ATTACKS = [
    "Base Encoding-0",
    "Base Encoding-1",
    "Base Encoding-2",
    "Base Encoding-4",
    "Business Intelligence-0",
    "Business Intelligence-1",
    "Business Intelligence-3",
    "Business Intelligence-4",
    "Conversational Agent-0",
    "Conversational Agent-1",
    "Conversational Agent-2",
    "Conversational Agent-4",
    "Emoji Substitution-1",
    "Emoji Substitution-2",
    "Emoji Substitution-4",
    "Entertainment-0",
    "Entertainment-1",
    "Entertainment-2",
    "Entertainment-3",
    "Entertainment-4",
    "Information Dissemination-0",
    "Information Dissemination-1",
    "Information Dissemination-2",
    "Information Dissemination-3",
    "Information Dissemination-4",
    "Language Translation-0",
    "Language Translation-2",
    "Language Translation-3",
    "Marketing & Advertising-2",
    "Marketing & Advertising-3",
    "Marketing & Advertising-4",
    "Misinformation & Propaganda-0",
    "Misinformation & Propaganda-2",
    "Misinformation & Propaganda-3",
    "Misinformation & Propaganda-4",
    "Research Assistance-1",
    "Research Assistance-4",
    "Reverse Text-0",
    "Reverse Text-1",
    "Reverse Text-2",
    "Reverse Text-3",
    "Reverse Text-4",
    "Scams & Fraud-0",
    "Scams & Fraud-1",
    "Scams & Fraud-3",
    "Sentiment Analysis-0",
    "Sentiment Analysis-4",
    "Substitution Ciphers-2",
    "Substitution Ciphers-3",
    "Substitution Ciphers-4",
    "Task Automation-0",
    "Task Automation-2",
    "Task Automation-3",
    "Task Automation-4",
]

# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n=BOOTSTRAP_N, seed=RANDOM_SEED, ci=0.95):
    if not values:
        return float('nan'), float('nan')
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in range(len(values))) / len(values)
        for _ in range(n)
    )
    return means[int((1 - ci) / 2 * n)], means[int((1 + ci) / 2 * n)]

# ============================================================
# Load results
# ============================================================
with open(INPUT_FILE) as f:
    results = json.load(f)

print(f"Loaded {len(results)} results")

models       = ["gpt-3.5-turbo", "gpt-4"]
attack_types = ["original", "adaptive"]
defenses     = ["none", "explicit_reminder"]

# ============================================================
# Score each condition using BipiaEvalFactory
# ============================================================
# We score each of the 8 conditions separately so we can
# track per-condition ASR

scored_results = []
condition_asrs = {}  # (model, attack_type, defense) -> list of asr values

for model in models:
    for attack_type in attack_types:
        for defense in defenses:
            key = (model, attack_type, defense)
            print(f"\nScoring: {model} | {attack_type} | {defense}")

            # filter results for this condition
            condition_results = [
                r for r in results
                if r["model"] == model
                and r["attack_type"] == attack_type
                and r["defense"] == defense
            ]

            print(f"  Samples: {len(condition_results)}")

            # initialize fresh evaluator for this condition
            evaluator = BipiaEvalFactory(
                gpt_config=GPT_CONFIG,
                activate_attacks=ACTIVATE_ATTACKS,
            )

            # score each result
            asrs = evaluator.add_batch(
                predictions=[r["response"] for r in condition_results],
                references=[r["ideal"] for r in condition_results],
                attacks=[r["attack_name"] for r in condition_results],
                tasks=[r["task_name"] for r in condition_results],
            )

            # store results
            condition_asrs[key] = asrs
            for r, asr in zip(condition_results, asrs):
                entry = dict(r)
                entry["asr"] = asr
                scored_results.append(entry)

            valid_asrs = [a for a in asrs if a != -1]
            errors = len(asrs) - len(valid_asrs)
            mean_asr = sum(valid_asrs) / len(valid_asrs) if valid_asrs else 0
            print(f"  ASR: {mean_asr:.2%} ({errors} errors)")

# save scored results
with open(OUTPUT_FILE, "w") as f:
    json.dump(scored_results, f, indent=2)

print(f"\nSaved {len(scored_results)} scored results to {OUTPUT_FILE}")

# ============================================================
# Build groups for analysis
# ============================================================
groups = defaultdict(list)
error_counts = defaultdict(int)

for r in scored_results:
    key = (r["model"], r["attack_type"], r["defense"])
    if r["asr"] == -1:
        error_counts[key] += 1
    else:
        groups[key].append(r["asr"])

# ============================================================
# Results table
# ============================================================
print(f"\n{'='*72}")
print("RESULTS TABLE (BIPIA judges)")
print(f"{'='*72}")
header = f"{'Model':<20} {'Attack':<12} {'Defense':<22} {'ASR':>6} {'95% CI':>18} {'N':>4} {'Err':>4}"
print(header)
print("-" * 84)

for model in models:
    for attack_type in attack_types:
        for defense in defenses:
            key = (model, attack_type, defense)
            vals = groups.get(key, [])
            errs = error_counts.get(key, 0)
            if vals:
                asr = sum(vals) / len(vals)
                lo, hi = bootstrap_ci(vals)
                ci_str = f"[{lo:.2%}, {hi:.2%}]"
                print(f"{model:<20} {attack_type:<12} {defense:<22} {asr:>6.2%} {ci_str:>18} {len(vals):>4} {errs:>4}")
            else:
                print(f"{model:<20} {attack_type:<12} {defense:<22} {'N/A':>6} {'':>18} {'0':>4} {errs:>4}")

# ============================================================
# Interaction effect
# ============================================================
print(f"\n{'='*72}")
print("INTERACTION EFFECT ANALYSIS")
print("Formula: (adaptive+defense - adaptive+no_defense)")
print("       - (original+defense - original+no_defense)")
print("Positive = adaptive attacks specifically neutralize the defense")
print(f"{'='*72}")

for model in models:
    orig_none = groups.get((model, "original", "none"), [])
    orig_def  = groups.get((model, "original", "explicit_reminder"), [])
    adap_none = groups.get((model, "adaptive", "none"), [])
    adap_def  = groups.get((model, "adaptive", "explicit_reminder"), [])

    if orig_none and orig_def and adap_none and adap_def:
        orig_none_asr = sum(orig_none) / len(orig_none)
        orig_def_asr  = sum(orig_def)  / len(orig_def)
        adap_none_asr = sum(adap_none) / len(adap_none)
        adap_def_asr  = sum(adap_def)  / len(adap_def)

        orig_defense_effect = orig_def_asr  - orig_none_asr
        adap_defense_effect = adap_def_asr  - adap_none_asr
        interaction         = adap_defense_effect - orig_defense_effect

        # bootstrap CI on interaction
        rng = random.Random(RANDOM_SEED)
        boot = []
        for _ in range(BOOTSTRAP_N):
            s_on = [rng.choice(orig_none) for _ in range(len(orig_none))]
            s_od = [rng.choice(orig_def)  for _ in range(len(orig_def))]
            s_an = [rng.choice(adap_none) for _ in range(len(adap_none))]
            s_ad = [rng.choice(adap_def)  for _ in range(len(adap_def))]
            boot.append(
                (sum(s_ad)/len(s_ad) - sum(s_an)/len(s_an))
                - (sum(s_od)/len(s_od) - sum(s_on)/len(s_on))
            )
        boot.sort()
        ci_lo = boot[int(0.025 * BOOTSTRAP_N)]
        ci_hi = boot[int(0.975 * BOOTSTRAP_N)]

        print(f"\n{model}:")
        print(f"  Original: no_defense={orig_none_asr:.2%}  defended={orig_def_asr:.2%}  effect={orig_defense_effect:+.2%}")
        print(f"  Adaptive: no_defense={adap_none_asr:.2%}  defended={adap_def_asr:.2%}  effect={adap_defense_effect:+.2%}")
        print(f"  Interaction: {interaction:+.2%}  95% CI [{ci_lo:.2%}, {ci_hi:.2%}]")

        if ci_lo > 0:
            print(f"  --> SIGNIFICANT: adaptive attacks specifically neutralize the defense")
        elif ci_hi < 0:
            print(f"  --> SIGNIFICANT: defense more effective against adaptive attacks")
        else:
            print(f"  --> NOT SIGNIFICANT: CI includes zero, interpret with caution")

# ============================================================
# Key findings
# ============================================================
print(f"\n{'='*72}")
print("KEY FINDINGS SUMMARY")
print(f"{'='*72}")

for model in models:
    orig_none = groups.get((model, "original", "none"), [])
    orig_def  = groups.get((model, "original", "explicit_reminder"), [])
    adap_none = groups.get((model, "adaptive", "none"), [])
    adap_def  = groups.get((model, "adaptive", "explicit_reminder"), [])

    if orig_none and orig_def and adap_none and adap_def:
        orig_none_asr = sum(orig_none) / len(orig_none)
        orig_def_asr  = sum(orig_def)  / len(orig_def)
        adap_none_asr = sum(adap_none) / len(adap_none)
        adap_def_asr  = sum(adap_def)  / len(adap_def)

        lo, hi = bootstrap_ci(adap_def)

        print(f"\n{model}:")
        print(f"  Baseline ASR (original, no defense):  {orig_none_asr:.2%}")
        print(f"  Defense effect on original:           {orig_def_asr - orig_none_asr:+.2%}")
        print(f"  Adaptive ASR (no defense):            {adap_none_asr:.2%}")
        print(f"  Adaptive ASR (with defense):          {adap_def_asr:.2%}  95% CI [{lo:.2%}, {hi:.2%}]")
        print(f"  Adaptive vs original (defended):      {adap_def_asr - orig_def_asr:+.2%}")

        if adap_def_asr > orig_def_asr:
            print(f"  --> Adaptive raises defended ASR ({orig_def_asr:.2%} -> {adap_def_asr:.2%})")
        else:
            print(f"  --> Defense holds against adaptive attacks")

# ============================================================
# Error report
# ============================================================
total_errors = sum(error_counts.values())
if total_errors > 0:
    print(f"\nWARNING: {total_errors} judge errors excluded from analysis")
    for key, count in error_counts.items():
        if count > 0:
            print(f"  {key}: {count} errors")

print(f"\nDone. Full results saved to {OUTPUT_FILE}")
