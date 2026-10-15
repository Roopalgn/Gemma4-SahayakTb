"""Verify correct_output preserves ground truth tiers.

For each eval sample, feeds the GROUND TRUTH assistant response through
correct_output with the user text and checks the tier isn't changed.
"""
import json
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.postprocess import correct_output

eval_path = Path(__file__).parent.parent / "data" / "eval.jsonl"
eval_data = [json.loads(l) for l in open(eval_path)][:20]

print("=== Checking ground truth consistency with new rules ===\n")
regressions = 0

for i, sample in enumerate(eval_data):
    user_text = next(m for m in sample["messages"] if m["role"] == "user")["content"]
    expected = json.loads(next(m for m in sample["messages"] if m["role"] == "assistant")["content"])
    tier = expected["dropout_risk_tier"]
    factors = expected.get("risk_factors", [])
    month = expected.get("treatment_month", 1)

    # Simulate: if model perfectly reproduces GT, does postprocess break it?
    test_output = copy.deepcopy({
        "dropout_risk_tier": tier,
        "treatment_month": month,
        "risk_factors": factors,
    })
    corrected = correct_output(test_output, user_text=user_text)
    final_tier = corrected["dropout_risk_tier"]

    if final_tier != tier:
        print(f"  CHANGED sample {i+1}: {tier} -> {final_tier} (factors={len(factors)}, month={month})")
        # Show what factors were added
        orig_ids = {f.get("factor", "") for f in factors}
        new_ids = {f.get("factor", "") for f in corrected.get("risk_factors", [])}
        added = new_ids - orig_ids
        if added:
            print(f"           Added factors: {added}")
        regressions += 1

if regressions == 0:
    print("  All 20 samples: postprocess preserves correct ground truth tiers.")
else:
    print(f"\n  {regressions} REGRESSIONS detected!")

print(f"\n=== Done ===")
