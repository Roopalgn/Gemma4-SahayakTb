"""Quick 20-sample evaluation script."""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import RISK_ASSESSMENT_SCHEMA, SYSTEM_PROMPT, validate_risk_assessment
from inference.postprocess import correct_output
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="Roopalgn/sahayak-tb-gemma4-e2b-gguf",
    filename="gemma-4-e2b-it.Q4_K_M.gguf",
)
EVAL_PATH = Path(__file__).parent.parent / "data" / "eval.jsonl"
N_SAMPLES = 20

print("Loading model...")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=0, n_ctx=4096, verbose=False)
print("Model loaded.")

data = [json.loads(l) for l in open(EVAL_PATH)][:N_SAMPLES]
results = []

for i, sample in enumerate(data):
    user_input = next(m for m in sample["messages"] if m["role"] == "user")["content"]
    expected = json.loads(next(m for m in sample["messages"] if m["role"] == "assistant")["content"])
    t0 = time.time()
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object", "schema": RISK_ASSESSMENT_SCHEMA},
            temperature=0.0,
            max_tokens=1024,
        )
        raw = resp["choices"][0]["message"]["content"]
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {i+1}/{N_SAMPLES}: ERROR ({elapsed:.0f}s) {e}", flush=True)
        results.append({"i": i, "expected": expected["dropout_risk_tier"], "predicted": "INVALID", "match": False, "valid": False, "latency": round(time.time() - t0, 1)})
        continue

    elapsed = time.time() - t0
    try:
        parsed = json.loads(raw)
        parsed = correct_output(parsed, user_text=user_input)
        valid, _ = validate_risk_assessment(parsed)
        pred_tier = parsed.get("dropout_risk_tier")
    except Exception:
        parsed = None
        valid = False
        pred_tier = None

    match = pred_tier == expected["dropout_risk_tier"]
    results.append({
        "i": i,
        "expected": expected["dropout_risk_tier"],
        "predicted": pred_tier or "INVALID",
        "match": match,
        "valid": valid,
        "latency": round(elapsed, 1),
    })
    status = "OK" if match else "MISS"
    tier_str = pred_tier or "INVALID"
    print(f"  {i+1}/{N_SAMPLES}: expected={expected['dropout_risk_tier']:>8} predicted={tier_str:>8} {status} ({elapsed:.0f}s)", flush=True)
    if not match and parsed:
        factors = parsed.get("risk_factors", [])
        print(f"         factors({len(factors)}): {[f.get('factor','?')[:30] for f in factors]}", flush=True)

# Summary
correct = sum(1 for r in results if r["match"])
schema_ok = sum(1 for r in results if r["valid"])
high_crit_tp = sum(1 for r in results if r["expected"] in ("high", "critical") and r["predicted"] in ("high", "critical"))
high_crit_total = sum(1 for r in results if r["expected"] in ("high", "critical"))
latencies = [r["latency"] for r in results]

print(f"\n=== RESULTS ({N_SAMPLES} samples) ===")
print(f"Tier accuracy:        {correct}/{N_SAMPLES} = {correct/N_SAMPLES:.0%}")
print(f"Schema compliance:    {schema_ok}/{N_SAMPLES} = {schema_ok/N_SAMPLES:.0%}")
if high_crit_total > 0:
    print(f"High+Critical recall: {high_crit_tp}/{high_crit_total} = {high_crit_tp/high_crit_total:.0%}")
print(f"Avg latency:          {sum(latencies)/len(latencies):.1f}s")
print(f"P95 latency:          {sorted(latencies)[int(0.95*len(latencies))]:.1f}s")

Path(__file__).parent.parent.joinpath("results").mkdir(exist_ok=True)
out_path = Path(__file__).parent.parent / "results" / "eval_20_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
