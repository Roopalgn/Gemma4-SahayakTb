"""
SAHAYAK-TB — Evaluation script.

Evaluates base and fine-tuned Gemma 4 E2B on the held-out eval set.
Computes: risk tier recall/precision/F1, schema compliance, factor accuracy.

Usage (on Kaggle or local with llama-cpp-python):
  python evaluate.py --model base --eval-data ../data/eval.jsonl --output ../results/baseline_results.json
  python evaluate.py --model finetuned --model-path ./sahayak-tb-Q4_K_M.gguf --eval-data ../data/eval.jsonl --output ../results/finetuned_results.json
"""

import argparse
import json
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import RISK_ASSESSMENT_SCHEMA, SYSTEM_PROMPT, validate_risk_assessment
from inference.postprocess import correct_output


def load_eval_data(path: str) -> list[dict]:
    """Load eval JSONL file, return list of {input, expected_output}."""
    data = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            messages = entry["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            assistant_msg = next(m for m in messages if m["role"] == "assistant")
            data.append({
                "input": user_msg["content"],
                "expected": json.loads(assistant_msg["content"]),
            })
    return data


def run_inference_llamacpp(model_path: str, eval_data: list[dict], schema: dict) -> list[dict]:
    """Run inference using llama-cpp-python."""
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=0,  # CPU only for benchmarking
        n_ctx=4096,
        verbose=False,
    )

    results = []
    for i, sample in enumerate(eval_data):
        start_time = time.time()
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sample["input"]},
                ],
                response_format={
                    "type": "json_object",
                    "schema": schema,
                },
                temperature=0.3,
                max_tokens=1024,
            )
            latency = time.time() - start_time
            raw_output = response["choices"][0]["message"]["content"]

            try:
                parsed = json.loads(raw_output)
                # Apply post-processing: citation grounding, dedup, tier correction
                parsed = correct_output(parsed)
                schema_valid, errors = validate_risk_assessment(parsed)
            except json.JSONDecodeError:
                parsed = None
                schema_valid = False
                errors = ["Output is not valid JSON"]

            results.append({
                "index": i,
                "input_preview": sample["input"][:100],
                "expected_tier": sample["expected"]["dropout_risk_tier"],
                "predicted_tier": parsed.get("dropout_risk_tier") if parsed else None,
                "expected_factors": len(sample["expected"]["risk_factors"]),
                "predicted_factors": len(parsed.get("risk_factors", [])) if parsed else 0,
                "schema_valid": schema_valid,
                "schema_errors": errors,
                "latency_seconds": round(latency, 2),
                "raw_output": raw_output[:500],
            })

        except Exception as e:
            latency = time.time() - start_time
            results.append({
                "index": i,
                "input_preview": sample["input"][:100],
                "expected_tier": sample["expected"]["dropout_risk_tier"],
                "predicted_tier": None,
                "schema_valid": False,
                "schema_errors": [str(e)],
                "latency_seconds": round(latency, 2),
                "raw_output": "",
            })

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(eval_data)}")

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    tiers = ["low", "moderate", "high", "critical"]
    total = len(results)

    # Schema compliance
    schema_ok = sum(1 for r in results if r["schema_valid"])

    # Tier accuracy
    tier_correct = sum(1 for r in results if r["predicted_tier"] == r["expected_tier"])

    # Per-tier recall and precision
    per_tier = {}
    for tier in tiers:
        tp = sum(1 for r in results if r["expected_tier"] == tier and r["predicted_tier"] == tier)
        fn = sum(1 for r in results if r["expected_tier"] == tier and r["predicted_tier"] != tier)
        fp = sum(1 for r in results if r["expected_tier"] != tier and r["predicted_tier"] == tier)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_tier[tier] = {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

    # Combined high+critical recall (the clinically critical metric)
    high_crit_tp = sum(1 for r in results
                       if r["expected_tier"] in ("high", "critical")
                       and r["predicted_tier"] in ("high", "critical"))
    high_crit_total = sum(1 for r in results
                          if r["expected_tier"] in ("high", "critical"))
    high_crit_recall = high_crit_tp / high_crit_total if high_crit_total > 0 else 0

    # Latency stats
    latencies = [r["latency_seconds"] for r in results]
    latencies.sort()

    return {
        "total_samples": total,
        "schema_compliance": round(schema_ok / total, 4) if total > 0 else 0,
        "schema_compliance_count": f"{schema_ok}/{total}",
        "tier_accuracy": round(tier_correct / total, 4) if total > 0 else 0,
        "high_critical_recall": round(high_crit_recall, 4),
        "per_tier_metrics": per_tier,
        "latency": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p50": round(latencies[len(latencies) // 2], 2) if latencies else 0,
            "p95": round(latencies[int(len(latencies) * 0.95)], 2) if latencies else 0,
            "p99": round(latencies[int(len(latencies) * 0.99)], 2) if latencies else 0,
            "min": round(min(latencies), 2) if latencies else 0,
            "max": round(max(latencies), 2) if latencies else 0,
        },
    }


def print_comparison(baseline: dict, finetuned: dict) -> str:
    """Print a comparison table of baseline vs finetuned metrics."""
    lines = []
    lines.append("=" * 70)
    lines.append("SAHAYAK-TB — Evaluation Comparison: Base vs Fine-Tuned")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<35} {'Base':>12} {'Fine-Tuned':>12} {'Delta':>10}")
    lines.append("-" * 70)

    def row(label, base_val, ft_val, fmt=".1%"):
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        lines.append(f"{label:<35} {format(base_val, fmt):>12} {format(ft_val, fmt):>12} {sign}{format(delta, fmt):>9}")

    row("Schema Compliance", baseline["schema_compliance"], finetuned["schema_compliance"])
    row("Tier Accuracy", baseline["tier_accuracy"], finetuned["tier_accuracy"])
    row("High+Critical Recall", baseline["high_critical_recall"], finetuned["high_critical_recall"])

    for tier in ["low", "moderate", "high", "critical"]:
        b = baseline["per_tier_metrics"][tier]
        f = finetuned["per_tier_metrics"][tier]
        row(f"  {tier} recall", b["recall"], f["recall"])
        row(f"  {tier} precision", b["precision"], f["precision"])

    lines.append("-" * 70)
    row("Avg Latency (s)", baseline["latency"]["mean"], finetuned["latency"]["mean"], ".2f")
    row("P95 Latency (s)", baseline["latency"]["p95"], finetuned["latency"]["p95"], ".2f")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAHAYAK-TB model")
    parser.add_argument("--model", choices=["base", "finetuned"], required=True)
    parser.add_argument("--model-path", type=str, help="Path to GGUF model file")
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--compare-with", type=str, help="Path to baseline results JSON for comparison")
    args = parser.parse_args()

    print(f"Loading eval data from {args.eval_data}...")
    eval_data = load_eval_data(args.eval_data)
    print(f"Loaded {len(eval_data)} eval samples")

    if not args.model_path:
        print("ERROR: --model-path is required (path to GGUF file)")
        sys.exit(1)

    print(f"Running inference with {args.model} model: {args.model_path}")
    results = run_inference_llamacpp(args.model_path, eval_data, RISK_ASSESSMENT_SCHEMA)

    print("Computing metrics...")
    metrics = compute_metrics(results)

    output = {
        "model_type": args.model,
        "model_path": args.model_path,
        "eval_data": args.eval_data,
        "metrics": metrics,
        "per_sample_results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Schema compliance: {metrics['schema_compliance']:.1%}")
    print(f"Tier accuracy: {metrics['tier_accuracy']:.1%}")
    print(f"High+Critical recall: {metrics['high_critical_recall']:.1%}")
    print(f"Avg latency: {metrics['latency']['mean']:.2f}s")

    # Print comparison if baseline provided
    if args.compare_with:
        with open(args.compare_with) as f:
            baseline = json.load(f)
        print("\n" + print_comparison(baseline["metrics"], metrics))


if __name__ == "__main__":
    main()
