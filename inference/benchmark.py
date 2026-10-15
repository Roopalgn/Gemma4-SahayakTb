"""
SAHAYAK-TB — GGUF export and llama.cpp CPU benchmark.

Benchmarks inference latency and throughput for the fine-tuned model
on CPU, suitable for documenting llama.cpp prize eligibility.

Usage:
  python benchmark.py --model ./sahayak-tb-Q4_K_M.gguf --eval-data ../data/eval.jsonl --output ../results/benchmark.json
  python benchmark.py --model ./sahayak-tb-Q4_K_M.gguf --quick  # 5 samples only
"""

import argparse
import json
import time
import sys
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import RISK_ASSESSMENT_SCHEMA, SYSTEM_PROMPT
from inference.postprocess import correct_output


def get_system_info() -> dict:
    """Gather system information for benchmark context."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        pass
    return info


def benchmark_model(model_path: str, eval_data: list[dict], n_threads: int = 0) -> dict:
    """Run benchmark: latency, tokens/sec, memory."""
    from llama_cpp import Llama

    print(f"Loading model: {model_path}")
    load_start = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=0,    # CPU only
        n_ctx=4096,
        n_threads=n_threads if n_threads > 0 else None,
        verbose=False,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    results = []
    total_tokens = 0

    for i, sample in enumerate(eval_data):
        start = time.time()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["input"]},
            ],
            response_format={
                "type": "json_object",
                "schema": RISK_ASSESSMENT_SCHEMA,
            },
            temperature=0.3,
            max_tokens=1024,
        )
        latency = time.time() - start

        content = response["choices"][0]["message"]["content"]
        # Apply post-processing (tier correction, dedup, citation grounding)
        try:
            parsed = json.loads(content)
            parsed = correct_output(parsed)
            content = json.dumps(parsed)
        except (json.JSONDecodeError, Exception):
            pass  # Benchmark still records raw latency even if parsing fails
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(content.split()) * 2)  # rough estimate
        total_tokens += completion_tokens

        results.append({
            "index": i,
            "latency_seconds": round(latency, 3),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(completion_tokens / latency, 1) if latency > 0 else 0,
            "output_length": len(content),
        })

        print(f"  Sample {i+1}/{len(eval_data)}: {latency:.2f}s, {completion_tokens} tokens")

    # Aggregate stats
    latencies = [r["latency_seconds"] for r in results]
    tps_values = [r["tokens_per_second"] for r in results]
    latencies.sort()

    return {
        "model_path": model_path,
        "model_size_mb": round(Path(model_path).stat().st_size / 1e6, 1),
        "load_time_seconds": round(load_time, 2),
        "num_samples": len(results),
        "total_tokens": total_tokens,
        "latency": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "median": round(latencies[len(latencies) // 2], 3),
            "p95": round(latencies[int(len(latencies) * 0.95)], 3),
            "p99": round(latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)], 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "throughput": {
            "mean_tokens_per_second": round(sum(tps_values) / len(tps_values), 1),
            "total_time_seconds": round(sum(latencies), 1),
        },
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SAHAYAK-TB model via llama.cpp")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--eval-data", type=str, default="../data/eval.jsonl")
    parser.add_argument("--output", type=str, default="../results/benchmark.json")
    parser.add_argument("--quick", action="store_true", help="Run only 5 samples")
    parser.add_argument("--threads", type=int, default=0, help="Number of CPU threads (0=auto)")
    args = parser.parse_args()

    # Load eval data
    eval_entries = []
    with open(args.eval_data) as f:
        for line in f:
            entry = json.loads(line)
            messages = entry["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            eval_entries.append({"input": user_msg["content"]})

    if args.quick:
        eval_entries = eval_entries[:5]

    print(f"Benchmarking {len(eval_entries)} samples on CPU...")
    print(f"System: {platform.platform()}")

    sys_info = get_system_info()
    bench_results = benchmark_model(args.model, eval_entries, args.threads)

    output = {
        "system_info": sys_info,
        "benchmark": bench_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"BENCHMARK RESULTS — {args.model}")
    print(f"{'=' * 50}")
    print(f"Model size: {bench_results['model_size_mb']} MB")
    print(f"Load time: {bench_results['load_time_seconds']}s")
    print(f"Samples: {bench_results['num_samples']}")
    print(f"Avg latency: {bench_results['latency']['mean']:.2f}s")
    print(f"P95 latency: {bench_results['latency']['p95']:.2f}s")
    print(f"Avg throughput: {bench_results['throughput']['mean_tokens_per_second']:.1f} tok/s")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
