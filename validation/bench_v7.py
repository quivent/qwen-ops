"""
V7 Benchmark: Speculative Decoding for Qwen3.5-27B on MLX

Compares:
  1. Stock mlx_lm (baseline)
  2. V5 monolithic compile (patch_model)
  3. Stock + speculative decoding (draft model, sweep n=2..6)
  4. V5 + speculative decoding

Draft model: mlx-community/Qwen3.5-0.8B-MLX-4bit (same tokenizer family)
"""

import time
import json
import mlx.core as mx
import mlx_lm

MAIN_MODEL = "mlx-community/Qwen3.5-27B-4bit"
DRAFT_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-4bit"

PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."
MAX_TOKENS = 256
WARMUP_TOKENS = 16


def measure_memory():
    mx.metal.reset_peak_memory()
    mx.eval(mx.zeros(1))
    return mx.metal.get_peak_memory() / 1e9


def bench_generate(model, tokenizer, prompt, max_tokens, draft_model=None, num_draft_tokens=None, label=""):
    """Run generation and measure throughput."""
    mx.metal.reset_peak_memory()

    kwargs = {"max_tokens": max_tokens, "verbose": False}
    if draft_model is not None:
        kwargs["draft_model"] = draft_model
        kwargs["num_draft_tokens"] = num_draft_tokens

    # Warmup
    _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=WARMUP_TOKENS, verbose=False)
    mx.metal.reset_peak_memory()

    # Timed run
    t0 = time.perf_counter()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt, **kwargs)
    mx.eval(mx.zeros(1))  # sync
    t1 = time.perf_counter()

    elapsed = t1 - t0
    # Count output tokens
    out_tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
    if out_tokens <= 0:
        out_tokens = max_tokens  # fallback estimate
    tps = out_tokens / elapsed
    peak_mem = mx.metal.get_peak_memory() / 1e9

    return {
        "label": label,
        "tokens": out_tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(tps, 1),
        "peak_mem_gb": round(peak_mem, 2),
        "output_preview": output[:200],
    }


def bench_stream(model, tokenizer, prompt, max_tokens, draft_model=None, num_draft_tokens=None, label=""):
    """Run streaming generation to measure per-token timing and acceptance rate."""
    kwargs = {"max_tokens": max_tokens}
    if draft_model is not None:
        kwargs["draft_model"] = draft_model
        kwargs["num_draft_tokens"] = num_draft_tokens

    # Warmup
    for _ in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=WARMUP_TOKENS):
        pass
    mx.metal.reset_peak_memory()

    t0 = time.perf_counter()
    n_tokens = 0
    n_draft = 0
    n_total = 0
    prompt_tps = 0.0

    for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, **kwargs):
        n_total += 1
        if resp.from_draft:
            n_draft += 1
        if resp.prompt_tps > 0:
            prompt_tps = resp.prompt_tps
        if resp.finish_reason:
            break

    t1 = time.perf_counter()
    elapsed = t1 - t0
    tps = n_total / elapsed if elapsed > 0 else 0
    accept_rate = n_draft / n_total if n_total > 0 else 0
    peak_mem = mx.metal.get_peak_memory() / 1e9

    return {
        "label": label,
        "tokens": n_total,
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(tps, 1),
        "accept_rate": round(accept_rate, 3),
        "draft_tokens": n_draft,
        "prompt_tps": round(prompt_tps, 1),
        "peak_mem_gb": round(peak_mem, 2),
    }


def main():
    print("=" * 70)
    print("V7 Benchmark: Speculative Decoding for Qwen3.5-27B")
    print("=" * 70)

    # Load main model
    print(f"\nLoading main model: {MAIN_MODEL}")
    model, tokenizer = mlx_lm.load(MAIN_MODEL)
    print(f"  Peak memory after load: {mx.metal.get_peak_memory() / 1e9:.1f} GB")

    results = []

    # --- Benchmark 1: Stock baseline ---
    print("\n[1/6] Stock mlx_lm baseline...")
    r = bench_stream(model, tokenizer, PROMPT, MAX_TOKENS, label="Stock baseline")
    results.append(r)
    print(f"  {r['tok_per_s']} tok/s, {r['peak_mem_gb']} GB")

    # --- Benchmark 2: V5 monolithic compile ---
    print("\n[2/6] V5 monolithic compile...")
    from fused_gdn import patch_model, unpatch_model
    patch_model(model)
    r = bench_stream(model, tokenizer, PROMPT, MAX_TOKENS, label="V5 monolithic")
    results.append(r)
    print(f"  {r['tok_per_s']} tok/s, {r['peak_mem_gb']} GB")
    unpatch_model(model)

    # Load draft model
    print(f"\nLoading draft model: {DRAFT_MODEL}")
    draft_model, _ = mlx_lm.load(DRAFT_MODEL)
    print(f"  Peak memory after draft load: {mx.metal.get_peak_memory() / 1e9:.1f} GB")

    # --- Benchmark 3: Stock + spec decode, sweep n ---
    for n in [2, 3, 4, 5, 6]:
        print(f"\n[3.{n}/6] Stock + spec decode (n={n})...")
        r = bench_stream(model, tokenizer, PROMPT, MAX_TOKENS,
                        draft_model=draft_model, num_draft_tokens=n,
                        label=f"Stock + spec(n={n})")
        results.append(r)
        print(f"  {r['tok_per_s']} tok/s, accept={r['accept_rate']:.1%}, {r['peak_mem_gb']} GB")

    # --- Benchmark 4: V5 + spec decode (best n from above) ---
    best_spec = max([r for r in results if "spec" in r["label"]], key=lambda r: r["tok_per_s"])
    best_n = int(best_spec["label"].split("n=")[1].rstrip(")"))
    print(f"\n[4/6] V5 + spec decode (n={best_n}, best from sweep)...")
    patch_model(model)
    r = bench_stream(model, tokenizer, PROMPT, MAX_TOKENS,
                    draft_model=draft_model, num_draft_tokens=best_n,
                    label=f"V5 + spec(n={best_n})")
    results.append(r)
    print(f"  {r['tok_per_s']} tok/s, accept={r['accept_rate']:.1%}, {r['peak_mem_gb']} GB")
    unpatch_model(model)

    # --- Correctness check ---
    print("\n[5/6] Correctness check (greedy decode)...")
    test_prompt = "The meaning of life is"
    out_stock = mlx_lm.generate(model, tokenizer, prompt=test_prompt, max_tokens=50, verbose=False, temperature=0.0)
    out_spec = mlx_lm.generate(model, tokenizer, prompt=test_prompt, max_tokens=50, verbose=False,
                               draft_model=draft_model, num_draft_tokens=3, temp=0.0)
    match = out_stock == out_spec
    print(f"  Greedy match: {match}")
    if not match:
        print(f"  Stock:  {out_stock[:100]}")
        print(f"  Spec:   {out_spec[:100]}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30} {'tok/s':>8} {'Accept':>8} {'Memory':>8}")
    print("-" * 70)
    baseline_tps = results[0]["tok_per_s"]
    for r in results:
        accept = f"{r.get('accept_rate', 0):.1%}" if r.get('accept_rate', 0) > 0 else "N/A"
        speedup = r["tok_per_s"] / baseline_tps if baseline_tps > 0 else 0
        print(f"{r['label']:<30} {r['tok_per_s']:>7.1f} {accept:>8} {r['peak_mem_gb']:>7.1f}G  ({speedup:.2f}x)")

    # Theoretical analysis
    print(f"\nTheoretical bandwidth min: 40.8 tok/s (13.4 GB @ 546 GB/s)")
    print(f"Achievable at 85% sustained: 34.6 tok/s")

    # Save results
    with open("bench_v7_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to bench_v7_results.json")


if __name__ == "__main__":
    main()
