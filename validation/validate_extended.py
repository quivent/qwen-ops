"""Extended validation: DeltaNet-only forward accuracy vs full model on Qwen3.5-27B.

Tests 100 tokens across 12 diverse prompts. Reports per-prompt match rate,
overall statistics, and first divergence point per prompt.
"""
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated"
MAX_NEW_TOKENS = 100

PROMPTS = [
    # Code generation - Python
    "def merge_sort(arr):\n    ",
    # Code generation - JavaScript
    "function debounce(fn, delay) {\n  ",
    # Math reasoning
    "Solve step by step: If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance?",
    # Math reasoning 2
    "What is the derivative of f(x) = x^3 * ln(x)? Show your work:",
    # Creative writing
    "The last human on Earth sat alone in a room. There was a knock at the door.",
    # Creative writing 2
    "Write a haiku about the ocean:\n",
    # Factual Q&A
    "What are the three laws of thermodynamics? Briefly explain each:",
    # Factual Q&A 2
    "Name the planets in our solar system in order from the Sun:",
    # Multi-turn conversation format
    "<|im_start|>user\nWhat is photosynthesis?<|im_end|>\n<|im_start|>assistant\n",
    # Multi-turn conversation 2
    "<|im_start|>user\nExplain recursion to a 5 year old.<|im_end|>\n<|im_start|>assistant\n",
    # Non-English - Chinese
    "用中文解释什么是人工智能：",
    # Non-English - French
    "Expliquez en français le concept de la relativité générale:",
]

def log(msg=""):
    print(msg, flush=True)

def set_skip_attention(layers, skip: bool):
    for layer in layers:
        if getattr(layer, "layer_type", "") == "full_attention":
            layer._skip_attention = skip

def main():
    log(f"Loading model from {MODEL_PATH} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")

    layers = model.model.layers

    # Count layer types
    full_attn = sum(1 for l in layers if getattr(l, "layer_type", "") == "full_attention")
    delta = sum(1 for l in layers if getattr(l, "layer_type", "") != "full_attention")
    log(f"Layers: {len(layers)} total, {full_attn} full_attention, {delta} deltanet/other")
    log(f"Generating {MAX_NEW_TOKENS} tokens per prompt, {len(PROMPTS)} prompts")
    log("=" * 90)

    results = []

    for idx, prompt in enumerate(PROMPTS):
        label = prompt[:60].replace("\n", "\\n")
        log(f"\n[{idx+1}/{len(PROMPTS)}] \"{label}...\"")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs.input_ids.shape[1]

        # --- Full model (verify) ---
        set_skip_attention(layers, False)
        t1 = time.time()
        with torch.no_grad():
            out_v = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        t_verify = time.time() - t1
        log(f"  verify done in {t_verify:.1f}s")

        # --- DeltaNet-only (draft) ---
        set_skip_attention(layers, True)
        t2 = time.time()
        with torch.no_grad():
            out_d = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        t_draft = time.time() - t2
        log(f"  draft done in {t_draft:.1f}s")

        # Reset
        set_skip_attention(layers, False)

        verify_toks = out_v[0][input_len:].tolist()
        draft_toks = out_d[0][input_len:].tolist()

        # Compare (handle different lengths)
        min_len = min(len(verify_toks), len(draft_toks))
        matches = sum(1 for i in range(min_len) if verify_toks[i] == draft_toks[i])
        total = max(len(verify_toks), len(draft_toks))

        # First divergence
        first_div = None
        for i in range(min_len):
            if verify_toks[i] != draft_toks[i]:
                first_div = i
                break
        if first_div is None and len(verify_toks) != len(draft_toks):
            first_div = min_len

        match_pct = 100 * matches / total if total > 0 else 100.0

        results.append({
            "prompt": label,
            "matches": matches,
            "total": total,
            "pct": match_pct,
            "first_div": first_div,
            "verify_len": len(verify_toks),
            "draft_len": len(draft_toks),
            "t_verify": t_verify,
            "t_draft": t_draft,
        })

        div_str = f"token {first_div}" if first_div is not None else "NONE (perfect match)"
        log(f"  Match: {matches}/{total} ({match_pct:.1f}%)  |  First divergence: {div_str}")
        log(f"  Time: verify={t_verify:.1f}s  draft={t_draft:.1f}s")

        if first_div is not None and first_div < total:
            # Show tokens around divergence
            ctx_start = max(0, first_div - 2)
            ctx_end = min(min_len, first_div + 3)
            v_ctx = tokenizer.decode(verify_toks[ctx_start:ctx_end])
            d_ctx = tokenizer.decode(draft_toks[ctx_start:ctx_end])
            v_tok = tokenizer.decode([verify_toks[first_div]]) if first_div < len(verify_toks) else "<EOS>"
            d_tok = tokenizer.decode([draft_toks[first_div]]) if first_div < len(draft_toks) else "<EOS>"
            log(f"  Divergence detail @ token {first_div}: verify='{v_tok}' vs draft='{d_tok}'")
            log(f"    verify context: ...{repr(v_ctx)}")
            log(f"    draft  context: ...{repr(d_ctx)}")

    # Summary
    log("\n" + "=" * 90)
    log("SUMMARY")
    log("=" * 90)
    total_matches = sum(r["matches"] for r in results)
    total_tokens = sum(r["total"] for r in results)
    perfect = sum(1 for r in results if r["first_div"] is None)
    overall_pct = 100 * total_matches / total_tokens if total_tokens else 0

    log(f"Prompts tested: {len(results)}")
    log(f"Tokens per prompt: {MAX_NEW_TOKENS}")
    log(f"Overall match rate: {total_matches}/{total_tokens} ({overall_pct:.2f}%)")
    log(f"Perfect match prompts: {perfect}/{len(results)}")

    # Per-prompt table
    log(f"\n{'#':<3} {'Match%':>7} {'Matches':>8} {'1st Div':>8}  Prompt")
    log("-" * 90)
    for i, r in enumerate(results):
        div_s = str(r["first_div"]) if r["first_div"] is not None else "-"
        log(f"{i+1:<3} {r['pct']:>6.1f}% {r['matches']:>4}/{r['total']:<4} {div_s:>8}  {r['prompt'][:55]}")

    # Divergence histogram
    divs = [r["first_div"] for r in results if r["first_div"] is not None]
    if divs:
        log(f"\nDivergence points: {sorted(divs)}")
        log(f"Earliest divergence: token {min(divs)}")
        log(f"Latest divergence: token {max(divs)}")
    else:
        log("\nNo divergences found - 100% match across all prompts!")

    total_time = sum(r["t_verify"] + r["t_draft"] for r in results)
    log(f"\nTotal wall time (generation only): {total_time:.0f}s ({total_time/60:.1f}min)")

if __name__ == "__main__":
    main()
