#!/usr/bin/env python3
"""
Test partial attention skip configurations for Qwen3.5-27B.

Measures how many full_attention layers can be skipped during draft mode
while maintaining useful token prediction accuracy.
"""

import torch
import time
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated"

# The 16 full_attention layer indices
ALL_ATTN_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

# Configurations to test: name -> list of indices to SKIP
CONFIGS = OrderedDict([
    ("skip_NONE (baseline)", []),
    ("skip_middle_8", [19, 23, 27, 31, 35, 39, 43, 47]),
    ("skip_middle_12", [11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55]),
    ("skip_all_but_first_last", [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59]),
    ("skip_alternating", [7, 15, 23, 31, 39, 47, 55, 63]),
    ("skip_ALL_16 (worst case)", ALL_ATTN_INDICES),
])

PROMPTS = [
    "The theory of relativity states that",
    "def fibonacci(n):\n    ",
    "Name the planets in our solar system:",
]

NUM_TOKENS = 50


def monkey_patch_decoder_layer(layer, layer_idx):
    """
    Monkey-patch a decoder layer's forward so that when layer._skip_attention is True
    and it's a full_attention layer, it skips the self_attn call (residual only for that part).
    """
    original_forward = layer.forward

    def patched_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        skip = getattr(layer, '_skip_attention', False)

        if skip and layer.layer_type == "full_attention":
            # Skip attention: just do residual (no attention contribution)
            residual = hidden_states
            # Still run input_layernorm + skip attention = residual only
            # hidden_states = residual (attention output is zero)
            hidden_states = residual

            # Still run MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states
        else:
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

    layer.forward = patched_forward


def set_skip_config(model, skip_indices):
    """Set _skip_attention flag on specified layer indices."""
    skip_set = set(skip_indices) if not isinstance(skip_indices, set) else skip_indices
    for i, layer in enumerate(model.model.layers):
        layer._skip_attention = (i in skip_set)


def generate_tokens(model, tokenizer, prompt, num_tokens=50):
    """Generate tokens greedily, returning list of token IDs."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = outputs[0][input_len:].tolist()
    return generated_ids


def compare_tokens(baseline, test):
    """Compare two token sequences, return match rate and first divergence."""
    min_len = min(len(baseline), len(test))
    matches = 0
    first_div = None
    for i in range(min_len):
        if baseline[i] == test[i]:
            matches += 1
        elif first_div is None:
            first_div = i

    match_rate = matches / min_len if min_len > 0 else 0.0
    return match_rate, first_div, min_len


def main():
    print("=" * 80)
    print("PARTIAL ATTENTION SKIP TEST - Qwen3.5-27B")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    # Check model structure - model.model IS the TextModel directly
    text_model = model.model
    num_layers = len(text_model.layers)
    print(f"Total layers: {num_layers}")
    full_attn_layers = [i for i, l in enumerate(text_model.layers) if l.layer_type == "full_attention"]
    print(f"Full attention layers ({len(full_attn_layers)}): {full_attn_layers}")
    print()

    # Monkey-patch all layers
    print("Monkey-patching decoder layers...")
    for i, layer in enumerate(text_model.layers):
        monkey_patch_decoder_layer(layer, i)
    print("Done.")
    print()

    # Run baseline first
    print("=" * 80)
    print("GENERATING BASELINE (no skip)")
    print("=" * 80)
    set_skip_config(model, [])
    baseline_results = {}
    for prompt in PROMPTS:
        short_prompt = prompt[:50].replace('\n', '\\n')
        print(f"  Prompt: '{short_prompt}...'")
        tokens = generate_tokens(model, tokenizer, prompt, NUM_TOKENS)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        baseline_results[prompt] = tokens
        print(f"  Output: {text[:100]}...")
        print()

    # Test each config
    all_results = {}
    for config_name, skip_indices in CONFIGS.items():
        if "baseline" in config_name:
            continue

        print("=" * 80)
        print(f"CONFIG: {config_name}")
        print(f"  Skipping {len(skip_indices)} attention layers: {skip_indices}")
        kept = [i for i in ALL_ATTN_INDICES if i not in skip_indices]
        print(f"  Keeping {len(kept)} attention layers: {kept}")
        print(f"  Compute savings: {len(skip_indices)}/16 = {len(skip_indices)/16*100:.0f}% of full attention skipped")
        print("=" * 80)

        set_skip_config(model, set(skip_indices))

        config_results = {}
        for prompt in PROMPTS:
            short_prompt = prompt[:50].replace('\n', '\\n')
            print(f"  Prompt: '{short_prompt}...'")

            t0 = time.time()
            tokens = generate_tokens(model, tokenizer, prompt, NUM_TOKENS)
            gen_time = time.time() - t0

            text = tokenizer.decode(tokens, skip_special_tokens=True)
            baseline_tokens = baseline_results[prompt]
            match_rate, first_div, length = compare_tokens(baseline_tokens, tokens)

            config_results[prompt] = {
                "tokens": tokens,
                "text": text,
                "match_rate": match_rate,
                "first_div": first_div,
                "length": length,
                "gen_time": gen_time,
            }

            print(f"    Match rate: {match_rate*100:.1f}%")
            print(f"    First divergence: token {first_div}")
            print(f"    Gen time: {gen_time:.2f}s")
            print(f"    Output: {text[:100]}...")
            print()

        all_results[config_name] = config_results

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'Config':<30} {'Layers Skipped':<16} {'Avg Match%':<12} {'Avg 1st Div':<14} {'Compute Save':<14}")
    print("-" * 86)

    for config_name, skip_indices in CONFIGS.items():
        if "baseline" in config_name:
            print(f"{'baseline (no skip)':<30} {'0/16':<16} {'100.0%':<12} {'N/A':<14} {'0%':<14}")
            continue

        results = all_results[config_name]
        avg_match = sum(r["match_rate"] for r in results.values()) / len(results)
        divs = [r["first_div"] for r in results.values() if r["first_div"] is not None]
        avg_div = sum(divs) / len(divs) if divs else float('inf')
        avg_div_str = f"{avg_div:.1f}" if divs else "N/A (100%)"
        save_pct = len(skip_indices) / 16 * 100

        print(f"{config_name:<30} {f'{len(skip_indices)}/16':<16} {f'{avg_match*100:.1f}%':<12} {avg_div_str:<14} {f'{save_pct:.0f}%':<14}")

    print()
    print("=" * 80)
    print("DETAILED TOKEN COMPARISON")
    print("=" * 80)

    for prompt in PROMPTS:
        short_prompt = prompt[:50].replace('\n', '\\n')
        print(f"\nPrompt: '{short_prompt}...'")
        baseline_tokens = baseline_results[prompt]
        baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
        print(f"  Baseline: {baseline_text[:120]}")

        for config_name in all_results:
            result = all_results[config_name][prompt]
            print(f"  {config_name}: {result['text'][:120]}")
            # Show token-by-token comparison for first 20 tokens
            diffs = []
            for i in range(min(20, len(baseline_tokens), len(result['tokens']))):
                if baseline_tokens[i] != result['tokens'][i]:
                    b_tok = tokenizer.decode([baseline_tokens[i]])
                    t_tok = tokenizer.decode([result['tokens'][i]])
                    diffs.append(f"    pos {i}: baseline='{b_tok}' vs '{t_tok}'")
            if diffs:
                print(f"    First diffs (of {20} tokens):")
                for d in diffs[:5]:
                    print(d)
        print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    best_config = None
    best_score = -1
    for config_name, skip_indices in CONFIGS.items():
        if "baseline" in config_name:
            continue
        results = all_results[config_name]
        avg_match = sum(r["match_rate"] for r in results.values()) / len(results)
        # Score: match_rate * savings (want high match AND high savings)
        savings = len(skip_indices) / 16
        # Only consider configs with >50% match rate
        if avg_match >= 0.5:
            score = avg_match * savings
            if score > best_score:
                best_score = score
                best_config = config_name

    if best_config:
        skip_indices = CONFIGS[best_config]
        results = all_results[best_config]
        avg_match = sum(r["match_rate"] for r in results.values()) / len(results)
        print(f"Best config: {best_config}")
        print(f"  Match rate: {avg_match*100:.1f}%")
        print(f"  Attention layers skipped: {len(skip_indices)}/16 ({len(skip_indices)/16*100:.0f}% savings)")
        print(f"  Score (match * savings): {best_score:.3f}")
    else:
        print("No config achieved >= 50% match rate.")
        print("Full attention skip causes too much divergence for useful drafting.")
        # Find the best match rate anyway
        best_match = 0
        best_name = None
        for config_name in all_results:
            results = all_results[config_name]
            avg_match = sum(r["match_rate"] for r in results.values()) / len(results)
            if avg_match > best_match:
                best_match = avg_match
                best_name = config_name
        if best_name:
            print(f"Highest match rate: {best_name} at {best_match*100:.1f}%")


if __name__ == "__main__":
    main()
