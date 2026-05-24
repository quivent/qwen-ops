#!/usr/bin/env python3
"""Diagnose where hidden states diverge between normal and skip-attention modes.

Loads the Qwen3.5-27B model, runs a single prompt ("Hello") through it twice:
  1. Normal mode (verify): all layers active
  2. Draft mode: full_attention layers skip their attention (identity passthrough)

Compares hidden states after each layer to find where divergence first appears
and whether it originates at attention layers (expected) or DeltaNet layers
(the real problem indicating fused kernel numerical sensitivity).
"""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated"
PROMPT = "Hello"


def get_layers(model):
    """Navigate model hierarchy to find decoder layers."""
    # Qwen3_5ForConditionalGeneration: model.model.language_model.layers
    # Qwen3_5ForCausalLM: model.model.layers
    if hasattr(model, 'model'):
        m = model.model
        if hasattr(m, 'language_model'):
            return m.language_model.layers
        elif hasattr(m, 'layers'):
            return m.layers
    raise RuntimeError("Cannot find decoder layers in model")


def capture_hidden_states(model, input_ids, layers):
    """Run forward pass and capture hidden states after each layer via hooks."""
    captured = {}

    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                # output is the hidden_states tensor directly (not a tuple)
                if isinstance(output, tuple):
                    captured[idx] = output[0].detach().cpu().float()
                else:
                    captured[idx] = output.detach().cpu().float()
            return hook_fn
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return captured


def main():
    print("=" * 70)
    print("DIVERGENCE DIAGNOSTIC: Normal vs Skip-Attention")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    print(f"Prompt: '{PROMPT}' -> {input_ids.shape[1]} tokens: {input_ids.tolist()}")

    # Load model on GPU in FP16
    print("\nLoading model on GPU (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    num_layers = len(layers)
    print(f"Found {num_layers} decoder layers")

    # Identify layer types
    layer_types = []
    for i, layer in enumerate(layers):
        lt = getattr(layer, 'layer_type', 'unknown')
        layer_types.append(lt)

    attn_indices = [i for i, lt in enumerate(layer_types) if lt == 'full_attention']
    delta_indices = [i for i, lt in enumerate(layer_types) if lt == 'linear_attention']
    print(f"  full_attention layers: {attn_indices}")
    print(f"  linear_attention layers: {len(delta_indices)} total")

    # Move input to model device
    device = next(model.parameters()).device
    input_ids_dev = input_ids.to(device)

    # --- Pass 1: Normal (verify) mode ---
    print("\n--- Pass 1: Normal mode (all layers active) ---")
    normal_states = capture_hidden_states(model, input_ids_dev, layers)
    print(f"  Captured {len(normal_states)} layer outputs")

    # Reset any recurrent state by reloading -- actually, for a single forward
    # pass with no cache, the recurrent state is computed fresh each time.
    # But to be safe, let's clear caches.
    gc.collect()
    torch.cuda.empty_cache()

    # --- Pass 2: Draft mode (skip attention on full_attention layers) ---
    print("\n--- Pass 2: Draft mode (skip attention on full_attention layers) ---")

    # Monkey-patch full_attention layers to skip their attention
    original_forwards = {}
    for i in attn_indices:
        layer = layers[i]
        original_forwards[i] = layer.forward

        # Create a new forward that skips self_attn but keeps MLP
        def make_skip_forward(orig_layer):
            def skip_forward(hidden_states, **kwargs):
                # Skip attention: just residual (no attention contribution)
                residual = hidden_states
                # Still apply input_layernorm + skip attention = just residual
                # (the attention output would have been added to residual)
                hidden_states = residual  # identity for attention block

                # MLP block still runs
                residual2 = hidden_states
                hidden_states = orig_layer.post_attention_layernorm(hidden_states)
                hidden_states = orig_layer.mlp(hidden_states)
                hidden_states = residual2 + hidden_states
                return hidden_states
            return skip_forward

        layer.forward = make_skip_forward(layer)

    draft_states = capture_hidden_states(model, input_ids_dev, layers)
    print(f"  Captured {len(draft_states)} layer outputs")

    # Restore original forwards
    for i in attn_indices:
        layers[i].forward = original_forwards[i]

    # --- Compare ---
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 70)
    print(f"{'Layer':>5} {'Type':>18} {'MaxAbsDiff':>12} {'MeanAbsDiff':>12} {'Status'}")
    print("-" * 70)

    first_diverge = None
    first_deltanet_diverge = None

    for i in range(num_layers):
        if i not in normal_states or i not in draft_states:
            print(f"{i:>5} {'???':>18} {'N/A':>12} {'N/A':>12} MISSING")
            continue

        n = normal_states[i]
        d = draft_states[i]
        diff = (n - d).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        lt = layer_types[i]

        is_attn = lt == 'full_attention'

        if max_diff == 0:
            status = "IDENTICAL"
        elif max_diff < 1e-5:
            status = "~identical (fp noise)"
        elif is_attn:
            status = "EXPECTED (attn skip)"
        else:
            status = "*** DIVERGED ***"

        if max_diff > 0 and first_diverge is None:
            first_diverge = i

        if max_diff > 1e-5 and lt == 'linear_attention' and first_deltanet_diverge is None:
            first_deltanet_diverge = i

        print(f"{i:>5} {lt:>18} {max_diff:>12.6e} {mean_diff:>12.6e} {status}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if first_diverge is not None:
        print(f"First divergence at layer {first_diverge} ({layer_types[first_diverge]})")
    else:
        print("No divergence detected (all layers identical)")

    if first_deltanet_diverge is not None:
        print(f"First DeltaNet divergence at layer {first_deltanet_diverge}")
        prev_attn = max(a for a in attn_indices if a < first_deltanet_diverge)
        print(f"  This is layer {first_deltanet_diverge - prev_attn} after attention layer {prev_attn}")
        print(f"  => Confirms: fused DeltaNet kernels amplify upstream differences")
    else:
        print("No DeltaNet-layer divergence detected")

    # Additional: show the top-5 most divergent DeltaNet layers
    delta_diffs = []
    for i in delta_indices:
        if i in normal_states and i in draft_states:
            max_d = (normal_states[i] - draft_states[i]).abs().max().item()
            delta_diffs.append((i, max_d))
    delta_diffs.sort(key=lambda x: x[1], reverse=True)

    if delta_diffs:
        print(f"\nTop-5 most divergent DeltaNet layers:")
        for idx, (layer_i, max_d) in enumerate(delta_diffs[:5]):
            print(f"  #{idx+1}: Layer {layer_i}, max_abs_diff = {max_d:.6e}")

    # Show how divergence grows across layers
    print(f"\nDivergence growth pattern (DeltaNet layers only):")
    for i in delta_indices:
        if i in normal_states and i in draft_states:
            max_d = (normal_states[i] - draft_states[i]).abs().max().item()
            if max_d > 0:
                bar = "#" * min(50, int(max_d * 10))
                print(f"  L{i:>2}: {max_d:.4e} {bar}")


if __name__ == "__main__":
    main()
