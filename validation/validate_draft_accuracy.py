"""Validate that DeltaNet-only forward matches full model output.

Usage:
    python validate_draft_accuracy.py [--model PATH] [--tokens N]
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="huihui-ai/Huihui-Qwen3.5-27B-abliterated")
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    layers = model.model.layers

    prompts = [
        "The theory of relativity states that",
        "def fibonacci(n):\n    ",
        "In 1969, humans first",
        "The chemical formula for water is",
        "Once upon a time in a dark forest,",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_v = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False)

        for layer in layers:
            if getattr(layer, 'layer_type', '') == "full_attention":
                layer._skip_attention = True
        with torch.no_grad():
            out_d = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False)
        for layer in layers:
            if getattr(layer, 'layer_type', '') == "full_attention":
                layer._skip_attention = False

        verify_toks = out_v[0][inputs.input_ids.shape[1]:].tolist()
        draft_toks = out_d[0][inputs.input_ids.shape[1]:].tolist()
        matches = sum(1 for v, d in zip(verify_toks, draft_toks) if v == d)
        first_div = next((i for i, (v, d) in enumerate(zip(verify_toks, draft_toks)) if v != d), len(verify_toks))

        print(f"'{prompt[:45]}...' -> {matches}/{len(verify_toks)} match ({100*matches/len(verify_toks):.0f}%), first diverge at token {first_div}")

if __name__ == "__main__":
    main()
