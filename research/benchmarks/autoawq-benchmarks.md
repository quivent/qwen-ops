# AutoAWQ Qwen3.5 Benchmark Results

Source: [autoawq-qwen35](https://github.com/quivent/autoawq-qwen35)

## Hardware: RTX 5090, vLLM 0.19.0, MTP=5

| Metric | GPTQ W4A16 | AWQ W4A16 |
|---|---:|---:|
| Single 256 tok | **151 tok/s** | 77 tok/s |
| MTP acceptance | **51%** | 31% |
| Batch=4 agg | **347 tok/s** | 313 tok/s |
| Model size | 19.5 GB | 18.6 GB |

## Key Finding

GPTQ's Hessian-optimal rounding preserves MTP head quality better than AWQ's activation-aware approach, resulting in higher MTP acceptance and throughput. For MTP-enabled serving, GPTQ is currently recommended. AWQ may perform better for non-MTP workloads.

## Quantization Notes

- `modules_to_not_convert = ["visual", "mtp", "in_proj_b", "in_proj_a"]` -- vision encoder and MTP head kept at full precision
- GDN beta/alpha projections excluded (48 out_features not divisible by pack_num=8)
- AWQ drops MTP weights during save; post-quantization injection via `inject_mtp_weights.py` required
