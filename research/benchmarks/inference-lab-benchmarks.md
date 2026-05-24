# Inference Lab Benchmark Results

Source: [qwen-inference-lab](https://github.com/quivent/qwen-inference-lab)

## Hardware: Apple M4 Max (16-core GPU, 128 GB, 546 GB/s)

| Configuration | tok/s | vs. baseline |
|---|---|---|
| Stock mlx_lm | 29.5 | 1.00x |
| V5 monolithic compile | 30.0 | 1.02x |
| Stock + spec decode (0.8B draft) | 37.6 | 1.27x |
| MTP head (self-speculative) | 36.9 | 1.25x |
| MTP + split-recurrence rollback | 42.7 | 1.45x |
| Adaptive MTP chain (Huihui abliterated) | 49.5 | 1.68x |
| Adaptive MTP chain (vanilla, revalidated) | **51.1** | **1.73x** |

## Dead Ends

- V6 custom Metal kernels: broke `mx.compile` fusion, slower than stock
- qmv_fast kernel tuning: register pressure killed occupancy
- group_size=128 quantization: 2.3x quant error for 2.8% speed
- GPU-resident AR loop: ~0% gain
- CPU draft model: 3716 ms/tok
- CoreML/ANE draft: `coremltools` broken on Python 3.14, DeltaNet ops unsupported

## Theoretical Limits

- Model: Qwen3.5-27B-4bit (13.7 GB total weights)
- Theoretical minimum: 25.1 ms/tok (39.8 tok/s at 100% BW utilization)
