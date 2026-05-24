"""
mlx-qwen-mtp: Multi-Token Prediction inference for Qwen3.5 on Apple Silicon.

First working MTP inference implementation. Every other framework strips MTP
weights on load. We reverse-engineered the architecture and built working inference.

Exports:
    MTPHead      - The MTP head nn.Module (one full transformer layer)
    load_mtp     - Load quantized MTP weights and attach to model
    mtp_generate - Generation loop with split-recurrence rollback (42.7 tok/s)
    patch_model  - Set up fused DeltaNet/attention kernels for maximum throughput
"""

from .mtp_head import MTPHead, load_mtp
from .generate import mtp_generate, patch_model

__all__ = ["MTPHead", "load_mtp", "mtp_generate", "patch_model"]
__version__ = "0.1.0"
