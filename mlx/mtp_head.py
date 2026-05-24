"""
MTPHead: Multi-Token Prediction head for Qwen3.5-27B.

Architecture (reverse-engineered from HF checkpoint):
  1. RMSNorm hidden + RMSNorm embedding -> concat [embed, hidden] -> fc projection
  2. One gated-attention transformer layer (same arch as Qwen3.5 attention)
  3. RMSNorm -> shared lm_head -> logits

15 weight tensors total:
  - pre_fc_norm_hidden.weight, pre_fc_norm_embedding.weight
  - fc.weight (10240 -> 5120)
  - input_layernorm.weight
  - q_proj.weight (5120 -> 12288), k_proj.weight (5120 -> 1024),
    v_proj.weight (5120 -> 1024), o_proj.weight (6144 -> 5120)
  - q_norm.weight, k_norm.weight
  - post_attention_layernorm.weight
  - gate_proj.weight (5120 -> 17408), up_proj.weight (5120 -> 17408),
    down_proj.weight (17408 -> 5120)
  - norm.weight (final, before shared lm_head)

Key discovery: concat order is [embed_norm, hidden_norm], NOT [hidden, embed].
This matches the GGUF eh_proj naming convention (embed-hidden).
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn


class MTPHead(nn.Module):
    """
    Qwen3.5 MTP head: predicts the next token from the main model's
    last hidden state + current token embedding.

    Architecture:
      1. RMSNorm hidden + RMSNorm embedding -> concat -> fc projection
      2. One gated-attention transformer layer (same arch as Qwen3.5 attn)
      3. RMSNorm -> shared lm_head -> logits
    """

    def __init__(self, hidden_size=5120, num_heads=24, num_kv_heads=4,
                 head_dim=256, intermediate_size=17408, rms_norm_eps=1e-6,
                 rope_theta=100000.0, partial_rotary_factor=0.25,
                 group_size=64, bits=4):
        super().__init__()
        self.hidden_size = hidden_size
        QL = partial(nn.QuantizedLinear, bias=False, group_size=group_size, bits=bits)

        # Pre-projection norms
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Concat projection: [embed; hidden] -> hidden
        self.fc = QL(hidden_size * 2, hidden_size)

        # Attention layer (gated Q, same as Qwen3NextAttention)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = QL(hidden_size, num_heads * head_dim * 2)
        self.k_proj = QL(hidden_size, num_kv_heads * head_dim)
        self.v_proj = QL(hidden_size, num_kv_heads * head_dim)
        self.o_proj = QL(num_heads * head_dim, hidden_size)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        rope_dims = int(head_dim * partial_rotary_factor)
        self.rope_dims = rope_dims
        self.rope_base = rope_theta

        # MLP
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = QL(hidden_size, intermediate_size)
        self.up_proj = QL(hidden_size, intermediate_size)
        self.down_proj = QL(intermediate_size, hidden_size)

        # Final norm (before shared lm_head)
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, hidden_states, token_embedding, lm_head_fn, cache=None, offset=0):
        """
        Args:
            hidden_states: [B, 1, D] -- last hidden state from main model
            token_embedding: [B, 1, D] -- embedding of the current token
            lm_head_fn: callable that maps [B, 1, D] -> [B, 1, vocab] logits
            cache: KVCache for this MTP attention layer (optional)
            offset: position offset for RoPE
        Returns:
            logits: [B, 1, vocab]
            h: [B, 1, D] -- pre-norm hidden for chaining
        """
        B, S, D = hidden_states.shape

        # 1. Norm + concat + project (embed first, hidden second -- matches GGUF eh_proj)
        h_norm = self.pre_fc_norm_hidden(hidden_states)
        e_norm = self.pre_fc_norm_embedding(token_embedding)
        combined = mx.concatenate([e_norm, h_norm], axis=-1)  # [B, 1, 2D]
        h = self.fc(combined)  # [B, 1, D]

        # 2. Attention layer (gated, with RoPE)
        residual = h
        h_attn = self.input_layernorm(h)

        q_out = self.q_proj(h_attn)
        q_out = q_out.reshape(B, S, self.num_heads, self.head_dim * 2)
        queries, gate = mx.split(q_out, 2, axis=-1)
        gate = gate.reshape(B, S, -1)

        keys = self.k_proj(h_attn).reshape(B, S, self.num_kv_heads, self.head_dim)
        values = self.v_proj(h_attn).reshape(B, S, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # RoPE
        offset_arr = mx.array(offset)
        queries = mx.fast.rope(queries, self.rope_dims, traditional=False,
                               base=self.rope_base, scale=1.0, offset=offset_arr)
        keys = mx.fast.rope(keys, self.rope_dims, traditional=False,
                            base=self.rope_base, scale=1.0, offset=offset_arr)

        # KV cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # SDPA
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        h = residual + self.o_proj(output * mx.sigmoid(gate))

        # 3. MLP
        residual = h
        h_mlp = self.post_attention_layernorm(h)
        h = residual + self.down_proj(nn.silu(self.gate_proj(h_mlp)) * self.up_proj(h_mlp))

        # 4. Final norm -> shared lm_head
        h_normed = self.norm(h)
        logits = lm_head_fn(h_normed)
        return logits, h  # return pre-norm hidden for chaining


def load_mtp(model, weights_path=None):
    """
    Load MTP head weights and attach to the model.

    Args:
        model: The Qwen3.5-27B model (already loaded via mlx_lm.load)
        weights_path: Path to mtp_weights.safetensors

    Returns:
        MTPHead module, ready for inference
    """
    import os

    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "mtp_weights.safetensors")

    print(f"Loading MTP weights from {weights_path}...")
    raw = mx.load(weights_path)

    # Create MTP head
    mtp = MTPHead()

    # Map from our flat names to HF checkpoint names
    # HF stores as model.mtp.layers.0.self_attn.q_proj.weight etc.
    weight_map = {
        "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
        "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
        "fc.weight": "mtp.fc.weight",
        "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
        "q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
        "k_proj.weight": "mtp.layers.0.self_attn.k_proj.weight",
        "v_proj.weight": "mtp.layers.0.self_attn.v_proj.weight",
        "o_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        "q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "gate_proj.weight": "mtp.layers.0.mlp.gate_proj.weight",
        "up_proj.weight": "mtp.layers.0.mlp.up_proj.weight",
        "down_proj.weight": "mtp.layers.0.mlp.down_proj.weight",
        "norm.weight": "mtp.norm.weight",
    }

    # Load into module -- handle quantized weights (weight + scales + biases)
    pairs = []
    for local_name, mtp_key in weight_map.items():
        if mtp_key in raw:
            pairs.append((local_name, raw[mtp_key]))
        # Check for quantized version (scales/biases)
        scales_key = mtp_key.replace(".weight", ".scales")
        biases_key = mtp_key.replace(".weight", ".biases")
        if scales_key in raw:
            pairs.append((local_name.replace(".weight", ".scales"), raw[scales_key]))
        if biases_key in raw:
            pairs.append((local_name.replace(".weight", ".biases"), raw[biases_key]))

    mtp.load_weights(pairs)
    mx.eval(mtp.parameters())

    total_mb = sum(v.nbytes for _, v in pairs) / 1e6
    print(f"MTP head loaded: {total_mb:.1f} MB, {len(pairs)} tensors")

    return mtp
