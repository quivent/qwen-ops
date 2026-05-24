"""AutoAWQ model implementation for Qwen3.5 (hybrid full-attention + GDN/DeltaNet)."""

import tqdm
from typing import List, Tuple
from typing import TYPE_CHECKING

from .base import BaseAWQForCausalLM

if TYPE_CHECKING:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )


class Qwen3_5AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Qwen3_5DecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual", "mtp", "in_proj_b", "in_proj_a"]

    @staticmethod
    def get_model_layers(model):
        """Return decoder layers, handling both ConditionalGeneration and CausalLM variants.

        ConditionalGeneration: model.model.language_model.layers
        CausalLM:             model.model.layers
        """
        if hasattr(model.model, "language_model"):
            return model.model.language_model.layers
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: "Qwen3_5DecoderLayer"):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        """Move embeddings and rotary_emb to device.

        For ConditionalGeneration models, also moves visual encoder and creates
        an alias at model.model.rotary_emb so that quantizer.py line 162
        (which hardcodes `self.model.model.rotary_emb(...)`) works correctly.
        """
        if hasattr(model.model, "language_model"):
            # ConditionalGeneration variant
            model.model.language_model.embed_tokens = (
                model.model.language_model.embed_tokens.to(device)
            )
            model.model.language_model.rotary_emb = (
                model.model.language_model.rotary_emb.to(device)
            )
            # Alias so quantizer.py's hardcoded path works
            model.model.rotary_emb = model.model.language_model.rotary_emb
            # Move visual encoder
            if hasattr(model.model, "visual"):
                model.model.visual = model.model.visual.to(device)
        else:
            # CausalLM variant
            model.model.embed_tokens = model.model.embed_tokens.to(device)
            model.model.rotary_emb = model.model.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module: "Qwen3_5DecoderLayer", input_feat, module_kwargs
    ):
        """Return scaling layers, branching on full_attention vs linear_attention.

        full_attention layers have `self_attn` with q_proj (2x width for gating),
        k_proj, v_proj, o_proj.

        linear_attention (GDN) layers have `linear_attn` with in_proj_qkv,
        in_proj_z, in_proj_b, in_proj_a, out_proj.
        """
        layers = []

        if hasattr(module, "self_attn"):
            # === Full attention layer ===

            # Attention input: layernorm -> q/k/v projections
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )

            # Attention output: v_proj -> o_proj
            # Only when shapes match (q_proj is 2x width due to gating, so we
            # check v_proj vs o_proj specifically)
            if (
                module.self_attn.v_proj.weight.shape
                == module.self_attn.o_proj.weight.shape
            ):
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

            # MLP gate/up
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )

            # MLP down
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

        elif hasattr(module, "linear_attn"):
            # === Linear attention (GDN/DeltaNet) layer ===

            # Input layernorm -> linear_attn input projections
            # Note: in_proj_b and in_proj_a are excluded from quantization
            # (48 out_features not divisible by pack_num=8), so exclude from scaling too
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.linear_attn.in_proj_qkv,
                        module.linear_attn.in_proj_z,
                    ],
                    inp=input_feat["linear_attn.in_proj_qkv"],
                    module2inspect=module.linear_attn,
                    kwargs=module_kwargs,
                )
            )

            # Skip out_proj scaling: no clean prev_op (output comes after
            # conv1d + gated delta rule, not a simple linear chain)

            # MLP gate/up
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )

            # MLP down
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

        return layers

    @staticmethod
    def fuse_layers(model):
        """Fuse QKV for full_attention layers. Skip linear_attention layers
        (GDN has non-standard conv1d + gated delta rule structure that cannot
        be fused with standard QKV fusion)."""
        fuser = Qwen3_5Fuser(model)
        fuser.fuse_transformer()


class Qwen3_5Fuser:
    def __init__(self, model):
        self.model = model

        # Determine if this is a ConditionalGeneration or CausalLM model
        if hasattr(model.model, "language_model"):
            self.text_model = model.model.language_model
            self.config = model.config.text_config
        else:
            self.text_model = model.model
            self.config = model.config

    def fuse_transformer(self):
        """Fuse QKV projections and norms for full_attention layers only.

        Linear attention layers are left unchanged since their structure
        (conv1d + gated delta rule) is incompatible with standard QKV fusion.
        """
        from awq.utils.fused_utils import fuse_qkv
        from awq.modules.fused.block import QwenBlock
        from awq.modules.fused.model import LlamaLikeModel
        from awq.modules.fused.norm import FasterTransformerRMSNorm

        blocks = []

        for i, module in enumerate(
            tqdm.tqdm(self.text_model.layers, desc="Fusing layers...")
        ):
            device = next(iter(module.state_dict().values())).device

            if hasattr(module, "self_attn"):
                # Full attention layer: fuse QKV
                qkv = fuse_qkv(
                    module,
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                )
                norm_1 = FasterTransformerRMSNorm(
                    module.input_layernorm.weight,
                    module.input_layernorm.eps,
                )
                norm_2 = FasterTransformerRMSNorm(
                    module.post_attention_layernorm.weight,
                    module.post_attention_layernorm.eps,
                )
                blocks.append(
                    QwenBlock(
                        hidden_size=self.config.hidden_size,
                        n_heads=self.config.num_attention_heads,
                        n_kv_heads=self.config.num_key_value_heads,
                        qkv_layer=qkv,
                        o_proj=module.self_attn.o_proj,
                        mlp=module.mlp,
                        norm_1=norm_1,
                        norm_2=norm_2,
                        dev=device,
                        max_seq_len=self.config.max_position_embeddings,
                        rope_theta=(self.config.rope_parameters or {}).get(
                            "rope_theta", getattr(self.config, "rope_theta", 10000.0)
                        ) if self.config.rope_parameters else getattr(self.config, "rope_theta", 10000.0),
                        q_norm=module.self_attn.q_norm,
                        k_norm=module.self_attn.k_norm,
                        head_dim=getattr(
                            self.config,
                            "head_dim",
                            self.config.hidden_size
                            // self.config.num_attention_heads,
                        ),
                    )
                )
            else:
                # Linear attention layer: cannot fuse, keep as-is
                blocks.append(module)

        # Replace the correct attribute on the original model
        fused_model = LlamaLikeModel(
            self.config.vocab_size,
            blocks,
            self.text_model.embed_tokens,
            self.text_model.norm,
        )
        if hasattr(self.model.model, "language_model"):
            self.model.model.language_model = fused_model
        else:
            self.model.model = fused_model
        setattr(fused_model, "blocks", fused_model.blocks)
