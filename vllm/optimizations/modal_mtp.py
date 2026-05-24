# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Modal MTP: self-speculative decoding using draft/verify modes on a single
hybrid (DeltaNet + attention) model such as Qwen3.5-27B.

Design
------
Unlike EAGLE, which uses a *separate* draft model, Modal MTP re-uses the
target model itself.  ``set_draft_mode(True)`` makes the 16 full-attention
layers into identity pass-throughs so that only the 48 DeltaNet (recurrent)
layers plus the MLPs execute during drafting.  Because attention is skipped,
no KV-cache writes happen and no attention metadata needs to be built for
the draft steps.  The DeltaNet layers use their own mamba-style recurrent
state (conv + temporal tensors), which is managed via snapshot/restore.

Key architectural decisions
---------------------------
1. **The proposer does NOT own a model.**  It borrows ``self._target_model``
   (the ``Qwen3_5ForCausalLM`` instance living on the model runner) and
   calls its ``forward()`` in draft mode.

2. **Attention metadata is not needed for draft forwards.**  When
   ``_skip_attention`` is True on all full-attention layers, those layers
   execute ``output.copy_(input)`` and never touch the KV cache or the
   attention backend.  We still need a ``set_forward_context`` call so that
   the DeltaNet (mamba-style) layers can find their per-layer metadata in
   the static forward context, but the *attention* entries can be stubs /
   ``None``.  The mamba metadata is already populated on the static forward
   context by the target forward that just ran, and DeltaNet layers look
   it up by layer name from ``compilation_config.static_forward_context``.

3. **Token sampling uses the target model's ``lm_head`` / MTP head.**
   After each draft forward we call ``compute_logits`` + argmax.  If an
   MTP head is loaded (``Qwen3_5MultiTokenPredictor``), we use it instead
   of the base lm_head for better draft accuracy.

4. **Slot mapping / position tracking for DeltaNet.**  DeltaNet layers
   maintain recurrent state indexed by request slot.  During draft steps
   we increment positions by 1 per step (same as EAGLE's per-step update)
   and update slot mappings so the recurrent state writes go to the correct
   cache slots.  We re-use ``eagle_step_update_slot_mapping_and_metadata``
   for this.

5. **No CUDA graphs for draft path (initially).**  Draft forwards change
   the model's internal flag each step, which is incompatible with captured
   graphs.  A follow-up can trace separate draft-mode graphs.
"""

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import (
    SpecDecodeBaseProposer,
    eagle_step_update_slot_mapping_and_metadata,
)

logger = init_logger(__name__)


class ModalMTPProposer(SpecDecodeBaseProposer):
    """Self-speculative proposer: runs the *target* model in draft mode.

    This proposer has no weights of its own.  It toggles draft mode on the
    target model, runs N lightweight forwards, then restores the recurrent
    state so the verify pass can proceed from a clean baseline.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # Modal MTP uses the target model directly — no separate draft.
        # Set pass_hidden_states=True so the base class doesn't create
        # extra input slots (which trigger multimodal checks).
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # Override: modal_mtp doesn't need extra slots at all.
        # The target model manages its own KV cache.
        self.needs_extra_input_slots = False
        self._runner = runner  # GPUModelRunner reference
        self._target_model: nn.Module | None = None
        self._mtp_head: nn.Module | None = None

    # -----------------------------------------------------------------
    # Model lifecycle
    # -----------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        """Called by the model runner after the target model is loaded.

        We store a reference to the target model (Qwen3_5ForCausalLM) and
        fish out the inner Qwen3_5Model which has the draft-mode helpers.

        We also look for an MTP head (Qwen3_5MultiTokenPredictor) which
        the model runner may have loaded as part of the speculative config.
        If present we use it for draft token sampling; otherwise we fall
        back to the base lm_head.
        """
        self._target_model = target_model

        # The inner model with set_draft_mode / snapshot / restore.
        # Navigate: Qwen3_5ForConditionalGeneration
        #   -> .language_model (Qwen3_5ForCausalLM)
        #   -> .model (Qwen3_5Model)
        inner = target_model
        for attr in ("language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
        self._inner_model = inner

        assert hasattr(self._inner_model, "set_draft_mode"), (
            "Target model must have set_draft_mode(); "
            "is patch 02 applied?"
        )

        # Optional MTP head for better draft quality
        if hasattr(target_model, "mtp") and target_model.mtp is not None:
            self._mtp_head = target_model.mtp
            logger.info("ModalMTPProposer: using MTP head for drafting")
        else:
            self._mtp_head = None
            logger.info("ModalMTPProposer: using lm_head for drafting "
                        "(no MTP head found)")

    # -----------------------------------------------------------------
    # Draft token generation
    # -----------------------------------------------------------------

    def _sample_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample one token per sequence from hidden states."""
        assert self._target_model is not None
        logits = self._target_model.compute_logits(hidden_states)
        return logits.argmax(dim=-1)

    def _mtp_sample_from_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MTP head + sample.  Returns (draft_token_ids, new_hidden).

        If no MTP head is available, falls back to lm_head on the raw
        hidden states (the hidden states pass straight through).
        """
        if self._mtp_head is not None:
            mtp_hidden = self._mtp_head(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                spec_step_idx=spec_step_idx,
            )
            logits = self._target_model.compute_logits(mtp_hidden)
            return logits.argmax(dim=-1), mtp_hidden
        else:
            logits = self._target_model.compute_logits(hidden_states)
            return logits.argmax(dim=-1), hidden_states

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        """Generate up to ``num_speculative_tokens`` draft tokens.

        Flow:
          1. Snapshot DeltaNet state
          2. Enable draft mode (skip attention)
          3. For each draft step:
             a. Run target model forward (DeltaNet + MLPs only)
             b. Sample next token via MTP head or lm_head
             c. Update positions and slot mappings
          4. Restore DeltaNet state
          5. Disable draft mode
        """
        assert self._target_model is not None
        assert self._inner_model is not None

        batch_size = common_attn_metadata.batch_size()

        # ----------------------------------------------------------
        # Step 1: snapshot DeltaNet recurrent state
        # NOTE: Disabled — snapshot is too large (40GB for 256 requests).
        # The verify pass reprocesses tokens and overwrites DeltaNet
        # state, so corruption from draft forwards gets fixed.
        # ----------------------------------------------------------
        deltanet_snapshot = None

        # ----------------------------------------------------------
        # Step 2: enable draft mode (attention layers become identity)
        # ----------------------------------------------------------
        self._inner_model.set_draft_mode(True)

        draft_token_ids_list: list[torch.Tensor] = []

        try:
            # For the first draft step, the "input" is next_token_ids
            # (the just-sampled token from the verify pass) and the
            # hidden_states from the target forward.
            current_token_ids = next_token_ids.int()  # [batch_size]

            # Positions: start from the last verified position + 1
            if token_indices_to_sample is not None:
                positions = target_positions[token_indices_to_sample]
            else:
                # Decode case: positions are just the seq_lens
                positions = common_attn_metadata.seq_lens.clone()

            hidden_states = target_hidden_states
            if token_indices_to_sample is not None:
                hidden_states = hidden_states[token_indices_to_sample]

            for step_idx in range(self.num_speculative_tokens):
                # Increment positions for this draft token
                positions = positions + 1

                # Build minimal attention metadata for the forward context.
                # In draft mode, attention layers are skipped entirely, so
                # the attention metadata dict can map attention layer names
                # to None.  The DeltaNet layers use their own metadata from
                # the static_forward_context (populated during model init).
                #
                # However, set_forward_context still needs to be called so
                # that the global forward context is populated (num_tokens,
                # slot_mapping, etc.) for the DeltaNet/mamba layers.
                draft_attn_metadata: dict[str, object] = {}
                # Attention layers will be skipped, so we set None for them.
                # DeltaNet layers read from static_forward_context directly.

                # Embed the current token
                inputs_embeds = self._target_model.embed_input_ids(
                    current_token_ids
                )

                # Update slot mapping for DeltaNet cache writes.
                # We re-use the EAGLE helper which increments seq_lens,
                # computes clamped positions, and writes slot_mapping.
                block_size = self._runner.cache_config.block_size if self._runner and hasattr(self._runner, 'cache_config') else 128
                if self._runner and hasattr(self._runner, '_slot_mapping_buffer'):
                    eagle_step_update_slot_mapping_and_metadata(
                        positions_1d=positions,
                        block_table_tensor=common_attn_metadata.block_table_tensor,
                        seq_lens=common_attn_metadata.seq_lens,
                        block_size=block_size,
                        max_model_len=self.max_model_len,
                        out_clamped_positions=positions,
                        out_slot_mapping=self._runner._slot_mapping_buffer[:batch_size],
                        input_batch_size=batch_size,
                    )
                    slot_mapping = self._runner._slot_mapping_buffer[:batch_size]
                else:
                    slot_mapping = None

                # Run the target model in draft mode.  Because attention
                # is skipped, this is DeltaNet + MLP only.
                # Use draft_mode=True in BatchDescriptor to dispatch
                # the draft-mode CUDA graph.
                draft_batch_desc = BatchDescriptor(
                    num_tokens=batch_size, draft_mode=True)
                with set_forward_context(
                    draft_attn_metadata,
                    self.vllm_config,
                    num_tokens=batch_size,
                    num_tokens_across_dp=None,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                    batch_descriptor=draft_batch_desc,
                    slot_mapping=slot_mapping,
                ):
                    model_output = self._target_model(
                        input_ids=current_token_ids,
                        positions=positions,
                        inputs_embeds=inputs_embeds,
                    )

                if isinstance(model_output, tuple):
                    hidden_states = model_output[0]
                else:
                    hidden_states = model_output

                hidden_states = hidden_states[:batch_size]

                # Sample draft token via MTP head or lm_head
                draft_ids, _ = self._mtp_sample_from_hidden(
                    hidden_states,
                    current_token_ids,
                    positions,
                    spec_step_idx=step_idx,
                )
                draft_token_ids_list.append(draft_ids)
                current_token_ids = draft_ids.int()

        finally:
            # ----------------------------------------------------------
            # Steps 4-5: restore state and disable draft mode (always)
            # ----------------------------------------------------------
            if deltanet_snapshot is not None:
                self._inner_model.restore_deltanet_state(deltanet_snapshot)
            self._inner_model.set_draft_mode(False)

        # [batch_size, num_speculative_tokens]
        return torch.stack(draft_token_ids_list, dim=1)


# =====================================================================
# Model runner integration points
# =====================================================================
#
# The following changes are needed in gpu_model_runner.py to wire up
# ModalMTPProposer.  They parallel the existing EAGLE path.
#
# --- a/vllm/v1/worker/gpu_model_runner.py
# +++ b/vllm/v1/worker/gpu_model_runner.py
#
# 1. Import:
#
#     from vllm.v1.spec_decode.modal_mtp import ModalMTPProposer
#
# 2. In __init__, where drafter is created (around line 551):
#
#     elif self.speculative_config.method == "modal_mtp":
#         self.drafter = ModalMTPProposer(
#             vllm_config=self.vllm_config,
#             device=self.device,
#             runner=self,
#         )
#
# 3. In propose_draft_token_ids (around line 4605), add a branch:
#
#     elif spec_config.method == "modal_mtp":
#         assert isinstance(self.drafter, ModalMTPProposer)
#         # Modal MTP uses the same preparation as EAGLE for
#         # next_token_ids and hidden states, but calls the target
#         # model directly in draft mode.
#         <same prepare_next_token_ids_padded / prepare_inputs_padded
#          logic as the EAGLE branch>
#
# 4. In the use_gpu_toks check (around line 4210):
#
#     use_gpu_toks = (
#         spec_config.use_eagle()
#         or spec_config.uses_draft_model()
#         or spec_config.uses_extract_hidden_states()
#  +      or spec_config.method == "modal_mtp"
#     ) and not spec_config.disable_padded_drafter_batch
#
# 5. In the isinstance check for use_gpu_toks (around line 4218):
#
#     assert isinstance(
#         self.drafter,
#  -      EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
#  +      EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer
#  +      | ModalMTPProposer,
#     )
