# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
MOSS-TTS-Realtime model implementation for SGLang.

This model wraps the MOSS-TTS-Realtime architecture (Qwen3 backbone +
local transformer) to work within SGLang's autoregressive inference
infrastructure. The model generates discrete audio tokens (RVQ codes)
which are then decoded to audio waveforms by an external codec.

Architecture:
  - Backbone: Qwen3 language model (28 layers, 2048 hidden)
  - Local Transformer: 4 layers, generates 16 RVQ codes per backbone step
  - Input: Multi-channel (1 text + 16 audio channels per position)
  - Output: Logits over audio code vocabulary (1027 tokens)

The model presents a standard forward() → LogitsProcessorOutput interface
to SGLang by:
  1. Returning first-codebook logits (vocab_size=1027) for SGLang sampling
  2. Internally generating the remaining 15 RVQ codes via local transformer
  3. Managing text token consumption and multi-channel input construction
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Audio token constants
AUDIO_PAD_TOKEN = 1024
AUDIO_BOS_TOKEN = 1025
AUDIO_EOS_TOKEN = 1026
AUDIO_VOCAB_SIZE = 1027
NUM_RVQ_CHANNELS = 16
REFERENCE_AUDIO_PAD_ID = 151654
TEXT_PAD_ID = 151655


class MossTTSLocalTransformer(nn.Module):
    """Local transformer for generating RVQ audio codes.

    A small 4-layer transformer that autoregressively generates 16 RVQ
    codebook tokens conditioned on the backbone's hidden state.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = getattr(config, "rvq", NUM_RVQ_CHANNELS)
        self.audio_vocab_size = getattr(config, "audio_vocab_size", AUDIO_VOCAB_SIZE)
        self.hidden_size = config.hidden_size

        # Per-codebook embedding and head
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(self.audio_vocab_size, self.hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.audio_vocab_size, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )

        # Transformer layers
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        repetition_window: int = 50,
        generated_history: Optional[torch.Tensor] = None,
        gen_step: int = 0,
    ) -> torch.Tensor:
        """Generate all RVQ codes for one backbone step.

        Args:
            backbone_hidden: Hidden state from backbone, shape (batch, 1, hidden).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.
            repetition_penalty: Repetition penalty factor.
            repetition_window: Window size for repetition penalty.
            generated_history: Previously generated tokens for repetition
                penalty, shape (batch, num_steps, num_codebooks).
            gen_step: Current generation step index.

        Returns:
            Generated audio codes, shape (batch, num_codebooks).
        """
        batch_size = backbone_hidden.shape[0]
        device = backbone_hidden.device
        output_tokens = torch.empty(
            batch_size, self.num_codebooks, dtype=torch.long, device=device
        )

        hidden = backbone_hidden  # (batch, 1, hidden_size)

        for i in range(self.num_codebooks):
            # Run through transformer layers
            h = hidden
            for layer in self.layers:
                h = layer(h)[0]
            h = self.norm(h)

            # Get logits for this codebook
            logits = self.lm_heads[i](h[:, -1:, :])  # (batch, 1, vocab)

            # Apply repetition penalty
            if (
                repetition_penalty
                and repetition_penalty != 1.0
                and generated_history is not None
            ):
                logits = self._apply_repetition_penalty(
                    logits,
                    generated_history[:, :gen_step, i],
                    repetition_penalty,
                    repetition_window,
                )

            # Sample token
            token = self._sample(logits, temperature, top_p, top_k)
            output_tokens[:, i] = token.squeeze(-1)

            # Next codebook input: embed the sampled token
            if i < self.num_codebooks - 1:
                token_embed = self.embed_tokens[i + 1](token)  # (batch, 1, hidden)
                hidden = token_embed

        return output_tokens

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample a token from logits."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        logits = logits.squeeze(1)  # (batch, vocab)

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumulative_probs <= (1 - top_p)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
                1, sorted_indices, sorted_mask
            )
            logits = logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        history: torch.Tensor,
        penalty: float,
        window: int,
    ) -> torch.Tensor:
        """Apply repetition penalty over a sliding window."""
        scores = logits.squeeze(1)  # (batch, vocab)
        if window and window > 0:
            history = history[:, -window:]
        cur = scores.gather(1, history)
        new = torch.where(cur < 0, cur * penalty, cur / penalty)
        scores.scatter_(1, history, new)
        return scores.unsqueeze(1)


class MossTTSRealtime(nn.Module):
    """MOSS-TTS-Realtime model adapted for SGLang inference.

    Wraps the backbone (Qwen3) and local transformer into SGLang's
    standard model interface. Returns LogitsProcessorOutput with
    first-codebook logits for SGLang's sampling pipeline.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.num_rvq = getattr(config, "rvq", NUM_RVQ_CHANNELS)
        self.audio_vocab_size = getattr(config, "audio_vocab_size", AUDIO_VOCAB_SIZE)

        # Resolve sub-configs
        lang_config = config.language_config
        if isinstance(lang_config, dict):
            from transformers.models.qwen3 import Qwen3Config

            lang_config = Qwen3Config(**lang_config)

        local_config = config.local_config
        if isinstance(local_config, dict):
            from sglang.srt.configs.moss_tts_realtime import (
                MossTTSRealtimeLocalTransformerConfig,
            )

            local_config = MossTTSRealtimeLocalTransformerConfig(**local_config)

        # Text embedding + audio channel embeddings (17 total)
        self.embed_tokens = nn.ModuleList()
        self.embed_tokens.append(
            nn.Embedding(
                lang_config.vocab_size,
                lang_config.hidden_size,
                getattr(lang_config, "pad_token_id", None),
            )
        )
        for _ in range(self.num_rvq):
            self.embed_tokens.append(
                nn.Embedding(
                    self.audio_vocab_size,
                    lang_config.hidden_size,
                    AUDIO_PAD_TOKEN,
                )
            )

        # Backbone: Qwen3 model (without lm_head)
        from transformers.models.qwen3 import Qwen3Model

        self.language_model = Qwen3Model(lang_config)

        # Local transformer for RVQ code generation
        self.local_transformer = MossTTSLocalTransformer(local_config)

        # Per-request state caches
        self._text_queue: Dict[int, List[int]] = {}
        self._text_cursor: Dict[int, int] = {}
        self._prev_audio_codes: Dict[int, torch.Tensor] = {}
        self._generated_history: Dict[int, List[torch.Tensor]] = {}

    def pad_input_ids(
        self, input_ids: List[int], _mm_inputs=None
    ) -> List[int]:
        """Handle multimodal input padding (no-op for TTS)."""
        return input_ids

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute summed embeddings for multi-channel input.

        Args:
            input_ids: Shape (batch, seq_len, 17) or (batch, seq_len).
        """
        if input_ids.ndim == 2:
            # Standard single-channel (text-only or audio-code-only)
            return self.embed_tokens[0](input_ids)

        # Multi-channel: sum embeddings across channels
        embeds = self.embed_tokens[0](input_ids[..., 0])
        for i in range(1, min(input_ids.shape[-1], len(self.embed_tokens))):
            embeds = embeds + self.embed_tokens[i](input_ids[..., i])
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        """Forward pass for SGLang inference.

        During prefill:
          - Processes multi-channel input through backbone
          - Stores text queue and audio state per request
          - Runs local transformer on final hidden state

        During decode:
          - Constructs multi-channel input from text queue + prev audio codes
          - Runs backbone (KV cached)
          - Runs local transformer for all 16 RVQ codes
          - Returns first-codebook logits

        Returns:
            LogitsProcessorOutput with next_token_logits of shape
            (batch, 1027) over the audio code vocabulary.
        """
        is_decode = forward_batch.forward_mode.is_decode()
        batch_size = forward_batch.batch_size
        device = input_ids.device

        if is_decode:
            return self._forward_decode(input_ids, positions, forward_batch)
        else:
            return self._forward_prefill(input_ids, positions, forward_batch)

    def _forward_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Prefill: process the full prompt and generate first audio codes.

        Reads multi_channel_ids, text_ids, and text_cursor from the
        MultimodalDataItem's model_specific_data (set by MossTTSProcessor).
        """
        device = input_ids.device
        batch_size = forward_batch.batch_size

        # Extract multi-channel IDs and text queue from mm_inputs
        multi_channel_ids = None
        if forward_batch.mm_inputs:
            for idx, mm_input in enumerate(forward_batch.mm_inputs):
                if mm_input is None:
                    continue
                for mm_item in getattr(mm_input, "mm_items", []):
                    msd = getattr(mm_item, "model_specific_data", {})
                    if "multi_channel_ids" in msd:
                        mc = msd["multi_channel_ids"]
                        if not isinstance(mc, torch.Tensor):
                            mc = torch.tensor(mc, dtype=torch.long, device=device)
                        # Add batch dim if needed
                        if mc.ndim == 2:
                            mc = mc.unsqueeze(0)
                        multi_channel_ids = mc

        if multi_channel_ids is not None:
            input_embeds = self.get_input_embeddings(multi_channel_ids)
        else:
            input_embeds = self.embed_tokens[0](input_ids)

        # Run backbone
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state

        # Get last hidden for local transformer
        backbone_h = hidden_states[:, -1:, :]

        # Store per-request state (text queue + cursor from processor)
        if forward_batch.req_pool_indices is not None:
            for idx, req_idx in enumerate(
                forward_batch.req_pool_indices.tolist()
            ):
                self._generated_history[req_idx] = []

                # Extract text_ids and text_cursor from mm_inputs
                if forward_batch.mm_inputs and idx < len(forward_batch.mm_inputs):
                    mm = forward_batch.mm_inputs[idx]
                    if mm is not None:
                        for mm_item in getattr(mm, "mm_items", []):
                            msd = getattr(mm_item, "model_specific_data", {})
                            if "text_ids" in msd:
                                self._text_queue[req_idx] = list(msd["text_ids"])
                                self._text_cursor[req_idx] = int(
                                    msd.get("text_cursor", 0)
                                )
                                break

        # Run local transformer to get first audio codes
        audio_codes = self.local_transformer.forward(
            backbone_h,
            temperature=0.8,
            top_p=0.6,
            top_k=30,
        )

        # Store audio codes per request
        if forward_batch.req_pool_indices is not None:
            for idx, req_idx in enumerate(
                forward_batch.req_pool_indices.tolist()
            ):
                self._prev_audio_codes[req_idx] = audio_codes[idx]
                self._generated_history[req_idx].append(audio_codes[idx])

        # Return first codebook logits (one-hot so SGLang picks the same
        # token the local transformer already sampled) PLUS all 16 RVQ
        # codes in customized_info so they flow to the serving handler.
        first_codes = audio_codes[:, 0]
        logits = torch.full(
            (batch_size, self.audio_vocab_size),
            float("-inf"),
            device=device,
        )
        logits.scatter_(1, first_codes.unsqueeze(1), 100.0)

        # Pack all 16 codes per request into customized_info
        all_codes_list = [
            audio_codes[i].tolist() for i in range(batch_size)
        ]

        return LogitsProcessorOutput(
            next_token_logits=logits,
            customized_info={"audio_rvq_codes": all_codes_list},
        )

    def _forward_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Decode: generate next audio codes given previous step.

        Constructs multi-channel input from text queue + previous audio
        codes, runs backbone + local transformer, and returns all 16 RVQ
        codes via customized_info for faithful codec decoding.
        """
        batch_size = forward_batch.batch_size
        device = input_ids.device

        # Construct multi-channel input: [text_token, prev_16_audio_codes]
        step_ids = torch.full(
            (batch_size, 1, 1 + self.num_rvq),
            AUDIO_PAD_TOKEN,
            dtype=torch.long,
            device=device,
        )

        if forward_batch.req_pool_indices is not None:
            for idx, req_idx in enumerate(
                forward_batch.req_pool_indices.tolist()
            ):
                # Get next text token from the queue
                if req_idx in self._text_queue and req_idx in self._text_cursor:
                    cursor = self._text_cursor[req_idx]
                    text_ids = self._text_queue[req_idx]
                    if cursor < len(text_ids):
                        step_ids[idx, 0, 0] = text_ids[cursor]
                        self._text_cursor[req_idx] = cursor + 1
                    else:
                        step_ids[idx, 0, 0] = TEXT_PAD_ID

                # Fill previous audio codes into channels 1..16
                if req_idx in self._prev_audio_codes:
                    step_ids[idx, 0, 1:] = self._prev_audio_codes[req_idx]

        # Compute summed multi-channel embeddings
        input_embeds = self.get_input_embeddings(step_ids)

        # Run backbone (KV cached)
        outputs = self.language_model(
            inputs_embeds=input_embeds.squeeze(1),
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        backbone_h = hidden_states[:, -1:, :]

        # Build history tensor for repetition penalty
        gen_history = None
        max_history = 0
        if forward_batch.req_pool_indices is not None:
            for req_idx in forward_batch.req_pool_indices.tolist():
                if req_idx in self._generated_history:
                    max_history = max(
                        max_history, len(self._generated_history[req_idx])
                    )
            if max_history > 0:
                gen_history = torch.full(
                    (batch_size, max_history, self.num_rvq),
                    AUDIO_PAD_TOKEN,
                    dtype=torch.long,
                    device=device,
                )
                for idx, req_idx in enumerate(
                    forward_batch.req_pool_indices.tolist()
                ):
                    if req_idx in self._generated_history:
                        hist = self._generated_history[req_idx]
                        for t, codes in enumerate(hist):
                            gen_history[idx, t] = codes

        # Run local transformer — generates all 16 RVQ codes
        audio_codes = self.local_transformer.forward(
            backbone_h,
            temperature=0.8,
            top_p=0.6,
            top_k=30,
            repetition_penalty=1.1,
            repetition_window=50,
            generated_history=gen_history,
            gen_step=max_history,
        )

        # Update per-request state
        if forward_batch.req_pool_indices is not None:
            for idx, req_idx in enumerate(
                forward_batch.req_pool_indices.tolist()
            ):
                self._prev_audio_codes[req_idx] = audio_codes[idx]
                if req_idx not in self._generated_history:
                    self._generated_history[req_idx] = []
                self._generated_history[req_idx].append(audio_codes[idx])

        # Return first codebook logits (one-hot) + all 16 codes in customized_info
        first_codes = audio_codes[:, 0]
        logits = torch.full(
            (batch_size, self.audio_vocab_size),
            float("-inf"),
            device=device,
        )
        logits.scatter_(1, first_codes.unsqueeze(1), 100.0)

        all_codes_list = [
            audio_codes[i].tolist() for i in range(batch_size)
        ]

        return LogitsProcessorOutput(
            next_token_logits=logits,
            customized_info={"audio_rvq_codes": all_codes_list},
        )

    def get_generated_audio_codes(self, req_idx: int) -> Optional[np.ndarray]:
        """Retrieve all generated audio codes for a completed request.

        Returns:
            Audio codes array of shape (num_steps, num_rvq) or None.
        """
        if req_idx in self._generated_history:
            codes = self._generated_history[req_idx]
            if codes:
                return (
                    torch.stack(codes).cpu().numpy()
                )
        return None

    def cleanup_request(self, req_idx: int):
        """Clean up per-request state after completion."""
        self._text_queue.pop(req_idx, None)
        self._text_cursor.pop(req_idx, None)
        self._prev_audio_codes.pop(req_idx, None)
        self._generated_history.pop(req_idx, None)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights from checkpoint.

        Maps MOSS-TTS-Realtime weight names to this model's structure.
        """
        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()

        for name, loaded_weight in weights:
            # Map weight names
            param_name = name

            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)
                loaded.add(param_name)

        unloaded = set(params_dict.keys()) - loaded
        if unloaded:
            logger.warning(
                f"Some weights were not loaded: {len(unloaded)} parameters. "
                f"First few: {list(unloaded)[:5]}"
            )


EntryClass = [MossTTSRealtime]
