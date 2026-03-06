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


# =====================================================================
# Local transformer for RVQ codebook generation
# (inference-only reimplementation of MossTTSRealtimeLocalTransformerForCausalLM)
# =====================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # (batch, 1, seq, dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return hidden_states.reshape(b, h * n_rep, s, d)


class _LocalRMSNorm(nn.Module):
    """RMSNorm (matches MossTTSRealtimeLocalTransformerRMSNorm weights)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class _LocalMLP(nn.Module):
    """MLP (matches MossTTSRealtimeLocalTransformerMLP weights)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LocalRotaryEmbedding(nn.Module):
    """Rotary embedding (matches MossTTSRealtimeLocalTransformerRotaryEmbedding)."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, rope_theta: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _LocalAttention(nn.Module):
    """Attention with QK-norm (matches MossTTSRealtimeLocalTransformerAttention)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=bias
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = _LocalRMSNorm(self.head_dim, eps=eps)
        self.k_norm = _LocalRMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        step_idx: int,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz = hidden_states.shape[0]
        hidden_shape = (bsz, 1, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        v = self.v_proj(hidden_states).view(hidden_shape)

        q = q.transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        k = k.transpose(1, 2)  # (batch, kv_heads, 1, head_dim)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Write to pre-allocated KV cache
        key_cache, value_cache = kv_cache
        key_cache[:, :, step_idx : step_idx + 1] = k
        value_cache[:, :, step_idx : step_idx + 1] = v

        # GQA: expand KV heads to match query heads
        k_full = _repeat_kv(key_cache, self.num_key_value_groups)
        v_full = _repeat_kv(value_cache, self.num_key_value_groups)

        # Scaled dot-product attention with causal mask
        attn_weights = (
            torch.matmul(q, k_full.transpose(2, 3)) * self.scaling
        )
        attn_weights = (
            attn_weights + causal_mask[:, :, step_idx : step_idx + 1, :]
        )
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v_full)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, 1, -1)
        return self.o_proj(attn_output)


class _LocalDecoderLayer(nn.Module):
    """Decoder layer (matches MossTTSRealtimeLocalTransformerDecoderLayer)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = _LocalAttention(config, layer_idx)
        self.mlp = _LocalMLP(config.hidden_size, config.intermediate_size)
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = _LocalRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = _LocalRMSNorm(
            config.hidden_size, eps=eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        step_idx: int,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, kv_cache, step_idx, causal_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _LocalTransformerModel(nn.Module):
    """Base model (matches MossTTSRealtimeLocalTransformer).

    Weight prefix: ``local_transformer.model.*``
    """

    def __init__(self, config):
        super().__init__()
        self.num_codebooks = getattr(config, "rvq", NUM_RVQ_CHANNELS)
        self.hidden_size = config.hidden_size
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        audio_vocab = getattr(config, "audio_vocab_size", AUDIO_VOCAB_SIZE)
        audio_pad = getattr(config, "audio_pad_token", AUDIO_PAD_TOKEN)
        # rvq - 1 = 15 embeddings: codebooks 1..15 (codebook 0 uses backbone hidden)
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(audio_vocab, self.hidden_size, audio_pad)
                for _ in range(self.num_codebooks - 1)
            ]
        )

        self.layers = nn.ModuleList(
            [
                _LocalDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm = _LocalRMSNorm(self.hidden_size, eps=eps)

        rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.rotary_emb = _LocalRotaryEmbedding(self.head_dim, rope_theta)


class MossTTSLocalTransformer(nn.Module):
    """CausalLM wrapper (matches MossTTSRealtimeLocalTransformerForCausalLM).

    Wraps ``_LocalTransformerModel`` and adds per-codebook LM heads.
    Autoregressively generates all 16 RVQ codes for a single backbone step
    using KV-cached attention across codebook positions.

    Weight prefix: ``local_transformer.*``
      - ``local_transformer.model.*`` → ``_LocalTransformerModel``
      - ``local_transformer.local_lm_heads.*`` → per-codebook heads
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_codebooks = getattr(config, "rvq", NUM_RVQ_CHANNELS)
        self.audio_vocab_size = getattr(
            config, "audio_vocab_size", AUDIO_VOCAB_SIZE
        )
        self.hidden_size = config.hidden_size

        self.model = _LocalTransformerModel(config)
        self.local_lm_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.audio_vocab_size, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )

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
        """Autoregressively generate all RVQ codes for one backbone step.

        Args:
            backbone_hidden: (batch, 1, hidden_size) from backbone.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.
            repetition_penalty: Repetition penalty factor.
            repetition_window: Window for repetition penalty.
            generated_history: Previous codes for rep. penalty,
                shape (batch, num_steps, num_codebooks).
            gen_step: Current generation step index.

        Returns:
            Audio codes tensor of shape (batch, num_codebooks).
        """
        batch_size = backbone_hidden.shape[0]
        device = backbone_hidden.device
        dtype = backbone_hidden.dtype

        output_tokens = torch.empty(
            batch_size, self.num_codebooks, dtype=torch.long, device=device
        )

        # Pre-allocate KV caches for all layers
        num_layers = len(self.model.layers)
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(
                batch_size,
                self.model.num_kv_heads,
                self.num_codebooks,
                self.model.head_dim,
                device=device,
                dtype=dtype,
            )
            v = torch.zeros_like(k)
            kv_caches.append((k, v))

        # Pre-compute causal mask: (1, 1, max_seq, max_seq)
        causal_mask = (
            torch.triu(
                torch.full(
                    (self.num_codebooks, self.num_codebooks),
                    float("-inf"),
                    device=device,
                    dtype=dtype,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Pre-compute all position embeddings: (1, num_codebooks, head_dim)
        all_positions = torch.arange(
            self.num_codebooks, device=device, dtype=torch.long
        ).unsqueeze(0)
        all_cos, all_sin = self.model.rotary_emb(
            backbone_hidden, all_positions
        )

        for i in range(self.num_codebooks):
            # Input: backbone hidden for codebook 0, else embed prev code
            if i == 0:
                hidden = backbone_hidden  # (batch, 1, hidden)
            else:
                hidden = self.model.embed_tokens[i - 1](
                    output_tokens[:, i - 1 : i]
                )

            # Position embeddings for this step
            cos_i = all_cos[:, i : i + 1]
            sin_i = all_sin[:, i : i + 1]

            # Run through decoder layers with KV cache
            h = hidden
            for layer_idx, layer in enumerate(self.model.layers):
                h = layer(
                    h, (cos_i, sin_i), kv_caches[layer_idx], i, causal_mask
                )
            h = self.model.norm(h)

            # Project to audio vocabulary
            logits = self.local_lm_heads[i](h[:, -1:, :])

            # Apply repetition penalty
            if (
                repetition_penalty
                and repetition_penalty != 1.0
                and generated_history is not None
                and gen_step > 0
            ):
                logits = self._apply_repetition_penalty(
                    logits,
                    generated_history[:, :gen_step, i],
                    repetition_penalty,
                    repetition_window,
                )

            # Sample
            token = self._sample(logits, temperature, top_p, top_k)
            output_tokens[:, i] = token.squeeze(-1)

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

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[
                ..., -1, None
            ]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=False
            )
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
        scores = logits.squeeze(1)
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

        # Backbone: SGLang's Qwen3 model (paged attention + KV cache)
        from sglang.srt.models.qwen3 import Qwen3Model as SglangQwen3Model

        self.language_model = SglangQwen3Model(
            lang_config, quant_config=quant_config, prefix="language_model"
        )
        self.hidden_size = lang_config.hidden_size

        # Local transformer for RVQ code generation
        self.local_transformer = MossTTSLocalTransformer(local_config)

        # Per-request state caches
        self._text_queue: Dict[int, List[int]] = {}
        self._text_cursor: Dict[int, int] = {}
        self._prev_audio_codes: Dict[int, torch.Tensor] = {}
        self._generated_history: Dict[int, List[torch.Tensor]] = {}

        # Pre-allocated buffers for CUDA graph compatibility.
        # During graph capture/replay the model cannot call .item() or use
        # Python dicts, so input embeddings and output codes go through
        # fixed-address GPU tensors that the graph can reference.
        _GRAPH_MAX_BS = 64
        self.register_buffer(
            "_graph_input_embeds",
            torch.zeros(_GRAPH_MAX_BS, lang_config.hidden_size),
            persistent=False,
        )
        self.register_buffer(
            "_audio_codes_buffer",
            torch.zeros(_GRAPH_MAX_BS, self.num_rvq, dtype=torch.long),
            persistent=False,
        )

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

        Uses SGLang's flat token representation: input_ids is (total_tokens,)
        across all requests in the batch. Multi-channel audio embeddings are
        scattered per-request using extend_start_loc offsets.
        """
        device = input_ids.device
        batch_size = forward_batch.batch_size

        # Text embeddings for all tokens (flat representation)
        input_embeds = self.embed_tokens[0](input_ids)  # (total_tokens, hidden)

        # Per-request token ranges for embedding scatter
        start_loc = forward_batch.extend_start_loc.cpu().numpy()
        seq_lens = forward_batch.extend_seq_lens.cpu().numpy()

        # Add audio channel embeddings and extract per-request state
        for idx in range(batch_size):
            req_idx = forward_batch.req_pool_indices[idx].item()
            self._generated_history[req_idx] = []

            mm = None
            if forward_batch.mm_inputs and idx < len(forward_batch.mm_inputs):
                mm = forward_batch.mm_inputs[idx]

            if mm is not None:
                for mm_item in getattr(mm, "mm_items", []):
                    msd = getattr(mm_item, "model_specific_data", {})

                    # Scatter audio channel embeddings (channels 1..16)
                    if "multi_channel_ids" in msd:
                        mc = msd["multi_channel_ids"]
                        if not isinstance(mc, torch.Tensor):
                            mc = torch.tensor(
                                mc, dtype=torch.long, device=device
                            )
                        start = int(start_loc[idx])
                        length = int(seq_lens[idx])
                        mc = mc[:length]
                        for ch in range(
                            1, min(mc.shape[-1], len(self.embed_tokens))
                        ):
                            input_embeds[start : start + length] += (
                                self.embed_tokens[ch](mc[:, ch])
                            )

                    # Store text queue and cursor for decode phase
                    if "text_ids" in msd:
                        self._text_queue[req_idx] = list(msd["text_ids"])
                        self._text_cursor[req_idx] = int(
                            msd.get("text_cursor", 0)
                        )
                    break

        # Run backbone with SGLang's paged attention
        hidden_states = self.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        # hidden_states: (total_tokens, hidden_size)

        # Extract last hidden state per request
        last_positions = [
            int(start_loc[idx] + seq_lens[idx] - 1)
            for idx in range(batch_size)
        ]
        backbone_h = hidden_states[last_positions].unsqueeze(1)  # (batch, 1, hidden)

        # Run local transformer to get first audio codes
        audio_codes = self.local_transformer.forward(
            backbone_h,
            temperature=0.8,
            top_p=0.6,
            top_k=30,
        )

        # Store audio codes per request
        for idx in range(batch_size):
            req_idx = forward_batch.req_pool_indices[idx].item()
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

        CUDA-graph-compatible: during capture, reads pre-computed embeddings
        from ``_graph_input_embeds`` and writes output codes to
        ``_audio_codes_buffer``.  In eager mode it computes everything inline.
        """
        batch_size = forward_batch.batch_size
        device = input_ids.device
        is_capturing = (
            device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        )

        if is_capturing:
            # Graph capture: use pre-allocated buffer (filled during capture
            # warmup or by cuda_graph_prepare before replay).
            input_embeds = self._graph_input_embeds[:batch_size]
        else:
            # Eager mode: compute multi-channel embeddings inline.
            input_embeds = self._compute_decode_embeds(forward_batch)

        # Run backbone with KV cache
        hidden_states = self.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        backbone_h = hidden_states.unsqueeze(1)  # (batch, 1, hidden)

        # Build repetition-penalty history (skipped during graph capture)
        gen_history = None
        max_history = 0
        if not is_capturing:
            gen_history, max_history = self._build_rep_history(
                forward_batch, device
            )

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

        # Write codes to pre-allocated buffer (graph-safe write)
        self._audio_codes_buffer[:batch_size] = audio_codes

        # Build one-hot logits for SGLang sampling
        first_codes = audio_codes[:, 0]
        logits = torch.full(
            (batch_size, self.audio_vocab_size),
            float("-inf"),
            device=device,
        )
        logits.scatter_(1, first_codes.unsqueeze(1), 100.0)

        if not is_capturing:
            # Eager mode: update state + build customized_info now.
            self._update_decode_state(forward_batch, audio_codes)
            codes_list = [
                audio_codes[i].tolist() for i in range(batch_size)
            ]
            return LogitsProcessorOutput(
                next_token_logits=logits,
                customized_info={"audio_rvq_codes": codes_list},
            )

        # During capture: return logits only; customized_info will be
        # populated by cuda_graph_post_replay() after each graph replay.
        return LogitsProcessorOutput(next_token_logits=logits)

    # ------------------------------------------------------------------
    # Helpers for decode embedding / state (called in eager mode only)
    # ------------------------------------------------------------------

    def _compute_decode_embeds(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Build multi-channel embeddings for decode (text + 16 audio)."""
        batch_size = forward_batch.batch_size
        device = self._graph_input_embeds.device
        dtype = self.embed_tokens[0].weight.dtype
        input_embeds = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        if forward_batch.req_pool_indices is None:
            return input_embeds

        for idx in range(batch_size):
            req_idx = forward_batch.req_pool_indices[idx].item()

            # Text channel (channel 0)
            text_token = TEXT_PAD_ID
            if req_idx in self._text_queue and req_idx in self._text_cursor:
                cursor = self._text_cursor[req_idx]
                text_ids = self._text_queue[req_idx]
                if cursor < len(text_ids):
                    text_token = text_ids[cursor]
                    self._text_cursor[req_idx] = cursor + 1

            text_id = torch.tensor(
                [text_token], device=device, dtype=torch.long
            )
            embed = self.embed_tokens[0](text_id)

            # Audio channels (1..16)
            if req_idx in self._prev_audio_codes:
                prev_codes = self._prev_audio_codes[req_idx]
                for ch in range(self.num_rvq):
                    embed = embed + self.embed_tokens[ch + 1](
                        prev_codes[ch : ch + 1]
                    )
            else:
                pad_id = torch.tensor(
                    [AUDIO_PAD_TOKEN], device=device, dtype=torch.long
                )
                for ch in range(self.num_rvq):
                    embed = embed + self.embed_tokens[ch + 1](pad_id)

            input_embeds[idx] = embed.squeeze(0)
        return input_embeds

    def _build_rep_history(
        self, forward_batch: ForwardBatch, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Build repetition-penalty history tensor."""
        max_history = 0
        if forward_batch.req_pool_indices is None:
            return None, 0
        for req_idx in forward_batch.req_pool_indices.tolist():
            if req_idx in self._generated_history:
                max_history = max(
                    max_history, len(self._generated_history[req_idx])
                )
        if max_history == 0:
            return None, 0

        batch_size = forward_batch.batch_size
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
                for t, codes in enumerate(self._generated_history[req_idx]):
                    gen_history[idx, t] = codes
        return gen_history, max_history

    def _update_decode_state(
        self, forward_batch: ForwardBatch, audio_codes: torch.Tensor
    ):
        """Update per-request state dicts after decode."""
        if forward_batch.req_pool_indices is None:
            return
        for idx, req_idx in enumerate(
            forward_batch.req_pool_indices.tolist()
        ):
            self._prev_audio_codes[req_idx] = audio_codes[idx]
            if req_idx not in self._generated_history:
                self._generated_history[req_idx] = []
            self._generated_history[req_idx].append(audio_codes[idx])

    # ------------------------------------------------------------------
    # CUDA graph hooks (called by CudaGraphRunner before/after replay)
    # ------------------------------------------------------------------

    def cuda_graph_prepare(self, forward_batch: ForwardBatch):
        """Pre-compute decode embeddings before CUDA graph replay.

        Called by ``CudaGraphRunner.replay()`` before graph replay.
        Fills ``_graph_input_embeds`` so the captured graph reads fresh
        embeddings from the same memory address.
        """
        embeds = self._compute_decode_embeds(forward_batch)
        bs = embeds.shape[0]
        self._graph_input_embeds[:bs] = embeds

    def cuda_graph_post_replay(
        self,
        output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> LogitsProcessorOutput:
        """Update state and populate customized_info after graph replay.

        Called by ``CudaGraphRunner.replay()`` after the graph has run.
        Reads generated codes from ``_audio_codes_buffer`` which was
        written by the replayed graph.
        """
        batch_size = forward_batch.batch_size
        audio_codes = self._audio_codes_buffer[:batch_size]
        self._update_decode_state(forward_batch, audio_codes)

        codes_list = [
            audio_codes[i].tolist() for i in range(batch_size)
        ]
        return LogitsProcessorOutput(
            next_token_logits=output.next_token_logits,
            full_logits=output.full_logits if hasattr(output, "full_logits") else None,
            hidden_states=output.hidden_states if hasattr(output, "hidden_states") else None,
            customized_info={"audio_rvq_codes": codes_list},
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
        Handles stacked Q/K/V → qkv_proj and gate/up → gate_up_proj
        for the SGLang backbone.
        """
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()

        for name, loaded_weight in weights:
            # Skip non-parameter cached tensors
            if "rotary_emb.inv_freq" in name:
                continue

            # Try stacked params mapping (only for backbone layers)
            is_stacked = False
            if "language_model." in name:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    stacked_name = name.replace(weight_name, param_name)
                    if stacked_name in params_dict:
                        param = params_dict[stacked_name]
                        weight_loader = getattr(
                            param, "weight_loader", None
                        )
                        if weight_loader:
                            weight_loader(param, loaded_weight, shard_id)
                            loaded.add(stacked_name)
                            is_stacked = True
                    break

            if is_stacked:
                continue

            # Direct parameter match
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded.add(name)

        unloaded = set(params_dict.keys()) - loaded
        if unloaded:
            logger.warning(
                f"Some weights were not loaded: {len(unloaded)} parameters. "
                f"First few: {list(unloaded)[:10]}"
            )


EntryClass = [MossTTSRealtime]
