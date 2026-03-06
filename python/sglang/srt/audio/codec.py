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
Audio codec wrapper for MOSS-Audio-Tokenizer and compatible discrete audio codecs.

Follows the ComponentLoader pattern from multimodal_gen/runtime/loader/.
Analogous to MOVA's DAC audio VAE (multimodal_gen/runtime/models/vaes/dac.py)
but for discrete audio token encoding/decoding rather than continuous latents.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioCodec:
    """Manages a discrete audio codec model for encoding/decoding audio.

    This wrapper handles the MOSS-Audio-Tokenizer (or compatible codecs)
    for two operations:
    - encode: audio waveform → discrete codec tokens (for reference audio in TTS)
    - decode: discrete codec tokens → audio waveform (for TTS output)
    """

    def __init__(self, model: torch.nn.Module, sample_rate: int = 24000):
        self.model = model
        self.sample_rate = sample_rate

    @staticmethod
    def load(
        codec_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "AudioCodec":
        """Load a codec model from a HuggingFace model path.

        Follows the MOVA ComponentLoader pattern
        (multimodal_gen/runtime/loader/component_loaders/vae_loader.py).

        Args:
            codec_path: HuggingFace model path or local directory.
            device: Device string (e.g. "cuda", "cpu"). Defaults to "cuda" if available.
            dtype: torch dtype. Defaults to float32.
        """
        from transformers import AutoModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float32

        logger.info(f"Loading audio codec from {codec_path}")
        model = AutoModel.from_pretrained(
            codec_path, trust_remote_code=True
        )
        model = model.to(device=device, dtype=dtype).eval()

        sample_rate = getattr(model.config, "sample_rate", None) or getattr(
            model.config, "sampling_rate", 24000
        )

        codec = AudioCodec(model, sample_rate=int(sample_rate))
        logger.info(
            f"Audio codec loaded: sample_rate={codec.sample_rate}, "
            f"device={device}, dtype={dtype}"
        )
        return codec

    @torch.inference_mode()
    def encode(self, audio_path: str) -> np.ndarray:
        """Encode an audio file into discrete codec tokens.

        Used for reference audio in voice cloning.

        Args:
            audio_path: Path or URL to the audio file.

        Returns:
            Codec token array of shape (num_quantizers, seq_len).
        """
        device = next(self.model.parameters()).device

        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav_tensor = wav.unsqueeze(0).to(device)
        encode_result = self.model.encode(wav_tensor)

        if isinstance(encode_result, dict):
            codes = encode_result["audio_codes"]
        elif hasattr(encode_result, "audio_codes"):
            codes = encode_result.audio_codes
        else:
            raise ValueError(
                "Unsupported codec.encode() result: "
                "expected dict/object with 'audio_codes'."
            )
        return codes.detach().cpu().numpy()

    @torch.inference_mode()
    def encode_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Encode raw audio bytes into discrete codec tokens."""
        import io
        import soundfile as sf

        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.T  # (channels, samples)

        if sr != self.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.sample_rate
            )

        device = next(self.model.parameters()).device
        wav_tensor = audio_tensor.unsqueeze(0).to(device)
        encode_result = self.model.encode(wav_tensor)

        if isinstance(encode_result, dict):
            codes = encode_result["audio_codes"]
        elif hasattr(encode_result, "audio_codes"):
            codes = encode_result.audio_codes
        else:
            raise ValueError(
                "Unsupported codec.encode() result: "
                "expected dict/object with 'audio_codes'."
            )
        return codes.detach().cpu().numpy()

    @torch.inference_mode()
    def decode(
        self, tokens: torch.Tensor, chunk_duration: int = 8
    ) -> torch.Tensor:
        """Decode discrete codec tokens into an audio waveform.

        Similar to MOVA's MOVADecodingStage (audio_vae.decode).

        Args:
            tokens: Audio code tokens of shape (num_quantizers, seq_len)
                or (batch, num_quantizers, seq_len).
            chunk_duration: Chunk duration for codec decoding.

        Returns:
            Audio waveform tensor.
        """
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)

        decode_result = self.model.decode(
            tokens.permute(1, 0) if tokens.ndim == 2 else tokens,
            chunk_duration=chunk_duration,
        )

        if isinstance(decode_result, dict):
            audio = decode_result["audio"]
        elif hasattr(decode_result, "audio"):
            audio = decode_result.audio
        else:
            audio = decode_result

        return audio[0].cpu().float() if audio.ndim > 2 else audio.cpu().float()

    @torch.inference_mode()
    def decode_streaming(
        self,
        tokens: torch.Tensor,
        chunk_size: int = 50,
        chunk_duration: int = 8,
    ) -> Iterator[torch.Tensor]:
        """Decode tokens in chunks for streaming audio output.

        Args:
            tokens: Audio code tokens of shape (num_quantizers, seq_len).
            chunk_size: Number of time steps per chunk.
            chunk_duration: Codec chunk duration.

        Yields:
            Audio waveform chunks.
        """
        seq_len = tokens.shape[-1]
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_tokens = tokens[..., start:end]
            yield self.decode(chunk_tokens, chunk_duration=chunk_duration)

    @torch.inference_mode()
    def decode_rvq_codes(
        self,
        rvq_codes: list,
        chunk_duration: int = 8,
    ) -> np.ndarray:
        """Decode full multi-codebook RVQ codes to audio.

        This is the primary decode path. Each element of rvq_codes is a
        list of 16 integers representing one time step's worth of RVQ
        codes across all codebooks (as produced by the local transformer).

        Args:
            rvq_codes: List of time steps, each a list of num_codebooks
                integers. Shape conceptually (num_steps, num_codebooks).
                Each integer is in [0, 1023] (valid codes) or a special
                token (1024=pad, 1025=bos, 1026=eos).
            chunk_duration: Codec chunk duration parameter.

        Returns:
            Audio waveform as numpy array of shape (num_samples,).
        """
        if not rvq_codes:
            return np.array([], dtype=np.float32)

        # Filter out time steps that are entirely special tokens
        filtered = []
        for step_codes in rvq_codes:
            if isinstance(step_codes, (list, tuple)):
                # Keep the step if at least the first code is a valid audio code
                if len(step_codes) > 0 and 0 <= step_codes[0] < 1024:
                    filtered.append(step_codes)
            elif isinstance(step_codes, int):
                # Single-codebook fallback
                if 0 <= step_codes < 1024:
                    filtered.append([step_codes])

        if not filtered:
            return np.array([], dtype=np.float32)

        # Determine num_codebooks from data
        num_codebooks = max(len(step) for step in filtered)

        # Build token tensor: (num_codebooks, seq_len)
        device = next(self.model.parameters()).device
        seq_len = len(filtered)
        codes = torch.full(
            (num_codebooks, seq_len), 0, dtype=torch.long, device=device
        )
        for t, step_codes in enumerate(filtered):
            for c in range(num_codebooks):
                if c < len(step_codes):
                    codes[c, t] = step_codes[c]
                else:
                    # Pad missing codebooks with 0
                    codes[c, t] = 0

        audio = self.decode(codes, chunk_duration=chunk_duration)
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio.astype(np.float32)

    @torch.inference_mode()
    def decode_tokens(
        self,
        token_ids: list,
        num_codebooks: int = 16,
        chunk_duration: int = 8,
    ) -> np.ndarray:
        """Decode a flat list of first-codebook token IDs to audio.

        Fallback path when only first-codebook tokens are available (e.g.
        from output_ids). Prefer decode_rvq_codes() when full multi-codebook
        data is available from meta_info["audio_rvq_codes"].

        Args:
            token_ids: List of first-codebook audio token IDs.
            num_codebooks: Number of RVQ codebooks (default: 16).
            chunk_duration: Codec chunk duration parameter.

        Returns:
            Audio waveform as numpy array of shape (num_samples,).
        """
        # Wrap each token as a single-element list and delegate
        rvq_codes = [[t] * num_codebooks for t in token_ids]
        return self.decode_rvq_codes(rvq_codes, chunk_duration=chunk_duration)
