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
Audio format conversion utilities for TTS output.

Reuses audio normalization patterns from
multimodal_gen/runtime/entrypoints/utils.py (_normalize_audio_to_numpy)
and follows MOVA's ffmpeg subprocess pattern for compressed formats.
"""

from __future__ import annotations

import io
import logging
import shutil
import struct
import subprocess
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Supported output formats and their MIME types
FORMAT_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}


def normalize_audio_to_numpy(audio) -> Optional[np.ndarray]:
    """Convert audio (torch.Tensor or numpy) to float32 numpy array in [-1, 1].

    Adapted from multimodal_gen/runtime/entrypoints/utils.py
    (_normalize_audio_to_numpy).
    """
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().float().clamp(-1.0, 1.0).cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio.astype(np.float32)
            audio_np = np.clip(audio_np, -1.0, 1.0)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
    except Exception:
        return None

    # Squeeze leading singletons
    while audio_np.ndim > 1 and audio_np.shape[0] == 1:
        audio_np = audio_np.squeeze(0)

    # Transpose (C, L) -> (L, C) if needed
    if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T

    return audio_np


def convert_audio(
    waveform: np.ndarray,
    sample_rate: int,
    output_format: str,
) -> bytes:
    """Convert a normalized audio waveform to the requested output format.

    Args:
        waveform: Audio samples as float32 numpy array, shape (samples,)
            or (samples, channels). Values in [-1, 1].
        sample_rate: Sample rate in Hz.
        output_format: Target format (wav, mp3, opus, flac, aac, pcm).

    Returns:
        Audio file as bytes.
    """
    output_format = output_format.lower()

    if output_format == "pcm":
        return _to_pcm(waveform)
    elif output_format == "wav":
        return _to_wav(waveform, sample_rate)
    elif output_format in ("mp3", "opus", "flac", "aac"):
        return _to_compressed(waveform, sample_rate, output_format)
    else:
        raise ValueError(
            f"Unsupported audio format: {output_format}. "
            f"Supported: {list(FORMAT_MIME_TYPES.keys())}"
        )


def get_mime_type(output_format: str) -> str:
    """Get MIME type for an audio format."""
    fmt = output_format.lower()
    if fmt not in FORMAT_MIME_TYPES:
        raise ValueError(f"Unknown format: {fmt}")
    return FORMAT_MIME_TYPES[fmt]


def _to_pcm(waveform: np.ndarray) -> bytes:
    """Convert to raw PCM int16 bytes."""
    if waveform.ndim == 2:
        waveform = waveform[:, 0]  # Take first channel
    int16_data = (waveform * 32767).astype(np.int16)
    return int16_data.tobytes()


def _to_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Convert to WAV format using scipy (same pattern as MOVA)."""
    try:
        import scipy.io.wavfile

        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        int16_data = (waveform * 32767).astype(np.int16)

        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sample_rate, int16_data)
        return buf.getvalue()
    except ImportError:
        # Fallback: manual WAV construction
        return _to_wav_manual(waveform, sample_rate)


def _to_wav_manual(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Write WAV format manually (no dependencies)."""
    if waveform.ndim == 2:
        waveform = waveform[:, 0]
    int16_data = (waveform * 32767).astype(np.int16)
    num_samples = len(int16_data)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(int16_data.tobytes())
    return buf.getvalue()


def _to_compressed(
    waveform: np.ndarray, sample_rate: int, output_format: str
) -> bytes:
    """Convert to compressed format using ffmpeg subprocess.

    Follows MOVA's ffmpeg integration pattern from
    multimodal_gen/runtime/entrypoints/utils.py (_mux_audio_np_into_mp4).
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError(
            f"ffmpeg is required for {output_format} encoding but was not found. "
            "Install ffmpeg or use 'wav' or 'pcm' format."
        )

    # Write input as WAV to temp file
    wav_bytes = _to_wav(waveform, sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        input_path = f.name

    # Map format to ffmpeg codec
    codec_map = {
        "mp3": ["-codec:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-codec:a", "libopus", "-b:a", "128k"],
        "flac": ["-codec:a", "flac"],
        "aac": ["-codec:a", "aac", "-b:a", "128k", "-strict", "experimental"],
    }
    codec_args = codec_map.get(output_format, [])

    suffix = f".{output_format}"
    if output_format == "aac":
        suffix = ".m4a"  # AAC needs container

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        output_path = f.name

    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            *codec_args,
            output_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr.decode(errors='replace')}"
            )

        with open(output_path, "rb") as f:
            return f.read()
    finally:
        import os

        for path in (input_path, output_path):
            try:
                os.unlink(path)
            except OSError:
                pass


class AudioFormatConverter:
    """Convenience wrapper around the audio format conversion functions.

    Provides an object-oriented interface for use in serving handlers.
    """

    def convert(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        output_format: str,
    ) -> bytes:
        """Convert audio waveform to the requested format."""
        normalized = normalize_audio_to_numpy(waveform)
        if normalized is None:
            normalized = waveform
        return convert_audio(normalized, sample_rate, output_format)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to float32 in [-1, 1]."""
        result = normalize_audio_to_numpy(audio)
        if result is None:
            return np.clip(audio.astype(np.float32), -1.0, 1.0)
        return result
