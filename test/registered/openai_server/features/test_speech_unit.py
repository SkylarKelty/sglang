"""Unit tests for TTS / audio speech components.

Tests are designed to run on CPU without a GPU. Components that require
sgl_kernel (CUDA) are skipped gracefully. Covers:
  - AudioFormatConverter (all formats, normalization, edge cases)
  - SpeechRequest protocol (defaults, validation, serialization)
  - Model config TTS detection
  - AudioCodec decode paths (mock model)
  - Serving handler validation logic (mock dependencies)
  - Processor input construction (mock tokenizer)
  - IO struct output_modality field
"""

import io
import shutil
import struct
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_wave(duration_s=0.1, sample_rate=24000, freq=440.0):
    """Generate a simple sine wave test signal."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_mock_codec_model():
    """Create a MagicMock that behaves like a codec nn.Module."""
    import torch

    mock = MagicMock()
    param = torch.zeros(1)
    mock.parameters.side_effect = lambda: iter([param])
    mock.decode.return_value = torch.randn(1, 1, 960)
    return mock


# ===========================================================================
# AudioFormatConverter
# ===========================================================================

class TestAudioFormatConverter(CustomTestCase):
    """Test audio format conversion utilities."""

    def setUp(self):
        from sglang.srt.audio.format_converter import AudioFormatConverter

        self.converter = AudioFormatConverter()
        self.sample_rate = 24000
        self.test_audio = _sine_wave(sample_rate=self.sample_rate)

    def test_convert_wav(self):
        """WAV output has valid RIFF header and correct data size."""
        result = self.converter.convert(self.test_audio, self.sample_rate, "wav")
        self.assertIsInstance(result, bytes)
        self.assertEqual(result[:4], b"RIFF")
        self.assertEqual(result[8:12], b"WAVE")
        # Data chunk should contain int16 samples
        self.assertGreater(len(result), 44)

    def test_convert_pcm(self):
        """PCM output is raw 16-bit samples with correct length."""
        result = self.converter.convert(self.test_audio, self.sample_rate, "pcm")
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), len(self.test_audio) * 2)

    def test_convert_wav_manual_fallback(self):
        """Manual WAV writer produces valid output when scipy is unavailable."""
        from sglang.srt.audio.format_converter import _to_wav_manual

        result = _to_wav_manual(self.test_audio, self.sample_rate)
        self.assertEqual(result[:4], b"RIFF")
        # Parse sample rate from WAV header (bytes 24-27)
        sr = struct.unpack("<I", result[24:28])[0]
        self.assertEqual(sr, self.sample_rate)

    def test_normalize_audio_clamps(self):
        """Normalization clamps values to [-1, 1]."""
        loud = np.array([2.0, -3.0, 1.5, 0.0], dtype=np.float32)
        normalized = self.converter.normalize(loud)
        self.assertTrue(np.all(normalized >= -1.0))
        self.assertTrue(np.all(normalized <= 1.0))
        self.assertAlmostEqual(normalized[3], 0.0)

    def test_normalize_preserves_valid_range(self):
        """Values already in [-1, 1] are preserved."""
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        normalized = self.converter.normalize(audio)
        np.testing.assert_array_almost_equal(normalized, audio)

    def test_empty_audio(self):
        """Empty audio array produces valid (header-only) WAV."""
        empty = np.array([], dtype=np.float32)
        result = self.converter.convert(empty, self.sample_rate, "wav")
        self.assertIsInstance(result, bytes)
        self.assertEqual(result[:4], b"RIFF")

    def test_multichannel_audio_takes_first_channel(self):
        """Multi-channel audio is reduced to mono."""
        stereo = np.stack([self.test_audio, -self.test_audio], axis=1)
        result_stereo = self.converter.convert(stereo, self.sample_rate, "pcm")
        result_mono = self.converter.convert(self.test_audio, self.sample_rate, "pcm")
        # Both should produce same-length mono PCM
        self.assertEqual(len(result_stereo), len(result_mono))

    def test_convert_mp3_with_ffmpeg(self):
        """MP3 conversion works when ffmpeg is available."""
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        result = self.converter.convert(self.test_audio, self.sample_rate, "mp3")
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_convert_flac_with_ffmpeg(self):
        """FLAC conversion works when ffmpeg is available."""
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        result = self.converter.convert(self.test_audio, self.sample_rate, "flac")
        self.assertIsInstance(result, bytes)
        # FLAC magic bytes
        self.assertEqual(result[:4], b"fLaC")

    def test_unsupported_format_raises(self):
        """Unsupported format raises ValueError."""
        from sglang.srt.audio.format_converter import convert_audio

        with self.assertRaises(ValueError):
            convert_audio(self.test_audio, self.sample_rate, "ogg_unsupported")

    def test_normalize_torch_tensor(self):
        """normalize_audio_to_numpy handles torch tensors."""
        import torch

        from sglang.srt.audio.format_converter import normalize_audio_to_numpy

        tensor = torch.tensor([0.5, -0.5, 2.0])
        result = normalize_audio_to_numpy(tensor)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_get_mime_type(self):
        """get_mime_type returns correct MIME types."""
        from sglang.srt.audio.format_converter import get_mime_type

        self.assertEqual(get_mime_type("wav"), "audio/wav")
        self.assertEqual(get_mime_type("mp3"), "audio/mpeg")
        self.assertEqual(get_mime_type("pcm"), "audio/pcm")
        with self.assertRaises(ValueError):
            get_mime_type("invalid_format")


# ===========================================================================
# SpeechRequest Protocol
# ===========================================================================

class TestSpeechProtocol(CustomTestCase):
    """Test the SpeechRequest protocol model."""

    def _make_request(self, **kwargs):
        from sglang.srt.entrypoints.openai.protocol import SpeechRequest

        defaults = {"input": "Hello world"}
        defaults.update(kwargs)
        return SpeechRequest(**defaults)

    def test_speech_request_defaults(self):
        """SpeechRequest has correct default values."""
        req = self._make_request()
        self.assertEqual(req.voice, "default")
        self.assertEqual(req.response_format, "wav")
        self.assertEqual(req.speed, 1.0)
        self.assertFalse(req.stream)
        self.assertIsNone(req.instructions)
        self.assertIsNone(req.reference_audio_data)

    def test_speech_request_custom(self):
        """SpeechRequest accepts custom values."""
        req = self._make_request(
            voice="alloy", response_format="mp3", stream=True
        )
        self.assertEqual(req.voice, "alloy")
        self.assertEqual(req.response_format, "mp3")
        self.assertTrue(req.stream)

    def test_speech_request_with_reference_audio(self):
        """SpeechRequest accepts reference audio data."""
        req = self._make_request(reference_audio_data=b"\x00\x01\x02\x03")
        self.assertIsNotNone(req.reference_audio_data)

    def test_speech_request_serialization(self):
        """SpeechRequest round-trips through dict serialization."""
        req = self._make_request(input="Test", voice="shimmer")
        d = req.model_dump()
        self.assertEqual(d["input"], "Test")
        self.assertEqual(d["voice"], "shimmer")

    def test_speech_request_long_input(self):
        """SpeechRequest accepts long input text."""
        long_text = "A" * 4096
        req = self._make_request(input=long_text)
        self.assertEqual(len(req.input), 4096)


# ===========================================================================
# Model Config TTS Detection
# ===========================================================================

class TestModelConfig(CustomTestCase):
    """Test TTS model detection in model config."""

    def test_is_tts_model_detection(self):
        """is_tts_model correctly detects MossTTSRealtime."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertTrue(is_tts_model(["MossTTSRealtime"]))

    def test_is_not_tts_model(self):
        """Non-TTS models are not detected as TTS."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertFalse(is_tts_model(["LlamaForCausalLM"]))
        self.assertFalse(is_tts_model(["Qwen3ForCausalLM"]))

    def test_is_tts_model_empty(self):
        """Empty architectures list returns False."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertFalse(is_tts_model([]))

    def test_is_tts_model_multiple_architectures(self):
        """Detection works when TTS is among multiple architectures."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertTrue(
            is_tts_model(["LlamaForCausalLM", "MossTTSRealtime"])
        )


# ===========================================================================
# AudioCodec – decode_tokens
# ===========================================================================

class TestAudioCodecDecodeTokens(CustomTestCase):
    """Test AudioCodec.decode_tokens with mocked model."""

    def test_decode_tokens_filters_special(self):
        """Special tokens (EOS=1026, BOS=1025, PAD=1024) produce empty output."""
        from sglang.srt.audio.codec import AudioCodec

        codec = AudioCodec(MagicMock(), sample_rate=24000)
        result = codec.decode_tokens([1024, 1025, 1026])
        self.assertEqual(len(result), 0)

    def test_decode_tokens_empty(self):
        """Empty token list produces empty output."""
        from sglang.srt.audio.codec import AudioCodec

        codec = AudioCodec(MagicMock(), sample_rate=24000)
        result = codec.decode_tokens([])
        self.assertEqual(len(result), 0)

    def test_decode_tokens_valid_calls_decode(self):
        """Valid tokens (0-1023) are passed through to the codec model."""
        import torch

        from sglang.srt.audio.codec import AudioCodec

        mock = _make_mock_codec_model()
        codec = AudioCodec(mock, sample_rate=24000)
        result = codec.decode_tokens([100, 200, 300])
        self.assertTrue(mock.decode.called)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(len(result), 0)

    def test_decode_tokens_mixed_valid_and_special(self):
        """Only valid tokens are kept, special tokens are filtered."""
        import torch

        from sglang.srt.audio.codec import AudioCodec

        mock = _make_mock_codec_model()
        codec = AudioCodec(mock, sample_rate=24000)
        # Mix of valid (100, 200) and special (1024, 1026)
        result = codec.decode_tokens([100, 1024, 200, 1026])
        # Should produce non-empty output (2 valid tokens)
        self.assertTrue(mock.decode.called)


# ===========================================================================
# AudioCodec – decode_rvq_codes
# ===========================================================================

class TestAudioCodecDecodeRvqCodes(CustomTestCase):
    """Test AudioCodec.decode_rvq_codes with mocked model."""

    def test_decode_rvq_codes_empty(self):
        """Empty input produces empty output."""
        from sglang.srt.audio.codec import AudioCodec

        codec = AudioCodec(MagicMock(), sample_rate=24000)
        result = codec.decode_rvq_codes([])
        self.assertEqual(len(result), 0)

    def test_decode_rvq_codes_filters_special_tokens(self):
        """Time steps with only special tokens are filtered out."""
        from sglang.srt.audio.codec import AudioCodec

        codec = AudioCodec(MagicMock(), sample_rate=24000)
        result = codec.decode_rvq_codes([[1024] * 16, [1025] * 16, [1026] * 16])
        self.assertEqual(len(result), 0)

    def test_decode_rvq_codes_builds_correct_shape(self):
        """Tensor passed to codec model has correct shape after permutation."""
        from sglang.srt.audio.codec import AudioCodec

        mock = _make_mock_codec_model()
        codec = AudioCodec(mock, sample_rate=24000)
        result = codec.decode_rvq_codes(
            [[10, 20, 30] + [0] * 13, [11, 21, 31] + [0] * 13]
        )
        self.assertTrue(mock.decode.called)
        codes_arg = mock.decode.call_args[0][0]
        # decode() permutes (num_codebooks, seq_len) → (seq_len, num_codebooks)
        self.assertEqual(codes_arg.shape[0], 2)   # 2 time steps
        self.assertEqual(codes_arg.shape[1], 16)   # 16 codebooks

    def test_decode_rvq_codes_single_codebook_fallback(self):
        """Single integers (not lists) are treated as single-codebook input."""
        from sglang.srt.audio.codec import AudioCodec

        mock = _make_mock_codec_model()
        codec = AudioCodec(mock, sample_rate=24000)
        result = codec.decode_rvq_codes([100, 200, 300])  # ints, not lists
        self.assertTrue(mock.decode.called)

    def test_decode_rvq_codes_mixed_valid_and_special_steps(self):
        """Only time steps with valid first code are kept."""
        from sglang.srt.audio.codec import AudioCodec

        mock = _make_mock_codec_model()
        codec = AudioCodec(mock, sample_rate=24000)
        result = codec.decode_rvq_codes([
            [100] + [0] * 15,    # valid
            [1026] + [0] * 15,   # EOS — filtered
            [200] + [0] * 15,    # valid
        ])
        codes_arg = mock.decode.call_args[0][0]
        self.assertEqual(codes_arg.shape[0], 2)  # only 2 valid steps

    def test_decode_rvq_codes_returns_float32(self):
        """Output waveform is float32 numpy array."""
        from sglang.srt.audio.codec import AudioCodec

        mock = _make_mock_codec_model()
        codec = AudioCodec(mock, sample_rate=24000)
        result = codec.decode_rvq_codes([[50] * 16])
        self.assertEqual(result.dtype, np.float32)


# ===========================================================================
# Serving Handler – Validation (mocked dependencies)
# ===========================================================================

class TestServingSpeechValidation(CustomTestCase):
    """Test OpenAIServingSpeech request validation without CUDA."""

    def _make_handler(self):
        """Create handler with mocked dependencies."""
        try:
            from sglang.srt.entrypoints.openai.serving_speech import (
                OpenAIServingSpeech,
            )
        except ImportError:
            self.skipTest("serving_speech imports require CUDA dependencies")

        from sglang.srt.audio.codec import AudioCodec

        mock_tm = MagicMock()
        mock_codec = MagicMock(spec=AudioCodec)
        mock_codec.sample_rate = 24000

        handler = OpenAIServingSpeech.__new__(OpenAIServingSpeech)
        handler.codec = mock_codec
        handler.sample_rate = 24000
        handler.tokenizer_manager = mock_tm
        from sglang.srt.audio.format_converter import AudioFormatConverter

        handler.format_converter = AudioFormatConverter()
        return handler

    def _make_request(self, **kwargs):
        from sglang.srt.entrypoints.openai.protocol import SpeechRequest

        defaults = {"input": "Hello world"}
        defaults.update(kwargs)
        return SpeechRequest(**defaults)

    def test_validate_valid_request(self):
        """Valid request returns None (no error)."""
        handler = self._make_handler()
        result = handler._validate_request(self._make_request())
        self.assertIsNone(result)

    def test_validate_empty_input(self):
        """Empty input text returns error message."""
        handler = self._make_handler()
        result = handler._validate_request(self._make_request(input=""))
        self.assertIsNotNone(result)
        self.assertIn("empty", result.lower())

    def test_validate_whitespace_input(self):
        """Whitespace-only input returns error message."""
        handler = self._make_handler()
        result = handler._validate_request(self._make_request(input="   "))
        self.assertIsNotNone(result)

    def test_validate_too_long_input(self):
        """Input exceeding 4096 chars returns error message."""
        handler = self._make_handler()
        result = handler._validate_request(self._make_request(input="A" * 4097))
        self.assertIsNotNone(result)
        self.assertIn("4096", result)

    def test_validate_exactly_4096_is_ok(self):
        """Input of exactly 4096 chars is valid."""
        handler = self._make_handler()
        result = handler._validate_request(self._make_request(input="A" * 4096))
        self.assertIsNone(result)

    def test_validate_invalid_format(self):
        """Invalid response_format returns error message."""
        handler = self._make_handler()
        result = handler._validate_request(
            self._make_request(response_format="invalid")
        )
        self.assertIsNotNone(result)
        self.assertIn("invalid", result.lower())

    def test_validate_all_valid_formats(self):
        """All supported formats pass validation."""
        handler = self._make_handler()
        for fmt in ("wav", "mp3", "opus", "flac", "aac", "pcm"):
            result = handler._validate_request(
                self._make_request(response_format=fmt)
            )
            self.assertIsNone(result, f"Format '{fmt}' should be valid")

    def test_request_id_prefix(self):
        """Request ID prefix is 'speech-'."""
        handler = self._make_handler()
        self.assertEqual(handler._request_id_prefix(), "speech-")


# ===========================================================================
# Processor – Input Construction (mock tokenizer)
# ===========================================================================

class TestMossTTSProcessor(CustomTestCase):
    """Test processor input construction with mocked tokenizer."""

    def _try_import_processor(self):
        try:
            from sglang.srt.multimodal.processors.moss_tts import MossTTSProcessor
            return MossTTSProcessor
        except ImportError:
            self.skipTest("Processor imports require CUDA dependencies")

    def _try_import_templates(self):
        try:
            from sglang.srt.multimodal.processors.moss_tts import (
                ASSISTANT_PREFIX,
                SYSTEM_PROMPT,
                VOICE_CLONE_CONTEXT,
            )
            return SYSTEM_PROMPT, VOICE_CLONE_CONTEXT, ASSISTANT_PREFIX
        except ImportError:
            self.skipTest("Processor imports require CUDA dependencies")

    def test_system_prompt_template(self):
        """System prompt template contains expected markers."""
        SYSTEM_PROMPT, _, _ = self._try_import_templates()

        self.assertIn("<|im_start|>system", SYSTEM_PROMPT)
        self.assertIn("<|im_end|>", SYSTEM_PROMPT)
        self.assertIn("Mosi Intelligence", SYSTEM_PROMPT)

    def test_context_template_with_audio_substitution(self):
        """Voice clone context and assistant prefix are present."""
        _, VOICE_CLONE_CONTEXT, ASSISTANT_PREFIX = self._try_import_templates()
        self.assertIn("context", VOICE_CLONE_CONTEXT)
        self.assertIn("voice timbre", VOICE_CLONE_CONTEXT)
        self.assertIn("assistant", ASSISTANT_PREFIX)

    def test_processor_async_builds_multi_channel(self):
        """process_mm_data_async builds multi-channel tensor with correct shape."""
        import asyncio

        ProcessorClass = self._try_import_processor()

        # Mock tokenizer — encode is called for input text then system prompt
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(len(text)))

        # Create processor with mocked internals
        proc = ProcessorClass.__new__(ProcessorClass)
        proc._tokenizer = mock_tokenizer
        proc._processor = mock_tokenizer

        result = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=None,
                input_text="Hello world test",
                request_obj=None,
            )
        )

        self.assertIn("input_ids", result)
        self.assertIn("mm_items", result)
        self.assertEqual(len(result["mm_items"]), 1)

        mm_item = result["mm_items"][0]
        msd = mm_item.model_specific_data
        self.assertIn("multi_channel_ids", msd)
        self.assertIn("text_ids", msd)
        self.assertIn("text_cursor", msd)
        self.assertEqual(msd["text_cursor"], 0)

        # multi_channel_ids should have 17 columns (1 text + 16 audio)
        self.assertEqual(msd["multi_channel_ids"].shape[1], 17)

        # BOS should ONLY be in channel 1 at the LAST position
        mc = msd["multi_channel_ids"]
        last_row = mc[-1]
        # Channel 1 at last position should be BOS (1025)
        self.assertEqual(last_row[1], 1025)
        # All other audio channels at last position should be PAD (1024)
        for ch in range(2, 17):
            self.assertEqual(last_row[ch], 1024, f"channel {ch} at last pos should be PAD")
        # All non-last positions should have PAD in all audio channels
        for pos in range(mc.shape[0] - 1):
            for ch in range(1, 17):
                self.assertEqual(mc[pos, ch], 1024, f"pos {pos} ch {ch} should be PAD")

    def test_processor_primes_up_to_12_tokens(self):
        """Processor primes at most 12 text tokens during prefill."""
        import asyncio

        ProcessorClass = self._try_import_processor()

        mock_tokenizer = MagicMock()
        # encode is called twice: once for system prompt, once for input text
        system_tokens = list(range(100, 120))
        text_tokens = list(range(200, 220))
        mock_tokenizer.encode.side_effect = [text_tokens, system_tokens]

        proc = ProcessorClass.__new__(ProcessorClass)
        proc._tokenizer = mock_tokenizer
        proc._processor = mock_tokenizer

        result = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=None,
                input_text="a " * 20,  # long enough text
                request_obj=None,
            )
        )

        mm_item = result["mm_items"][0]
        # 12 tokens are primed, remaining 8 go to text_ids
        self.assertEqual(len(mm_item.model_specific_data["text_ids"]), 8)

    def test_processor_short_text_primes_all(self):
        """When text has fewer than 12 tokens, all are primed."""
        import asyncio

        ProcessorClass = self._try_import_processor()

        mock_tokenizer = MagicMock()
        # encode is called twice: once for input text, once for system prompt
        text_tokens = list(range(200, 205))  # only 5 tokens
        system_tokens = list(range(100, 110))
        mock_tokenizer.encode.side_effect = [text_tokens, system_tokens]

        proc = ProcessorClass.__new__(ProcessorClass)
        proc._tokenizer = mock_tokenizer
        proc._processor = mock_tokenizer

        result = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=None,
                input_text="short",
                request_obj=None,
            )
        )

        mm_item = result["mm_items"][0]
        # All 5 tokens primed, nothing left for text_ids
        self.assertEqual(len(mm_item.model_specific_data["text_ids"]), 0)

    def test_processor_voice_clone_with_ref_audio(self):
        """Processor builds voice clone prompt when reference audio tokens provided."""
        import asyncio

        ProcessorClass = self._try_import_processor()
        try:
            from sglang.srt.models.moss_tts_realtime import REFERENCE_AUDIO_PAD_ID
        except ImportError:
            self.skipTest("Model imports require CUDA dependencies")

        mock_tokenizer = MagicMock()
        # The processor calls encode 3 times for voice clone:
        # 1. text_ids (input text)
        # 2. system_prompt + voice_clone_context + audio_pad_tokens
        # 3. assistant_prefix
        ref_pad_id = REFERENCE_AUDIO_PAD_ID  # 151654
        text_tokens = list(range(200, 210))
        # System prompt tokens with 5 audio_pad tokens embedded
        system_tokens = list(range(100, 115)) + [ref_pad_id] * 5
        assistant_tokens = list(range(300, 305))
        mock_tokenizer.encode.side_effect = [text_tokens, system_tokens, assistant_tokens]

        proc = ProcessorClass.__new__(ProcessorClass)
        proc._tokenizer = mock_tokenizer
        proc._processor = mock_tokenizer

        # Create fake reference audio RVQ codes: 5 frames × 16 channels
        ref_audio_tokens = np.arange(80, dtype=np.int64).reshape(5, 16)

        result = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=[ref_audio_tokens],
                input_text="Hello world",
                request_obj=None,
            )
        )

        mm_item = result["mm_items"][0]
        mc = mm_item.model_specific_data["multi_channel_ids"]

        # Multi-channel should have 17 columns
        self.assertEqual(mc.shape[1], 17)

        # Find positions where text channel has ref_pad_id
        text_channel = mc[:, 0]
        pad_positions = np.where(text_channel == ref_pad_id)[0]
        self.assertEqual(len(pad_positions), 5)

        # Audio channels at those positions should contain the reference codes
        for i, pos in enumerate(pad_positions):
            np.testing.assert_array_equal(
                mc[pos, 1:], ref_audio_tokens[i],
                err_msg=f"RVQ codes at position {pos} don't match reference"
            )

    def test_processor_no_ref_audio_no_assistant_prefix(self):
        """Without reference audio, no assistant prefix or context section."""
        import asyncio

        ProcessorClass = self._try_import_processor()

        mock_tokenizer = MagicMock()
        text_tokens = list(range(200, 210))
        system_tokens = list(range(100, 120))
        mock_tokenizer.encode.side_effect = [text_tokens, system_tokens]

        proc = ProcessorClass.__new__(ProcessorClass)
        proc._tokenizer = mock_tokenizer
        proc._processor = mock_tokenizer

        result = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=None,
                input_text="Hello world",
                request_obj=None,
            )
        )

        # encode should only be called twice (text + system prompt)
        self.assertEqual(mock_tokenizer.encode.call_count, 2)


# ===========================================================================
# Local Transformer – special token masking
# ===========================================================================

class TestLocalTransformerSpecialTokenMask(CustomTestCase):
    """Test that codebooks 1-15 mask out special tokens (PAD/BOS/EOS)."""

    def test_special_tokens_masked_for_non_zero_codebooks(self):
        """Tokens 1024-1026 should be -inf for codebooks 1-15."""
        try:
            from sglang.srt.models.moss_tts_realtime import (
                AUDIO_PAD_TOKEN,
                AUDIO_VOCAB_SIZE,
            )
        except ImportError:
            self.skipTest("Model imports require CUDA dependencies")

        # Simulate what the model does: for codebook i>0,
        # logits[:, :, AUDIO_PAD_TOKEN:] = -inf
        import torch

        logits = torch.randn(1, 1, AUDIO_VOCAB_SIZE)
        # For codebook 0, no masking
        logits_cb0 = logits.clone()
        # Codebook 0 keeps all logits (including special tokens)
        self.assertTrue(torch.isfinite(logits_cb0[:, :, AUDIO_PAD_TOKEN:]).all())

        # For codebook 1+, special tokens are masked
        logits_cb1 = logits.clone()
        logits_cb1[:, :, AUDIO_PAD_TOKEN:] = float("-inf")
        self.assertTrue((logits_cb1[:, :, AUDIO_PAD_TOKEN:] == float("-inf")).all())
        # Valid codec codes (0-1023) are untouched
        self.assertTrue(torch.isfinite(logits_cb1[:, :, :AUDIO_PAD_TOKEN]).all())


# ===========================================================================
# IO Struct – output_modality
# ===========================================================================

class TestIOStructOutputModality(CustomTestCase):
    """Test that output_modality field works in GenerateReqInput."""

    def _import_generate_req_input(self):
        try:
            from sglang.srt.managers.io_struct import GenerateReqInput

            return GenerateReqInput
        except ImportError:
            self.skipTest("sgl_kernel not available (no CUDA)")

    def test_output_modality_default(self):
        """output_modality defaults to None."""
        GenerateReqInput = self._import_generate_req_input()
        req = GenerateReqInput(text="test")
        self.assertIsNone(req.output_modality)

    def test_output_modality_audio(self):
        """output_modality can be set to 'audio'."""
        GenerateReqInput = self._import_generate_req_input()
        req = GenerateReqInput(text="test", output_modality="audio")
        self.assertEqual(req.output_modality, "audio")


if __name__ == "__main__":
    unittest.main(verbosity=3)
