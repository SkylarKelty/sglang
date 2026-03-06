"""Unit tests for the audio codec and format converter modules."""

import io
import struct
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")


class TestAudioFormatConverter(CustomTestCase):
    """Test audio format conversion utilities."""

    def setUp(self):
        from sglang.srt.audio.format_converter import AudioFormatConverter

        self.converter = AudioFormatConverter()
        # Create a simple sine wave test signal
        self.sample_rate = 24000
        t = np.linspace(0, 0.1, int(self.sample_rate * 0.1), dtype=np.float32)
        self.test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_convert_wav(self):
        """Test WAV format conversion produces valid WAV bytes."""
        result = self.converter.convert(self.test_audio, self.sample_rate, "wav")
        self.assertIsInstance(result, bytes)
        self.assertTrue(len(result) > 44)  # WAV header is 44 bytes
        # Check RIFF header
        self.assertEqual(result[:4], b"RIFF")

    def test_convert_pcm(self):
        """Test PCM format produces raw audio bytes."""
        result = self.converter.convert(self.test_audio, self.sample_rate, "pcm")
        self.assertIsInstance(result, bytes)
        # PCM should be raw 16-bit samples
        expected_len = len(self.test_audio) * 2  # 16-bit = 2 bytes per sample
        self.assertEqual(len(result), expected_len)

    def test_normalize_audio(self):
        """Test audio normalization to [-1, 1] range."""
        # Create audio with values outside [-1, 1]
        loud_audio = np.array([2.0, -3.0, 1.5], dtype=np.float32)
        normalized = self.converter.normalize(loud_audio)
        self.assertTrue(np.all(normalized >= -1.0))
        self.assertTrue(np.all(normalized <= 1.0))

    def test_empty_audio(self):
        """Test handling of empty audio array."""
        empty = np.array([], dtype=np.float32)
        result = self.converter.convert(empty, self.sample_rate, "wav")
        self.assertIsInstance(result, bytes)


class TestSpeechProtocol(CustomTestCase):
    """Test the SpeechRequest protocol model."""

    def test_speech_request_defaults(self):
        """Test SpeechRequest has correct defaults."""
        from sglang.srt.entrypoints.openai.protocol import SpeechRequest

        req = SpeechRequest(input="Hello world")
        self.assertEqual(req.voice, "default")
        self.assertEqual(req.response_format, "wav")
        self.assertEqual(req.speed, 1.0)
        self.assertFalse(req.stream)
        self.assertIsNone(req.instructions)
        self.assertIsNone(req.reference_audio_data)

    def test_speech_request_custom(self):
        """Test SpeechRequest with custom values."""
        from sglang.srt.entrypoints.openai.protocol import SpeechRequest

        req = SpeechRequest(
            input="Test text",
            voice="alloy",
            response_format="mp3",
            stream=True,
        )
        self.assertEqual(req.input, "Test text")
        self.assertEqual(req.voice, "alloy")
        self.assertEqual(req.response_format, "mp3")
        self.assertTrue(req.stream)


class TestModelConfig(CustomTestCase):
    """Test TTS model detection in model config."""

    def test_is_tts_model_detection(self):
        """Test that is_tts_model correctly detects MossTTSRealtime."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertTrue(is_tts_model(["MossTTSRealtimeForCausalLM"]))

    def test_is_not_tts_model(self):
        """Test that non-TTS models are not detected as TTS."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertFalse(is_tts_model(["LlamaForCausalLM"]))

    def test_is_tts_model_empty(self):
        """Test handling of empty architectures list."""
        from sglang.srt.configs.model_config import is_tts_model

        self.assertFalse(is_tts_model([]))


class TestAudioCodecDecodeTokens(CustomTestCase):
    """Test AudioCodec.decode_tokens with mocked model."""

    def test_decode_tokens_filters_special(self):
        """Test that special tokens (EOS, BOS, PAD) are filtered out."""
        from sglang.srt.audio.codec import AudioCodec

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [MagicMock(device=MagicMock(type="cpu"))]
        )

        codec = AudioCodec(mock_model, sample_rate=24000)

        # All special tokens should result in empty array
        result = codec.decode_tokens([1024, 1025, 1026])
        self.assertEqual(len(result), 0)

    def test_decode_tokens_empty(self):
        """Test decode with empty token list."""
        from sglang.srt.audio.codec import AudioCodec

        mock_model = MagicMock()
        codec = AudioCodec(mock_model, sample_rate=24000)

        result = codec.decode_tokens([])
        self.assertEqual(len(result), 0)


class TestAudioCodecDecodeRvqCodes(CustomTestCase):
    """Test AudioCodec.decode_rvq_codes with mocked model."""

    def test_decode_rvq_codes_empty(self):
        """Test decode_rvq_codes with empty input."""
        from sglang.srt.audio.codec import AudioCodec

        mock_model = MagicMock()
        codec = AudioCodec(mock_model, sample_rate=24000)
        result = codec.decode_rvq_codes([])
        self.assertEqual(len(result), 0)

    def test_decode_rvq_codes_filters_special_tokens(self):
        """Test that time steps with only special tokens are filtered."""
        from sglang.srt.audio.codec import AudioCodec

        mock_model = MagicMock()
        codec = AudioCodec(mock_model, sample_rate=24000)
        # All special tokens: pad=1024, bos=1025, eos=1026
        result = codec.decode_rvq_codes([[1024] * 16, [1025] * 16, [1026] * 16])
        self.assertEqual(len(result), 0)

    def test_decode_rvq_codes_builds_correct_shape(self):
        """Test that decode_rvq_codes builds the right tensor shape."""
        import torch

        from sglang.srt.audio.codec import AudioCodec

        mock_model = MagicMock()
        # Make parameters() return a tensor so device detection works
        param = torch.zeros(1)
        mock_model.parameters.return_value = iter([param])
        # Mock decode to return a 1D waveform
        mock_model.decode.return_value = torch.randn(1, 1, 480)

        codec = AudioCodec(mock_model, sample_rate=24000)
        result = codec.decode_rvq_codes(
            [[10, 20, 30] + [0] * 13, [11, 21, 31] + [0] * 13]
        )
        # Should have called model.decode with a tensor
        self.assertTrue(mock_model.decode.called)
        call_args = mock_model.decode.call_args
        codes_arg = call_args[0][0] if call_args[0] else call_args[1].get("tokens")
        # The tensor passed to self.decode() has shape (num_codebooks, seq_len)
        self.assertEqual(codes_arg.shape[-1], 2)  # 2 time steps


class TestIOStructOutputModality(CustomTestCase):
    """Test that output_modality field works in GenerateReqInput."""

    def test_output_modality_default(self):
        """Test that output_modality defaults to None."""
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(text="test")
        self.assertIsNone(req.output_modality)

    def test_output_modality_audio(self):
        """Test setting output_modality to audio."""
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(text="test", output_modality="audio")
        self.assertEqual(req.output_modality, "audio")


if __name__ == "__main__":
    unittest.main(verbosity=3)
