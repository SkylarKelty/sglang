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
OpenAI-compatible speech synthesis endpoint handler for TTS models.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Union

import numpy as np
from fastapi import Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.audio.codec import AudioCodec
from sglang.srt.audio.format_converter import AudioFormatConverter
from sglang.srt.entrypoints.openai.protocol import SpeechRequest
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Content-type mapping for audio formats
AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "audio/L16",
}

# Recommended sampling parameters for MOSS-TTS-Realtime
DEFAULT_TTS_SAMPLING_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.6,
    "top_k": 30,
    "repetition_penalty": 1.1,
    "max_new_tokens": 5000,  # ~40 seconds of audio
    "stop_token_ids": [1026],  # Audio EOS token — stop when codebook-0 emits EOS
}


class OpenAIServingSpeech(OpenAIServingBase):
    """Handler for /v1/audio/speech requests."""

    def __init__(
        self,
        tokenizer_manager: "TokenizerManager",
        codec: AudioCodec,
        sample_rate: int = 24000,
    ):
        super().__init__(tokenizer_manager)
        self.codec = codec
        self.sample_rate = sample_rate
        self.format_converter = AudioFormatConverter()

    def _request_id_prefix(self) -> str:
        return "speech-"

    def _validate_request(self, request: SpeechRequest) -> Optional[str]:
        """Validate speech synthesis request."""
        if not request.input or not request.input.strip():
            return "Input text must not be empty."

        if len(request.input) > 4096:
            return "Input text must be 4096 characters or less."

        valid_formats = set(AUDIO_CONTENT_TYPES.keys())
        if request.response_format not in valid_formats:
            return (
                f"Invalid response_format '{request.response_format}'. "
                f"Supported formats: {', '.join(sorted(valid_formats))}"
            )

        if request.speed != 1.0:
            logger.warning(
                "Speed parameter is not supported by MOSS-TTS-Realtime "
                "and will be ignored."
            )

        if request.instructions:
            logger.warning(
                "Instructions parameter is not supported by MOSS-TTS-Realtime "
                "and will be ignored."
            )

        return None

    def _convert_to_internal_request(
        self,
        request: SpeechRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, SpeechRequest]:
        """Convert speech request to internal format.

        If reference audio is provided, encode it into RVQ codes using
        the codec before passing to the processor.  The processor will
        use the pre-encoded tokens to build the voice-clone prompt.
        """
        audio_data = None
        if request.reference_audio_data:
            try:
                rvq_codes = self.codec.encode_bytes(request.reference_audio_data)
                # rvq_codes shape: (1, num_quantizers, seq_len) or (num_quantizers, seq_len)
                codes = np.array(rvq_codes).squeeze()
                if codes.ndim == 2:
                    # Transpose to (T, channels) for the processor
                    if codes.shape[0] == 16 and codes.shape[1] != 16:
                        codes = codes.T
                audio_data = [codes]
                logger.info(
                    f"Encoded reference audio: shape={codes.shape}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to encode reference audio, ignoring: {e}"
                )

        adapted_request = GenerateReqInput(
            text=request.input,
            audio_data=audio_data,
            sampling_params=DEFAULT_TTS_SAMPLING_PARAMS.copy(),
            stream=request.stream,
            modalities=["audio"],
            output_modality="audio",
            routing_key=self.extract_routing_key(raw_request),
        )

        return adapted_request, request

    async def create_speech(
        self,
        request: SpeechRequest,
        raw_request: Request,
    ) -> Union[Response, StreamingResponse, ORJSONResponse]:
        """Main entry point for speech synthesis requests."""
        return await self.handle_request(request, raw_request)

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: SpeechRequest,
        raw_request: Request,
    ) -> Union[Response, ORJSONResponse]:
        """Handle non-streaming speech synthesis request.

        Collects all generated audio tokens (full 16-codebook RVQ codes
        from meta_info["audio_rvq_codes"]), decodes them to a waveform
        with the codec, then converts to the requested format.
        """
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        # Extract full RVQ codes from meta_info (set by model's customized_info)
        meta_info = ret.get("meta_info", {})
        rvq_codes = meta_info.get("audio_rvq_codes")

        if not rvq_codes:
            # Fallback: use output_ids (first codebook only)
            output_ids = ret.get("output_ids", [])
            if not output_ids:
                return self.create_error_response(
                    "No audio tokens generated.", status_code=500
                )
            rvq_codes = [[t] for t in output_ids]

        # Decode audio codes to waveform
        try:
            audio_waveform = self.codec.decode_rvq_codes(rvq_codes)
        except Exception as e:
            logger.error(f"Audio codec decode failed: {e}")
            return self.create_error_response(
                "Audio decoding failed.", status_code=500
            )

        # Convert to requested format
        try:
            audio_bytes = self.format_converter.convert(
                audio_waveform, self.sample_rate, request.response_format
            )
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return self.create_error_response(
                f"Format conversion to '{request.response_format}' failed.",
                status_code=500,
            )

        content_type = AUDIO_CONTENT_TYPES.get(
            request.response_format, "application/octet-stream"
        )
        return Response(content=audio_bytes, media_type=content_type)

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: SpeechRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming speech synthesis request.

        Streams audio chunks as they are generated.
        """
        content_type = AUDIO_CONTENT_TYPES.get(
            request.response_format, "application/octet-stream"
        )
        return StreamingResponse(
            self._generate_speech_stream(adapted_request, request, raw_request),
            media_type=content_type,
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_speech_stream(
        self,
        adapted_request: GenerateReqInput,
        request: SpeechRequest,
        raw_request: Request,
    ) -> AsyncGenerator[bytes, None]:
        """Generate streaming audio chunks.

        Collects full RVQ codes from meta_info["audio_rvq_codes"] in
        chunks, decodes each chunk with the codec, and yields audio bytes.
        """
        rvq_buffer = []  # List of [16 codes] per time step
        chunk_size = 50  # Decode every ~50 time steps

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                meta_info = content.get("meta_info", {})
                finish_reason = meta_info.get("finish_reason")
                rvq_codes = meta_info.get("audio_rvq_codes")

                if rvq_codes:
                    # rvq_codes is a list of 16 codes for this step
                    if isinstance(rvq_codes, list):
                        if isinstance(rvq_codes[0], list):
                            rvq_buffer.extend(rvq_codes)
                        else:
                            rvq_buffer.append(rvq_codes)

                # Decode chunk when buffer is large enough or generation is done
                if (len(rvq_buffer) >= chunk_size) or (
                    finish_reason and rvq_buffer
                ):
                    try:
                        audio_chunk = self.codec.decode_rvq_codes(rvq_buffer)
                        audio_bytes = self.format_converter.convert(
                            audio_chunk,
                            self.sample_rate,
                            request.response_format,
                        )
                        yield audio_bytes
                    except Exception as e:
                        logger.error(f"Streaming audio decode error: {e}")
                    rvq_buffer = []

        except ValueError as e:
            logger.error(f"Speech generation error: {e}")
