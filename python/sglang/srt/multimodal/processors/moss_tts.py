"""
Multimodal processor for MOSS-TTS-Realtime model.

Builds the multi-channel input expected by the TTS model:
  - System prompt with TTS engine description
  - Optional voice clone section with reference audio RVQ codes
  - Input text tokens for the model to speak

The processor tokenizes the full prompt and separates it into:
  - input_ids: the prompt tokens consumed during prefill
  - text_ids (in model_specific_data): the input text tokens that the
    model consumes one-per-step during autoregressive decode

The multi-channel format follows the original MossTTSRealtime inference:
  - Each position has 17 values: [text_token, audio_ch1, ..., audio_ch16]
  - Audio channels are AUDIO_PAD_TOKEN (1024) during prefill
  - The LAST prefill position has AUDIO_BOS_TOKEN (1025) in channel 1
    to signal the start of audio generation

Voice cloning (reference audio):
  - Reference audio is pre-encoded to RVQ codes by the serving layer
  - A context section with <|audio_pad|> placeholder tokens is appended
    to the system prompt
  - The placeholder positions are filled with actual RVQ codes in the
    audio channels (1:)
  - An <|im_start|>assistant prefix is appended after the context section
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.moss_tts_realtime import (
    AUDIO_BOS_TOKEN,
    AUDIO_PAD_TOKEN,
    NUM_RVQ_CHANNELS,
    REFERENCE_AUDIO_PAD_ID,
    MossTTSRealtime,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

logger = logging.getLogger(__name__)

# System prompt template used by MOSS-TTS-Realtime (must match original)
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are a highly expressive text-to-speech (TTS) engine developed by "
    "Mosi Intelligence. \n"
    "You possess natural language understanding, emotional modeling, and "
    "multi-style speech generation capabilities, allowing you to generate "
    "the corresponding speech based on the text given in the assistant."
    "<|im_end|>\n"
)

# Voice clone context template (matches original make_voice_clone_prompt)
VOICE_CLONE_CONTEXT = (
    "<|im_start|>context\n"
    "The assistant section should be synthesized using the following voice timbre:"
)

# Assistant turn prefix (used only with voice cloning)
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"


class MossTTSProcessor(BaseMultimodalProcessor):
    """Processor for MOSS-TTS-Realtime model.

    Constructs the multi-channel input format expected by the model.
    For TTS, the 'audio_data' in the request is optional pre-encoded
    reference audio RVQ codes for voice cloning.
    """

    models = [MossTTSRealtime]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Build the TTS input from text and optional reference audio.

        Args:
            audio_data: If provided, should be a list containing a numpy
                array of pre-encoded RVQ codes with shape (T, channels).
                The serving layer encodes raw audio bytes before passing
                them here.

        Returns a dict with:
          - input_ids: tokenized prompt (system prompt + text prefix)
          - mm_items: list with a single MultimodalDataItem that carries
            multi_channel_ids, text_ids and text_cursor in model_specific_data
        """
        tokenize = (
            self._tokenizer.encode
            if hasattr(self._tokenizer, "encode")
            else self._processor.encode
        )

        # Tokenize the input text (what the model should speak)
        text_ids = tokenize(input_text, add_special_tokens=False)

        # Check for pre-encoded reference audio RVQ codes
        ref_audio_tokens = None
        if audio_data and len(audio_data) > 0:
            ref = audio_data[0]
            if isinstance(ref, np.ndarray) and ref.ndim == 2:
                # Ensure shape is (T, channels)
                if ref.shape[0] == NUM_RVQ_CHANNELS and ref.shape[1] != NUM_RVQ_CHANNELS:
                    ref = ref.T
                if ref.shape[1] == NUM_RVQ_CHANNELS:
                    ref_audio_tokens = ref
                else:
                    logger.warning(
                        f"Reference audio tokens have unexpected shape {ref.shape}, ignoring"
                    )

        # Build system prompt multi-channel tokens
        if ref_audio_tokens is not None:
            sys_mc, system_ids = self._build_voice_clone_prompt(
                tokenize, ref_audio_tokens
            )
        else:
            sys_mc, system_ids = self._build_plain_prompt(tokenize)

        # Build text prefix: first N text tokens "prime" the model.
        # The original inference primes with ~12 tokens.
        num_prime = min(12, len(text_ids))
        prime_text_ids = text_ids[:num_prime]
        remaining_text_ids = text_ids[num_prime:]

        # Build text prefix multi-channel tokens
        prime_len = len(prime_text_ids)
        text_mc = np.full(
            (prime_len, 1 + NUM_RVQ_CHANNELS),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        text_mc[:, 0] = prime_text_ids

        # Signal start of audio generation: BOS in channel 1 at
        # the LAST text position only (matches original inference).
        text_mc[prime_len - 1, 1] = AUDIO_BOS_TOKEN

        # Concatenate system prompt + text prefix
        multi_channel = np.concatenate([sys_mc, text_mc], axis=0)
        prefill_text_ids = system_ids + prime_text_ids

        # Create a MultimodalDataItem that carries:
        # - multi_channel_ids: the multi-channel prefill tensor
        # - text_ids: remaining text tokens for the model to consume during decode
        # - text_cursor: starting position in text_ids (0)
        mm_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=np.zeros(1, dtype=np.float32),  # placeholder
            model_specific_data={
                "multi_channel_ids": multi_channel,
                "text_ids": remaining_text_ids,
                "text_cursor": 0,
            },
        )

        return {
            "input_ids": prefill_text_ids,
            "mm_items": [mm_item],
        }

    def _build_plain_prompt(self, tokenize):
        """Build system prompt without voice cloning."""
        system_ids = tokenize(SYSTEM_PROMPT, add_special_tokens=False)
        sys_len = len(system_ids)
        sys_mc = np.full(
            (sys_len, 1 + NUM_RVQ_CHANNELS),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        sys_mc[:, 0] = system_ids
        return sys_mc, system_ids

    def _build_voice_clone_prompt(self, tokenize, ref_audio_tokens):
        """Build system prompt with voice clone context section.

        Matches the original processor's make_ensemble + make_voice_clone_prompt:
        1. System prompt text + voice clone context with <|audio_pad|> placeholders
        2. Tokenize the combined text
        3. Find <|audio_pad|> token positions
        4. Fill audio channels (1:) at those positions with actual RVQ codes
        """
        num_ref_frames = ref_audio_tokens.shape[0]

        # Build the full system text with voice clone section
        # The <|audio_pad|> tokens act as positional placeholders
        audio_pad_str = "<|audio_pad|>" * num_ref_frames
        system_text = SYSTEM_PROMPT + VOICE_CLONE_CONTEXT + audio_pad_str

        system_ids = tokenize(system_text, add_special_tokens=False)
        sys_len = len(system_ids)

        # Create multi-channel array
        sys_mc = np.full(
            (sys_len, 1 + NUM_RVQ_CHANNELS),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        sys_mc[:, 0] = system_ids

        # Find <|audio_pad|> token positions and fill with RVQ codes
        system_ids_arr = np.array(system_ids)
        audio_pad_positions = np.where(system_ids_arr == REFERENCE_AUDIO_PAD_ID)[0]

        if len(audio_pad_positions) > 0:
            start_pos = audio_pad_positions[0]
            end_pos = audio_pad_positions[-1] + 1
            num_positions = end_pos - start_pos
            # Clip reference tokens to available positions
            num_to_fill = min(num_positions, num_ref_frames)
            sys_mc[start_pos : start_pos + num_to_fill, 1:] = ref_audio_tokens[:num_to_fill]
        else:
            logger.warning(
                "No <|audio_pad|> tokens found in system prompt; "
                "reference audio will not be used for conditioning."
            )

        # Append assistant prefix for voice clone mode
        assistant_ids = tokenize(ASSISTANT_PREFIX, add_special_tokens=False)
        asst_mc = np.full(
            (len(assistant_ids), 1 + NUM_RVQ_CHANNELS),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        asst_mc[:, 0] = assistant_ids

        sys_mc = np.concatenate([sys_mc, asst_mc], axis=0)
        system_ids = system_ids + assistant_ids

        return sys_mc, system_ids
