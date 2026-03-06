"""
Multimodal processor for MOSS-TTS-Realtime model.

Builds the multi-channel input expected by the TTS model:
  - System prompt with TTS engine description
  - Optional reference audio codes at <|audio_pad|> positions
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


class MossTTSProcessor(BaseMultimodalProcessor):
    """Processor for MOSS-TTS-Realtime model.

    Constructs the multi-channel input format expected by the model.
    For TTS, the 'audio_data' in the request is optional reference audio
    for voice cloning, not the content to transcribe.
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

        Returns a dict with:
          - input_ids: tokenized prompt (system prompt + text prefix)
          - mm_items: list with a single MultimodalDataItem that carries
            multi_channel_ids, text_ids and text_cursor in model_specific_data
        """
        # Tokenize the input text (what the model should speak)
        if hasattr(self._tokenizer, "encode"):
            text_ids = self._tokenizer.encode(
                input_text, add_special_tokens=False
            )
        else:
            text_ids = self._processor.encode(
                input_text, add_special_tokens=False
            )

        # Build system prompt (matches original MossTTSRealtimeProcessor)
        system_prompt_text = SYSTEM_PROMPT

        if hasattr(self._tokenizer, "encode"):
            system_ids = self._tokenizer.encode(
                system_prompt_text, add_special_tokens=False
            )
        else:
            system_ids = self._processor.encode(
                system_prompt_text, add_special_tokens=False
            )

        # Build system prompt multi-channel tokens
        sys_len = len(system_ids)
        sys_mc = np.full(
            (sys_len, 1 + NUM_RVQ_CHANNELS),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        sys_mc[:, 0] = system_ids

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

        # If reference audio provided, load it and add to model_specific_data
        has_ref_audio = audio_data and len(audio_data) > 0
        if has_ref_audio:
            from sglang.srt.utils import load_audio

            ref_audio = load_audio(audio_data[0])
            mm_item.model_specific_data["reference_audio"] = ref_audio

        return {
            "input_ids": prefill_text_ids,
            "mm_items": [mm_item],
        }
