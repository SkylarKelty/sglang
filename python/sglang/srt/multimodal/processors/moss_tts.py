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

# System prompt template used by MOSS-TTS-Realtime
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are a highly expressive text-to-speech (TTS) engine that generates "
    "natural speech with subtle inflections: gentle pauses at commas, slight "
    "rises at questions, and calm, warm phrasing. Keep it conversational and "
    "human-like — not robotic or overly dramatic.\n"
    "<|im_end|>\n"
)

CONTEXT_TEMPLATE_WITH_AUDIO = (
    "<|im_start|>context\n"
    "The assistant section should be synthesized using the following voice "
    "timbre:{audio_pad_tokens}\n"
    "<|im_end|>\n"
)

CONTEXT_TEMPLATE_NO_AUDIO = (
    "<|im_start|>context\n"
    "The assistant section should be synthesized using the default voice.\n"
    "<|im_end|>\n"
)

ASSISTANT_PREFIX = "<|im_start|>assistant\n"


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
          - input_ids: tokenized prompt (system + context + assistant prefix)
          - mm_items: list with a single MultimodalDataItem that carries
            text_ids and text_cursor in model_specific_data, so the model
            can consume text tokens one-per-step during decode.
        """
        has_ref_audio = audio_data and len(audio_data) > 0

        # Build context section
        if has_ref_audio:
            num_ref_tokens = 500
            audio_pad_tokens = "<|audio_pad|>" * num_ref_tokens
            context = CONTEXT_TEMPLATE_WITH_AUDIO.format(
                audio_pad_tokens=audio_pad_tokens
            )
        else:
            context = CONTEXT_TEMPLATE_NO_AUDIO

        # Build the full prompt (system + context + assistant prefix)
        # The input_text is what the model should speak — it becomes the
        # text token queue consumed during decode, not part of prefill.
        prompt_prefix = SYSTEM_PROMPT + context + ASSISTANT_PREFIX

        # Tokenize the prompt prefix (consumed during prefill)
        if hasattr(self._tokenizer, "encode"):
            prefix_ids = self._tokenizer.encode(
                prompt_prefix, add_special_tokens=False
            )
            text_ids = self._tokenizer.encode(
                input_text, add_special_tokens=False
            )
        else:
            prefix_ids = self._processor.encode(
                prompt_prefix, add_special_tokens=False
            )
            text_ids = self._processor.encode(
                input_text, add_special_tokens=False
            )

        # Build multi-channel input_ids for prefill.
        # Each position has 17 values: [text_token, audio_ch1, ..., audio_ch16]
        # During prefill, audio channels are all AUDIO_PAD_TOKEN (no audio yet).
        # We prepend a few text tokens from text_ids to "prime" the model
        # (the MOSS-TTS inferencer primes with ~12 text tokens).
        num_prime = min(12, len(text_ids))
        prime_text_ids = text_ids[:num_prime]
        remaining_text_ids = text_ids[num_prime:]

        # The prefill input is: prefix_ids + prime_text_ids
        prefill_text_ids = prefix_ids + prime_text_ids

        # Build the multi-channel tensor for prefill:
        # shape = (seq_len, 1 + NUM_RVQ_CHANNELS) = (seq_len, 17)
        seq_len = len(prefill_text_ids)
        multi_channel = np.full(
            (seq_len, 1 + NUM_RVQ_CHANNELS),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        # Channel 0 = text tokens
        multi_channel[:, 0] = prefill_text_ids
        # Audio channels stay as AUDIO_PAD_TOKEN during prefill
        # Except: set audio BOS for the last (primed) positions
        for i in range(len(prefix_ids), seq_len):
            multi_channel[i, 1:] = AUDIO_BOS_TOKEN

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
        if has_ref_audio:
            from sglang.srt.utils import load_audio

            ref_audio = load_audio(audio_data[0])
            mm_item.model_specific_data["reference_audio"] = ref_audio

        return {
            "input_ids": prefill_text_ids,
            "mm_items": [mm_item],
        }
