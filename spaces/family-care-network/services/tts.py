from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from config.models import TTS_CONFIG


def tts_language_code(language: str) -> str:
    return TTS_CONFIG["language_map"].get(language, language)


def model_id_for_language(language: str) -> str:
    code = tts_language_code(language)
    return TTS_CONFIG["model_map"].get(code, TTS_CONFIG["model_id"])


@lru_cache(maxsize=3)
def _load_model(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoTokenizer, VitsModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    return tokenizer, model


def synthesize(text: str, language: str) -> dict[str, Any]:
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("TTS text is empty.")

    model_id = model_id_for_language(language)
    tokenizer, model = _load_model(model_id)

    import torch

    inputs = tokenizer(clean_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy().astype(np.float32)
    return {
        "audio": (int(model.config.sampling_rate), waveform),
        "model_used": model_id,
        "language_code": tts_language_code(language),
    }

