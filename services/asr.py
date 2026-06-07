from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from config.models import ASR_CONFIG


def asr_language_code(language: str) -> str:
    return ASR_CONFIG["supported_languages"].get(ASR_CONFIG.get("display_languages", {}).get(language, language), language)


@lru_cache(maxsize=4)
def _load_model(model_key: str) -> tuple[Any, Any, str]:
    cfg = ASR_CONFIG[model_key]
    if cfg["type"] == "mms":
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        model = Wav2Vec2ForCTC.from_pretrained(cfg["model_id"])
        return processor, model, "mms"

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(cfg["model_id"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"])
    return processor, model, "whisper"


def normalize_audio(audio: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    sample_rate, waveform = audio
    waveform = waveform.astype(np.float32)
    if waveform.max(initial=0) > 1.5:
        waveform = waveform / 32768.0
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return sample_rate, waveform


def resample_if_needed(waveform: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    if sample_rate == target_rate:
        return waveform
    import librosa

    return librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_rate)


def transcribe(audio: tuple[int, np.ndarray] | None, language: str, model_key: str = "primary") -> dict[str, Any]:
    if audio is None:
        return {
            "text": "",
            "confidence": 0.0,
            "low_confidence": True,
            "model_used": ASR_CONFIG[model_key]["model_id"],
        }

    sample_rate, waveform = normalize_audio(audio)
    language_code = asr_language_code(language)
    processor, model, model_type = _load_model(model_key)

    if model_type == "mms":
        waveform = resample_if_needed(waveform, sample_rate, 16000)
        processor.tokenizer.set_target_lang(language_code)
        model.load_adapter(language_code)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        import torch

        with torch.no_grad():
            logits = model(**inputs).logits
        ids = logits.argmax(dim=-1)
        text = processor.batch_decode(ids)[0]
        confidence = float(logits.softmax(-1).max(-1).values.mean())
    else:
        waveform = resample_if_needed(waveform, sample_rate, 16000)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        import torch

        with torch.no_grad():
            ids = model.generate(inputs["input_features"])
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        confidence = 1.0 if text.strip() else 0.0

    return {
        "text": text,
        "confidence": confidence,
        "low_confidence": confidence < ASR_CONFIG["confidence_threshold"] or len(text.strip()) < 3,
        "model_used": ASR_CONFIG[model_key]["model_id"],
        "language_code": language_code,
    }
