from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


@dataclass
class ModalResult:
    ok: bool
    status: str
    data: dict[str, Any]
    error: str | None = None
    latency_ms: int | None = None


def modal_base_url() -> str:
    return os.getenv("MODAL_API_BASE_URL", "").rstrip("/")


def modal_configured() -> bool:
    return bool(modal_base_url())


def _disabled_result(service: str) -> ModalResult:
    return ModalResult(
        ok=False,
        status="needs_review",
        data={"service": service},
        error="MODAL_API_BASE_URL is not configured; no model inference was run.",
    )


def _post(path: str, payload: dict[str, Any], timeout: int = 150) -> ModalResult:
    if not modal_configured():
        return _disabled_result(path.strip("/"))
    start = time.perf_counter()
    try:
        response = requests.post(f"{modal_base_url()}/{path.strip('/')}", json=payload, timeout=timeout)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        data = response.json()
        return ModalResult(ok=True, status=data.get("status", "complete"), data=data, latency_ms=latency_ms)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ModalResult(ok=False, status="needs_review", data={}, error=str(exc), latency_ms=latency_ms)


def modal_health() -> ModalResult:
    if not modal_configured():
        return _disabled_result("health")
    start = time.perf_counter()
    try:
        response = requests.get(f"{modal_base_url()}/health", timeout=30)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        return ModalResult(ok=True, status="complete", data=response.json(), latency_ms=latency_ms)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ModalResult(ok=False, status="needs_review", data={}, error=str(exc), latency_ms=latency_ms)


def transcribe_audio(audio: tuple[int, np.ndarray] | None, language: str, model_key: str = "primary") -> ModalResult:
    if audio is None:
        return ModalResult(ok=False, status="needs_review", data={}, error="No audio was provided.")
    sample_rate, waveform = audio
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    import io
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, waveform, int(sample_rate), format="WAV")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return _post(
        "transcribe",
        {
            "audio_wav_base64": encoded,
            "language": language,
            "model_key": model_key,
        },
    )


def translate_text(text: str, source_language: str) -> ModalResult:
    if source_language == "eng":
        return ModalResult(
            ok=True,
            status="complete",
            data={
                "translated_text": text,
                "model_id": "identity",
                "source_language": "eng",
                "target_language": "eng",
            },
        )
    return _post("translate", {"text": text, "source_language": source_language})


def analyze_concern(payload: dict[str, Any]) -> ModalResult:
    return _post("analyze", payload)


def synthesize_speech(text: str, language: str) -> ModalResult:
    return _post("speak", {"text": text, "language": language})
