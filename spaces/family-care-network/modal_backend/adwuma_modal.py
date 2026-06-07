from __future__ import annotations

import base64
import io
import json
from typing import Any

import modal


APP_NAME = "adwuma-pa-inference"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "accelerate",
        "fastapi[standard]",
        "librosa",
        "numpy",
        "protobuf",
        "sentencepiece",
        "soundfile",
        "torch",
        "transformers>=4.40",
    )
)

app = modal.App(APP_NAME, image=image)

CPU_COST_LIMITS = dict(min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=5, timeout=150)
GPU_COST_LIMITS = dict(
    gpu="A10G",
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=10,
    timeout=150,
    startup_timeout=600,
)


@app.function(**CPU_COST_LIMITS)
@modal.fastapi_endpoint(method="GET", docs=True)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "cost_policy": {
            "min_containers": 0,
            "max_containers": 1,
            "gpu_scaledown_window_seconds": GPU_COST_LIMITS["scaledown_window"],
            "cron_default": "not enabled during development",
        },
        "endpoints": ["health", "translate", "transcribe", "analyze", "speak"],
    }


@app.function(**CPU_COST_LIMITS)
@modal.fastapi_endpoint(method="POST")
def translate(payload: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    text = (payload.get("text") or "").strip()
    source_language = payload.get("source_language") or "twi"
    if not text:
        return {"status": "needs_review", "error": "No text provided for translation."}
    if source_language == "eng":
        return {"status": "complete", "translated_text": text, "model_id": "identity"}

    model_id = "ninte/twi-en-nllb-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    src_lang = "twi_Latn"
    tgt_lang = "eng_Latn"
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    forced_bos_token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        forced_bos_token_id = token_id if isinstance(token_id, int) and token_id >= 0 else None
    generated = model.generate(**inputs, max_new_tokens=256, forced_bos_token_id=forced_bos_token_id)
    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return {
        "status": "complete",
        "translated_text": translated,
        "model_id": model_id,
        "source_language": src_lang,
        "target_language": tgt_lang,
    }


@app.function(**GPU_COST_LIMITS)
@modal.fastapi_endpoint(method="POST")
def transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    encoded = payload.get("audio_wav_base64")
    language = payload.get("language") or "twi"
    if not encoded:
        return {"status": "needs_review", "error": "No audio_wav_base64 payload provided."}

    wav_bytes = base64.b64decode(encoded)
    waveform, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        waveform = librosa.resample(np.asarray(waveform), orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    language_code = "aka" if language in {"twi", "fat", "aka"} else "eng"
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    processor.tokenizer.set_target_lang(language_code)
    model.load_adapter(language_code)

    import torch

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = logits.argmax(dim=-1)
    text = processor.batch_decode(ids)[0]
    confidence = float(logits.softmax(-1).max(-1).values.mean())
    return {
        "status": "complete",
        "text": text,
        "confidence": confidence,
        "low_confidence": confidence < 0.4 or len(text.strip()) < 3,
        "model_used": model_id,
        "language_code": language_code,
    }


@app.function(**GPU_COST_LIMITS)
@modal.fastapi_endpoint(method="POST")
def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    prompt = analysis_prompt(payload)
    messages = [
        {"role": "system", "content": "Return strict JSON only. Do not include markdown."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=384, temperature=0.2)
    output_ids = generated[0][len(inputs.input_ids[0]) :]
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "needs_review", "model_id": model_id, "error": "Qwen returned non-JSON output.", "raw": raw}
    return {"status": "complete", "model_id": model_id, "analysis": analysis}


@app.function(**GPU_COST_LIMITS)
@modal.fastapi_endpoint(method="POST")
def speak(payload: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer, VitsModel

    text = (payload.get("text") or "").strip()
    language = payload.get("language") or "twi"
    if not text:
        return {"status": "needs_review", "error": "No text provided for TTS."}
    model_id = "facebook/mms-tts-eng" if language == "eng" else "facebook/mms-tts-aka"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)

    import torch

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().cpu().numpy()

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, waveform, int(model.config.sampling_rate), format="WAV")
    return {
        "status": "complete",
        "model_used": model_id,
        "audio_wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "sample_rate": int(model.config.sampling_rate),
    }


def analysis_prompt(payload: dict[str, Any]) -> str:
    return f"""
Analyze this family elder check-in. Use the English translation for scoring, but preserve evidence from both original and English where useful.

Original text:
{payload.get("original_text", "")}

English translation:
{payload.get("english_translation", "")}

Member:
{json.dumps(payload.get("member", {}), ensure_ascii=False)}

Request:
{json.dumps(payload.get("request", {}), ensure_ascii=False)}

Recent history:
{json.dumps(payload.get("recent_history", []), ensure_ascii=False)}

Return exactly this JSON shape:
{{
  "summary": "one sentence",
  "concern_level": 0,
  "flags": [],
  "sentiment": "stable|worried|urgent",
  "evidence": [],
  "recommended_action": "normal|reminder|nudge_relative|call|urgent_review",
  "confidence": "low|medium|high"
}}

Concern scoring:
0-3 green/stable, 4-6 amber/check with relative, 7-10 red/immediate follow-up.
"""
