from __future__ import annotations

import base64
import io
import json
from typing import Any

import modal


APP_NAME = "ani-kese-inference"

web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

translate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentencepiece", "torch", "transformers>=4.40")
)

audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("accelerate", "librosa", "numpy", "protobuf", "sentencepiece", "soundfile", "torch", "transformers>=4.40")
)

llm_image = modal.Image.debian_slim(python_version="3.11").pip_install("accelerate", "torch", "transformers>=4.40")

app = modal.App(APP_NAME)

CPU_COST_LIMITS = dict(min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=5, timeout=420)
GPU_COST_LIMITS = dict(
    gpu="A10G",
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=10,
    timeout=420,
    startup_timeout=600,
)


@app.function(image=web_image, **CPU_COST_LIMITS)
@modal.asgi_app(label="api")
def api():
    from fastapi import FastAPI

    web = FastAPI(title="Ani Kɛse Modal Inference")

    @web.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": APP_NAME,
            "cost_policy": {
                "min_containers": 0,
                "max_containers": 1,
                "cpu_scaledown_window_seconds": CPU_COST_LIMITS["scaledown_window"],
                "gpu_scaledown_window_seconds": GPU_COST_LIMITS["scaledown_window"],
                "cron_default": "not enabled during development",
            },
            "endpoints": ["health", "translate", "transcribe", "analyze", "speak"],
        }

    @web.post("/translate")
    def translate(payload: dict[str, Any]) -> dict[str, Any]:
        return translate_impl.remote(payload)

    @web.post("/transcribe")
    def transcribe(payload: dict[str, Any]) -> dict[str, Any]:
        return transcribe_impl.remote(payload)

    @web.post("/analyze")
    def analyze(payload: dict[str, Any]) -> dict[str, Any]:
        return analyze_impl.remote(payload)

    @web.post("/speak")
    def speak(payload: dict[str, Any]) -> dict[str, Any]:
        return speak_impl.remote(payload)

    return web


@app.function(image=translate_image, **CPU_COST_LIMITS)
def translate_impl(payload: dict[str, Any]) -> dict[str, Any]:
    text = (payload.get("text") or "").strip()
    source_language = payload.get("source_language") or "twi"
    if not text:
        return {"status": "needs_review", "error": "No text provided for translation."}
    if source_language == "eng":
        return {"status": "complete", "translated_text": text, "model_id": "identity"}

    tokenizer, model = _translation_model()
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
        "model_id": "ninte/twi-en-nllb-v2",
        "source_language": src_lang,
        "target_language": tgt_lang,
    }


@app.function(image=audio_image, **GPU_COST_LIMITS)
def transcribe_impl(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

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

    processor, model = _asr_model()
    language_code = "aka" if language in {"twi", "fat", "aka"} else "eng"
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
        "model_used": "facebook/mms-1b-all",
        "language_code": language_code,
    }


@app.function(image=llm_image, **GPU_COST_LIMITS)
def analyze_impl(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    tokenizer, model = _qwen_model()
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    prompt = analysis_prompt(payload)
    messages = [
        {"role": "system", "content": "Return strict JSON only. Do not include markdown."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=384, temperature=0.2)
    output_ids = generated[0][len(inputs.input_ids[0]) :]
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "needs_review", "model_id": model_id, "error": "Qwen returned non-JSON output.", "raw": raw}
    return {"status": "complete", "model_id": model_id, "analysis": analysis}


@app.function(image=audio_image, **GPU_COST_LIMITS)
def speak_impl(payload: dict[str, Any]) -> dict[str, Any]:
    import soundfile as sf
    import torch

    text = (payload.get("text") or "").strip()
    language = payload.get("language") or "twi"
    if not text:
        return {"status": "needs_review", "error": "No text provided for TTS."}
    tokenizer, model, model_id = _tts_model(language)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, waveform, int(model.config.sampling_rate), format="WAV")
    return {
        "status": "complete",
        "model_used": model_id,
        "audio_wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "sample_rate": int(model.config.sampling_rate),
    }


def _translation_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    global _TRANSLATION_CACHE
    try:
        return _TRANSLATION_CACHE
    except NameError:
        model_id = "ninte/twi-en-nllb-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _TRANSLATION_CACHE = (tokenizer, model)
        return _TRANSLATION_CACHE


def _asr_model():
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    global _ASR_CACHE
    try:
        return _ASR_CACHE
    except NameError:
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        _ASR_CACHE = (processor, model)
        return _ASR_CACHE


def _qwen_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _QWEN_CACHE
    try:
        return _QWEN_CACHE
    except NameError:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        _QWEN_CACHE = (tokenizer, model)
        return _QWEN_CACHE


def _tts_model(language: str):
    from transformers import AutoTokenizer, VitsModel

    global _TTS_CACHE
    try:
        cache = _TTS_CACHE
    except NameError:
        cache = {}
        _TTS_CACHE = cache
    model_id = "facebook/mms-tts-eng" if language == "eng" else "facebook/mms-tts-aka"
    if model_id not in cache:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        cache[model_id] = (tokenizer, model, model_id)
    return cache[model_id]


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
