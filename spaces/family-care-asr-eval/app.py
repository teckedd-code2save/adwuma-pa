from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

MODEL_REGISTRY = {
    "MMS-1B-all (recommended)": {
        "model_id": "facebook/mms-1b-all",
        "type": "mms",
        "parameter_count": "1B",
        "notes": "Native multilingual ASR with Twi target language and Fante/Akan coverage.",
    },
    "Ani Kɛse Akan Whisper fine-tune": {
        "model_id": "teckedd/whisper_small-waxal_akan-asr-v1",
        "type": "whisper",
        "parameter_count": "0.2B",
        "notes": "Published Akan fine-tune; useful for Well-Tuned badge validation.",
    },
    "GiftMark Akan Whisper": {
        "model_id": "GiftMark/akan-whisper-model",
        "type": "whisper",
        "parameter_count": "0.2B",
        "notes": "Community Akan fallback, Twi-oriented.",
    },
}

LANGUAGE_CODES = {
    "Twi": "aka",
    "Fante": "aka",
    "Ghanaian English": "eng",
}

VOTES_PATH = Path("community_votes.jsonl")


@lru_cache(maxsize=4)
def load_model(model_name: str) -> tuple[Any, Any, str]:
    cfg = MODEL_REGISTRY[model_name]
    if cfg["type"] == "mms":
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        model = Wav2Vec2ForCTC.from_pretrained(cfg["model_id"])
        return processor, model, "mms"

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(cfg["model_id"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"])
    return processor, model, "whisper"


def prepare_audio(audio: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    sample_rate, waveform = audio
    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.max(initial=0) > 1.5:
        waveform = waveform / 32768.0
    return sample_rate, waveform


def maybe_resample(waveform: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    if sample_rate == target_rate:
        return waveform
    import librosa

    return librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_rate)


def transcribe_one(audio: tuple[int, np.ndarray] | None, language: str, model_name: str) -> dict[str, Any]:
    if audio is None:
        return {
            "model": model_name,
            "text": "",
            "confidence": 0.0,
            "error": "No audio provided.",
        }

    sample_rate, waveform = prepare_audio(audio)
    processor, model, model_type = load_model(model_name)

    try:
        if model_type == "mms":
            waveform = maybe_resample(waveform, sample_rate, 16000)
            processor.tokenizer.set_target_lang(language)
            model.load_adapter(language)
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
            import torch

            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_ids = logits.argmax(dim=-1)
            text = processor.batch_decode(predicted_ids)[0]
            confidence = float(logits.softmax(-1).max(-1).values.mean())
        else:
            waveform = maybe_resample(waveform, sample_rate, 16000)
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
            import torch

            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"])
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            confidence = 1.0 if text.strip() else 0.0
    except Exception as exc:
        return {
            "model": model_name,
            "text": "",
            "confidence": 0.0,
            "error": str(exc),
        }

    return {
        "model": model_name,
        "text": text.strip(),
        "confidence": confidence,
        "error": "",
    }


def rough_wer(reference: str, prediction: str) -> str:
    ref = reference.lower().split()
    hyp = prediction.lower().split()
    if not ref:
        return "No reference text provided"
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return f"{dp[-1][-1] / len(ref):.1%}"


def format_result(result: dict[str, Any], reference: str) -> str:
    cfg = MODEL_REGISTRY[result["model"]]
    if result["error"]:
        return f"### {result['model']}\nError: {result['error']}\n"
    wer = rough_wer(reference, result["text"])
    low_conf = result["confidence"] < 0.4 or len(result["text"]) < 3
    fallback = "\nLow confidence: ask the speaker to type the message in the main app." if low_conf else ""
    return (
        f"### {result['model']}\n"
        f"Model ID: `{cfg['model_id']}`\n\n"
        f"Parameters: {cfg['parameter_count']}\n\n"
        f"Confidence: {result['confidence']:.2f}\n\n"
        f"Rough WER: {wer}\n\n"
        f"Transcript:\n{result['text']}{fallback}\n"
    )


def run(audio, language_label: str, model_name: str, reference: str) -> str:
    language = LANGUAGE_CODES[language_label]
    if model_name == "Compare all":
        names = list(MODEL_REGISTRY)
    else:
        names = [model_name]
    results = [transcribe_one(audio, language, name) for name in names]
    return "\n\n---\n\n".join(format_result(result, reference or "") for result in results)


def read_votes() -> list[dict[str, Any]]:
    if not VOTES_PATH.exists():
        return []
    votes = []
    for line in VOTES_PATH.read_text().splitlines():
        try:
            votes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return votes


def vote_summary_markdown() -> str:
    votes = read_votes()
    if not votes:
        return "No community votes yet. Compare the models, then vote for the output that best captured the meaning."

    model_counts = Counter(vote["model"] for vote in votes)
    language_counts = Counter(vote["language"] for vote in votes)
    rows = ["### Current Community Votes", "", "| Model | Votes |", "|---|---:|"]
    for model_name in MODEL_REGISTRY:
        rows.append(f"| {model_name} | {model_counts.get(model_name, 0)} |")
    rows.extend(["", "### Language Coverage", "", "| Language | Samples |", "|---|---:|"])
    for language_name in LANGUAGE_CODES:
        rows.append(f"| {language_name} | {language_counts.get(language_name, 0)} |")
    rows.append(f"\nTotal votes: {len(votes)}")
    return "\n".join(rows)


def recent_votes_markdown(limit: int = 6) -> str:
    votes = read_votes()
    if not votes:
        return "No comments yet."
    rows = ["### Recent Notes"]
    for vote in reversed(votes[-limit:]):
        note = vote.get("note") or "No note provided."
        rows.append(f"- {vote['language']} - **{vote['model']}**: {note}")
    return "\n".join(rows)


def record_vote(language: str, model_name: str, note: str) -> tuple[str, str, str]:
    vote = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": language,
        "model": model_name,
        "note": (note or "").strip()[:500],
    }
    with VOTES_PATH.open("a") as handle:
        handle.write(json.dumps(vote) + "\n")
    return "Vote saved. Thanks for helping evaluate Akan ASR.", vote_summary_markdown(), recent_votes_markdown()


with gr.Blocks(title="Ani Kɛse ASR Eval") as demo:
    gr.Markdown(
        """
# Ani Kɛse ASR Eval

First step for the hackathon build: test Twi and Fante speech recognition on real family recordings before wiring ASR into the main care app.
        """
    )

    with gr.Tabs():
        with gr.Tab("Compare ASR"):
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or upload audio")
                with gr.Column():
                    language = gr.Dropdown(list(LANGUAGE_CODES.keys()), value="Twi", label="Language")
                    model = gr.Dropdown(list(MODEL_REGISTRY.keys()) + ["Compare all"], value="Compare all", label="Model")
                    reference = gr.Textbox(
                        label="Optional exact reference text",
                        lines=3,
                        placeholder="Paste the exact words if you want rough WER. Leave blank for meaning-based comparison.",
                    )
                    button = gr.Button("Transcribe", variant="primary")

            output = gr.Markdown(label="Results")
            gr.Markdown(
                "WER only appears when exact reference text is provided. For this project, the practical test is whether the transcript preserves health or care signals."
            )

            with gr.Row():
                vote_language = gr.Dropdown(list(LANGUAGE_CODES.keys()), value="Twi", label="Vote language")
                vote_model = gr.Dropdown(list(MODEL_REGISTRY.keys()), value="MMS-1B-all (recommended)", label="Best model for this sample")
            vote_note = gr.Textbox(
                label="What made it best?",
                lines=3,
                placeholder="Example: It caught the word about walking pain, even though spelling was rough.",
            )
            vote_button = gr.Button("Save community vote", variant="primary")
            vote_status = gr.Textbox(label="Vote status", interactive=False)

        with gr.Tab("Community Results"):
            refresh_votes = gr.Button("Refresh votes")
            vote_summary = gr.Markdown(vote_summary_markdown())
            recent_votes = gr.Markdown(recent_votes_markdown())

    button.click(run, inputs=[audio_input, language, model, reference], outputs=output)
    language.change(lambda value: value, inputs=language, outputs=vote_language)
    vote_button.click(record_vote, inputs=[vote_language, vote_model, vote_note], outputs=[vote_status, vote_summary, recent_votes])
    refresh_votes.click(lambda: (vote_summary_markdown(), recent_votes_markdown()), outputs=[vote_summary, recent_votes])

demo.launch()
