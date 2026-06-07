from __future__ import annotations

import gradio as gr

from config.models import ASR_CONFIG
from services.asr import transcribe


def run_asr(audio, language_label, model_label):
    if audio is None:
        return "No audio provided."
    language = ASR_CONFIG["supported_languages"][language_label]
    model_key = {
        "MMS-1B-all": "primary",
        "Adwuma Pa fine-tune": "fine_tuned",
        "GiftMark Akan Whisper": "fallback",
    }[model_label]
    result = transcribe(audio, language, model_key)
    if result["low_confidence"]:
        return (
            f"Model: {result['model_used']}\n"
            f"Confidence: {result['confidence']:.2f}\n"
            f"Transcript: {result['text']}\n\n"
            "Low confidence. Ask the speaker to type the message for the care workflow."
        )
    return f"Model: {result['model_used']}\nConfidence: {result['confidence']:.2f}\nTranscript: {result['text']}"


def build_eval():
    with gr.Blocks(title="Adwuma Pa ASR Eval") as demo:
        gr.Markdown(
            """
# Adwuma Pa ASR Eval

Record Twi, Fante, or Ghanaian English samples and compare small ASR models before choosing the production path.
            """
        )
        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Voice sample")
            with gr.Column():
                language = gr.Dropdown(list(ASR_CONFIG["supported_languages"].keys()), value="Twi", label="Language")
                model = gr.Dropdown(
                    ["MMS-1B-all", "Adwuma Pa fine-tune", "GiftMark Akan Whisper"],
                    value="MMS-1B-all",
                    label="Model",
                )
                button = gr.Button("Transcribe", variant="primary")
        output = gr.Textbox(label="Transcription", lines=8)
        button.click(run_asr, inputs=[audio, language, model], outputs=output)
    return demo


if __name__ == "__main__":
    build_eval().launch()

