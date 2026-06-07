# Adwuma Pa — Family Care Network
### Full Product Specification · Version 2
*"Adwuma Pa" means "Good Work" in Twi*

---

## 0. North Star

> A small AI that speaks Twi and Fante, quietly checks on your elders, routes concern to the nearest relative, and calls when silence goes too long — so no one slips away unnoticed.

Built for the Build Small Hackathon · June 5–15, 2026
Track: **Backyard AI**
Constraint: ≤ 32B parameters total · Real, no mocks.

---

## 1. The Problem

Extended Ghanaian families are geographically scattered. Elders don't broadcast when they are unwell. By the time news travels through informal channels, it is often too late to intervene. The coordinator (one person) cannot manually check on everyone. Group chats are noisy and don't create accountability.

**The specific failure:** A relative is sick. Nobody nearby knows. The coordinator is far away. The loop never closes.

---

## 2. Core Users

| Role | Who | How they interact |
|---|---|---|
| **Elder** | Dad, aunties, uncles | WhatsApp link → simple web page → text or voice reply |
| **First-party contact** | Nearby relative | WhatsApp nudge → goes to check in → reports back |
| **Coordinator** | You | Gradio dashboard — sees everyone, flags, open loops |

---

## 3. ASR — Honest Assessment & Decision

### ❌ Whisper Large v3 — DO NOT USE for Twi/Fante

Whisper Large v3 **does not officially support Twi**. Unofficial tests on Twi produce 45–50% Word Error Rate — completely unusable for a real app. Do not use Whisper as the primary ASR for any Akan language.

### ✅ Confirmed Working Options

#### Option A — `facebook/mms-1b-all` (PRIMARY RECOMMENDATION)
- **What it is:** Meta's Massively Multilingual Speech ASR model, 1B parameters, fine-tuned on 1,107 languages including Twi (`twi`) and Fante (covered under Akan)
- **Why:** Halves Whisper's WER on low-resource languages. Covers both Twi and Fante natively. No fine-tuning needed to start.
- **Language codes:** `twi` for Twi, `fat` for Fante
- **HuggingFace:** `facebook/mms-1b-all`
- **Expected WER:** ~30–40% out of the box on conversational Twi (acceptable for concern detection; not transcription verbatim)

#### Option B — `GiftMark/akan-whisper-model` (FALLBACK)
- **What it is:** Whisper-small fine-tuned specifically on Akan (Asante Twi + Akuapem Twi)
- **WER:** ~29% on test set — comparable to MMS but Twi-only, no Fante
- **HuggingFace:** `GiftMark/akan-whisper-model`
- **Use case:** Swap in if MMS underperforms on Asante Twi specifically

#### Option C — `teckedd/whisper_small-waxal_akan-asr-v1` (YOUR OWN FINE-TUNE ⭐)
- **What it is:** Whisper Small fine-tuned by you on the WaxalNLP Akan ASR dataset — 0.2B parameters
- **WER:** 34.28% on evaluation set
- **Training data:** WaxalNLP `aka_asr` dataset — different corpus from GiftMark's model, so may generalise differently to your family's speech
- **HuggingFace:** `teckedd/whisper_small-waxal_akan-asr-v1`
- **Why this matters:** Using your own published model means you already qualify for the **Well-Tuned bonus badge** — no additional fine-tuning run needed
- **Type:** `whisper` (use Whisper loading path in ASRService)
- Described in Section 6 below
- Only attempt if Days 3–4 reveal MMS accuracy is unacceptable for your family's specific dialect/speaking style

### Fante Reality Check
MMS covers Fante under the Akan umbrella but accuracy will be meaningfully lower than Twi. **Pragmatic approach for the hackathon:**
- Fante speakers get both text and voice input options
- Voice is processed via MMS with `fat` language code
- If transcription confidence is low (blank output or garbled), fall back gracefully: show "We couldn't catch that clearly — could you type it instead?"
- Document this honestly in the demo: "Twi-optimized; Fante improvement in progress"

---

## 4. ASR Evaluation Space (Build First)

**Before wiring ASR into the main app, validate it independently.**

### 4.1 What to Build

A standalone Gradio Space on HuggingFace: `build-small-hackathon/family-care-asr-eval`

**UI:**
- Microphone input (record directly in browser)
- File upload (upload a .wav or .mp3)
- Language selector dropdown: `Twi (twi)` | `Fante (fat)` | `Ghanaian English (eng)`
- Model selector dropdown: `MMS-1B-all` | `teckedd/whisper-waxal-akan (Your fine-tune ⭐)` | `GiftMark/akan-whisper` | `Compare all three`
- Transcription output panel
- If "Compare all three" selected: side-by-side output for all three models with WER estimate if reference text is provided

**Purpose:** Record your dad, an auntie, or yourself speaking natural Twi. See exactly what the model produces. Make the swap decision based on real data, not benchmarks.

### 4.2 Gradio ASR Eval Code

```python
import gradio as gr
from transformers import pipeline, AutoProcessor, AutoModelForCTC
import torch
import numpy as np

# ── Model registry — swap by changing this dict ──────────────────────────────
MODEL_REGISTRY = {
    "MMS-1B-all (Recommended)": {
        "model_id": "facebook/mms-1b-all",
        "type": "mms",
    },
    "teckedd/whisper-waxal-akan (Your fine-tune ⭐)": {
        "model_id": "teckedd/whisper_small-waxal_akan-asr-v1",
        "type": "whisper",
    },
    "GiftMark/akan-whisper (Community fine-tune)": {
        "model_id": "GiftMark/akan-whisper-model",
        "type": "whisper",
    },
}

LANGUAGE_CODES = {
    "Twi": "twi",
    "Fante": "fat",
    "Ghanaian English": "eng",
}

_loaded_models = {}

def load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    cfg = MODEL_REGISTRY[model_name]
    if cfg["type"] == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        model = Wav2Vec2ForCTC.from_pretrained(cfg["model_id"])
        _loaded_models[model_name] = (processor, model, "mms")
    else:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(cfg["model_id"])
        model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"])
        _loaded_models[model_name] = (processor, model, "whisper")
    return _loaded_models[model_name]

def transcribe(audio, language: str, model_name: str):
    if audio is None:
        return "No audio provided."
    sr, waveform = audio
    waveform = waveform.astype(np.float32) / 32768.0  # normalize int16
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # stereo → mono

    lang_code = LANGUAGE_CODES[language]
    processor, model, model_type = load_model(model_name)

    if model_type == "mms":
        processor.tokenizer.set_target_lang(lang_code)
        model.load_adapter(lang_code)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    else:
        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"])
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def compare_models(audio, language: str):
    results = {}
    for name in MODEL_REGISTRY:
        try:
            results[name] = transcribe(audio, language, name)
        except Exception as e:
            results[name] = f"Error: {e}"
    output = ""
    for name, text in results.items():
        output += f"### {name}\n{text}\n\n"
    return output

with gr.Blocks(title="Adwuma Pa — ASR Evaluation") as demo:
    gr.Markdown("## 🎙️ Adwuma Pa — ASR Model Evaluation\nTest Twi and Fante speech recognition before wiring into the main app.")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Audio")
        with gr.Column():
            language_dd = gr.Dropdown(choices=list(LANGUAGE_CODES.keys()), value="Twi", label="Language")
            model_dd = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()) + ["Compare All"],
                value="MMS-1B-all (Recommended)",
                label="Model"
            )
            run_btn = gr.Button("Transcribe", variant="primary")

    output_box = gr.Textbox(label="Transcription Output", lines=6)

    def run(audio, language, model_name):
        if model_name == "Compare All":
            return compare_models(audio, language)
        return transcribe(audio, language, model_name)

    run_btn.click(run, inputs=[audio_input, language_dd, model_dd], outputs=output_box)

demo.launch()
```

**Deploy this to HF Spaces first. Run real Twi recordings through it. Pick your model. Only then move to the main app.**

---

## 5. Model Swap Architecture

The main app is built so ASR, LLM, and TTS models can each be swapped with a config change — no code rewrite.

### 5.1 Central Config File: `config/models.py`

```python
# ─────────────────────────────────────────────────
# ADWUMA PA — MODEL CONFIGURATION
# Change model IDs here. Nothing else needs to change.
# ─────────────────────────────────────────────────

ASR_CONFIG = {
    "model_id": "facebook/mms-1b-all",   # swap options:
                                           # "teckedd/whisper_small-waxal_akan-asr-v1" (your fine-tune, Well-Tuned badge ⭐)
                                           # "GiftMark/akan-whisper-model" (community fine-tune)
    "type": "mms",                         # "mms" | "whisper"
    "default_language": "twi",
    "supported_languages": {
        "Twi": "twi",
        "Fante": "fat",
        "English": "eng",
    },
    "confidence_threshold": 0.4,           # below this → ask user to retype
}

LLM_CONFIG = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",  # swap to any instruct model
    "max_new_tokens": 512,
    "temperature": 0.3,
}

TTS_CONFIG = {
    "model_id": "facebook/mms-tts",
    "language_map": {
        "twi": "twi",
        "fat": "fat",
        "eng": "eng",
    },
}
```

### 5.2 ASR Service: `services/asr.py`

```python
from config.models import ASR_CONFIG
from transformers import AutoProcessor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
import torch, numpy as np

class ASRService:
    def __init__(self):
        self.cfg = ASR_CONFIG
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if self.cfg["type"] == "mms":
            self._processor = AutoProcessor.from_pretrained(self.cfg["model_id"])
            self._model = Wav2Vec2ForCTC.from_pretrained(self.cfg["model_id"])
        elif self.cfg["type"] == "whisper":
            self._processor = WhisperProcessor.from_pretrained(self.cfg["model_id"])
            self._model = WhisperForConditionalGeneration.from_pretrained(self.cfg["model_id"])

    def transcribe(self, waveform: np.ndarray, sample_rate: int, language: str = None) -> dict:
        self._load()
        lang = language or self.cfg["default_language"]
        waveform = waveform.astype(np.float32) / 32768.0
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if self.cfg["type"] == "mms":
            self._processor.tokenizer.set_target_lang(lang)
            self._model.load_adapter(lang)
            inputs = self._processor(waveform, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(ids)[0]
            confidence = float(logits.softmax(-1).max(-1).values.mean())
        else:
            inputs = self._processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                ids = self._model.generate(inputs["input_features"])
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            confidence = 1.0  # Whisper doesn't expose token confidence easily

        low_confidence = confidence < self.cfg["confidence_threshold"] or len(text.strip()) < 3
        return {
            "text": text,
            "language": lang,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "model_used": self.cfg["model_id"],
        }

asr_service = ASRService()  # singleton
```

---

## 6. Fine-Tuning Plan (Optional — Only If MMS WER Is Unacceptable)

### 6.1 Datasets to Use

| Dataset | Source | Size | Access | Notes |
|---|---|---|---|---|
| `ghananlpcommunity/twi-speech-text-multispeaker-16k` | HuggingFace | 21,138 pairs | Free, direct HF load | **Primary dataset. Multi-speaker, best quality.** |
| Mozilla Common Voice Twi (cv-corpus-25.0) | [Mozilla Data Collective](https://datacollective.mozillafoundation.org) | 341 clips / 0.41h validated | **No longer on HF. Must download from Mozilla Data Collective directly.** Sign up at datacollective.mozillafoundation.org, search "Common Voice Twi", download .tar.gz manually. | Small but validated |
| `GiftMark/akan-whisper-model` training data | HuggingFace (`Lagyamfi/akan_audio_processed`) | ~few hundred samples | Free HF load | Akan mixed dialect |

**⚠️ Common Voice Note:** As of October 2025, ALL Mozilla Common Voice datasets were removed from HuggingFace and are now exclusively hosted at [datacollective.mozillafoundation.org](https://datacollective.mozillafoundation.org). You must create a free account there and download manually. The Twi corpus (cv-corpus-25.0) is small (341 clips, 0.29h validated) — useful as a validation set, not for training alone.

### 6.2 Fine-Tuning Script: `finetune/finetune_mms_twi.py`

```python
"""
Fine-tune facebook/mms-1b-all on Twi data and push to HuggingFace Hub.

Usage:
  modal run finetune/finetune_mms_twi.py

Requirements:
  - HF_TOKEN env var set (for pushing to hub)
  - Modal set up with A10G GPU
"""

import modal

app = modal.App("adwuma-pa-finetune")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.40",
        "datasets",
        "accelerate",
        "evaluate",
        "jiwer",          # WER calculation
        "soundfile",
        "librosa",
        "huggingface_hub",
    )
)

@app.function(
    gpu="A10G",
    image=image,
    timeout=60 * 60 * 6,   # 6 hour max
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def finetune():
    import os
    import torch
    import numpy as np
    from datasets import load_dataset, Audio, DatasetDict
    from transformers import (
        AutoProcessor,
        Wav2Vec2ForCTC,
        TrainingArguments,
        Trainer,
    )
    from dataclasses import dataclass
    from typing import Dict, List, Union
    import evaluate

    HF_TOKEN = os.environ["HF_TOKEN"]
    BASE_MODEL = "facebook/mms-1b-all"
    TARGET_LANG = "twi"
    OUTPUT_REPO = "your-hf-username/mms-twi-adwuma-pa"  # ← change this

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    ds_main = load_dataset(
        "ghananlpcommunity/twi-speech-text-multispeaker-16k",
        trust_remote_code=True,
    )
    # Cast audio column to 16kHz
    ds_main = ds_main.cast_column("audio", Audio(sampling_rate=16000))

    # Split if no pre-existing split
    if "train" not in ds_main:
        ds_split = ds_main["train"].train_test_split(test_size=0.1, seed=42)
        ds = DatasetDict({"train": ds_split["train"], "test": ds_split["test"]})
    else:
        ds = ds_main

    print(f"Train: {len(ds['train'])} samples | Test: {len(ds['test'])} samples")

    # ── Load processor & model ────────────────────────────────────────────────
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_target_lang(TARGET_LANG)

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        ignore_mismatched_sizes=True,
        target_lang=TARGET_LANG,
    )
    # Freeze feature extractor, only train adapter + classifier
    model.freeze_base_model()
    model.load_adapter(TARGET_LANG)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def preprocess(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding=True,
        )
        with processor.as_target_processor():
            labels = processor(batch["text"], return_tensors="pt", padding=True)
        batch["input_values"] = inputs.input_values[0]
        batch["labels"] = labels.input_ids[0]
        return batch

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names, num_proc=4)

    # ── Data collator ─────────────────────────────────────────────────────────
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: AutoProcessor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": f["input_values"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    collator = DataCollatorCTCWithPadding(processor=processor)

    # ── Metrics ───────────────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="./mms-twi-checkpoints",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        fp16=True,
        learning_rate=1e-4,
        warmup_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        push_to_hub=True,
        hub_model_id=OUTPUT_REPO,
        hub_token=HF_TOKEN,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Pushing to HuggingFace Hub...")
    trainer.push_to_hub(
        commit_message="Fine-tuned MMS on Twi — Adwuma Pa",
        language="twi",
        tags=["twi", "ghana", "akan", "asr", "adwuma-pa"],
    )
    processor.push_to_hub(OUTPUT_REPO, token=HF_TOKEN)

    print(f"✅ Model pushed to https://huggingface.co/{OUTPUT_REPO}")


@app.local_entrypoint()
def main():
    finetune.remote()
```

**After fine-tuning:** Update `config/models.py` to point to your new model:
```python
ASR_CONFIG = {
    "model_id": "your-hf-username/mms-twi-adwuma-pa",
    "type": "mms",
    ...
}
```
Zero other code changes needed.

---

## 7. Updated System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  SCHEDULER (Modal Cron)              │
│  Fires check-in links · Monitors silence · Triggers  │
│  escalation · Triggers calls                         │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   CHECK-IN WEB PAGE     │
        │   (HF Space · FastAPI)  │
        │   Text + Voice input    │
        │   Twi / Fante / English │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────────────────────────┐
        │   AI PROCESSING LAYER  (Modal A10G)          │
        │                                              │
        │   ASR: config/models.py → ASRService         │
        │   ├── PRIMARY:  facebook/mms-1b-all          │
        │   └── FALLBACK: GiftMark/akan-whisper-model  │
        │                                              │
        │   LLM: Qwen2.5-7B-Instruct                  │
        │   └── concern scoring · summarization        │
        │                                              │
        │   TTS: facebook/mms-tts (twi / fat)         │
        │   └── call speech generation                 │
        └────────────┬────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   RELAY ENGINE          │
        │   Find nearest contact  │
        │   Send WhatsApp nudge   │
        │   Collect field report  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   VOICE CALL ENGINE     │
        │   Twilio + Modal        │
        │   TTS (Twi/Fante)       │
        │   ASR transcription     │
        │   Summary → dashboard   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   COORDINATOR DASHBOARD │
        │   Gradio · HF Space     │
        │   Family map · Alerts   │
        │   Check-in history      │
        │   Loop closure status   │
        └─────────────────────────┘
```

---

## 8. Full Feature Spec

### 8.1 Family Registry
- Coordinator adds family members via dashboard
- Each member record: name, phone, WhatsApp number, location (city/region), language preference (Twi / Fante / English), first-party contacts
- Stored in SQLite on HF Space persistent volume
- Coordinator can edit, deactivate, reactivate members

### 8.2 Check-in Link System
- Each elder gets a unique persistent URL: `your-space.hf.space/checkin/uncle-kwame`
- Opens a mobile-friendly page in their language
- Warm greeting in Twi/Fante + their name
- Two input options: text box OR voice recording (Web Audio API)
- On submit: saved, timestamped, processing triggered
- If voice transcription confidence is low → graceful fallback: prompt to retype
- Page confirms receipt: *"Meda wo ase, Opanyin Kwame"*

### 8.3 AI Processing Pipeline
Each check-in goes through:

**Step 1 — ASR (if voice)**
→ `ASRService.transcribe()` via `config/models.py`
→ Runs on Modal A10G
→ Returns text + confidence score
→ Low confidence → fallback message to user

**Step 2 — Concern Scoring (Qwen2.5-7B)**
→ Receives transcript (or typed text)
→ System prompt: translate internally → extract health/emotional/social/financial signals
→ Returns structured JSON:
```json
{
  "summary": "Elder mentioned tiredness and hasn't left the house in a week.",
  "concern_level": 6,
  "flags": ["fatigue", "social_isolation"],
  "language_detected": "twi",
  "sentiment": "neutral-negative"
}
```

**Concern thresholds:**
- 0–3: Green. Log and move on.
- 4–6: Amber. Notify coordinator. Queue first-party nudge.
- 7–10: Red. Immediate coordinator alert + first-party nudge + escalate to call within 24h.

### 8.4 Silence Detection & Escalation Schedule
Modal cron job runs every 6 hours:
```
For each active elder:
  last_checkin < 7 days AND concern < 4  → do nothing
  last_checkin >= 7 days                 → send reminder WhatsApp link
  last_checkin >= 10 days                → amber alert to coordinator
  last_checkin >= 14 days OR concern ≥ 7 → trigger voice call
  no answer after call                   → red alert + nudge first-party contacts
```
Coordinator can adjust thresholds per person.

### 8.5 First-Party Contact Relay
- System finds assigned first-party contacts for flagged elder
- WhatsApp message via Twilio:
  > *"Hi Ama, we haven't heard from Uncle Kofi in Obuasi in a while. Could you check on him? Reply here or use this link: [url]"*
- First-party uses same check-in page (different prompt — what did they observe)
- Report summarized, attached to elder's record
- Coordinator sees: elder status + field report + who checked in + when

### 8.6 Voice Call Engine (Killer Feature)
**Outbound call flow:**
1. Twilio dials elder's phone
2. On answer: TTS speaks greeting in Twi/Fante
3. Elder speaks — Twilio records
4. Audio → Modal → MMS ASR transcribes
5. Qwen processes → concern scored
6. If concern high: follow-up question spoken
7. Warm close in Twi/Fante
8. Transcript + summary → dashboard
9. Coordinator notified immediately

### 8.7 Coordinator Dashboard (Gradio)
**Views:**
- **Family Map** — all members, color-coded (green/amber/red), last check-in, last summary
- **Alert Feed** — chronological flags, unresolved loops
- **Member Detail** — full history, call transcripts, field reports, concern trend
- **Loop Tracker** — open items: who was nudged, who responded, what they reported
- **Settings** — add/edit members, thresholds, toggle call per person

**Manual actions:**
- Send check-in link now
- Trigger a call now
- Mark loop resolved
- Add a note to member record

---

## 9. Space Setup, GitHub Connection & Agent Management

### 9.1 Space Creation — Exact Selections

Create two Spaces under the `build-small-hackathon` org. Use English names for discoverability — cultural identity lives in the description and the app itself.

**Space 1 — ASR Evaluation (build first, free hardware)**

| Field | Value |
|---|---|
| Owner | `build-small-hackathon` |
| Space name | `family-care-asr-eval` |
| Description | `Ghanaian language speech recognition eval — Twi & Fante ASR model comparison for family wellness app` |
| SDK | Gradio |
| Template | Blank |
| Hardware | Free (CPU) |
| Storage Bucket | None |
| Visibility | Public |
| License | Apache 2.0 |
| Dev Mode | Off |

**Space 2 — Main App**

| Field | Value |
|---|---|
| Owner | `build-small-hackathon` |
| Space name | `family-care-network` |
| Description | `AI-powered family wellness network for Ghanaian elders — multilingual check-ins in Twi & Fante, silence detection, and automated care relay` |
| SDK | Gradio |
| Template | Blank |
| Hardware | ZeroGPU (switch after setup — free on-demand GPU) |
| Storage Bucket | Create `family-care-storage` and attach at creation |
| Visibility | Public |
| License | Apache 2.0 |
| Dev Mode | Off |

> Do not enable Dev Mode — it requires Team & Enterprise. GitHub sync (below) replaces it cleanly.

---

### 9.2 GitHub → HuggingFace Space Sync

All code lives in GitHub. HF Spaces auto-update on every push to `main`. You get proper git history, PRs, and collaboration — and the Space always reflects your latest code.

**Step 1 — Create GitHub repo**
```bash
gh repo create adwuma-pa --public --clone
cd adwuma-pa
```

**Step 2 — Repo structure**
```
adwuma-pa/
├── app.py                       # Main Gradio dashboard entry point
├── asr_eval.py                  # ASR eval Space entry point
├── requirements.txt
├── config/
│   └── models.py                # Model swap config (Section 5.1)
├── services/
│   ├── asr.py                   # ASRService (Section 5.2)
│   ├── llm.py                   # Qwen concern scoring
│   └── tts.py                   # MMS TTS
├── scheduler/
│   └── cron.py                  # Modal cron jobs
├── finetune/
│   └── finetune_mms_twi.py      # Fine-tuning script (Section 6.2)
├── db/
│   ├── schema.sql               # SQLite schema (Section 13)
│   └── database.py              # DB helpers
├── pages/
│   └── checkin.py               # FastAPI check-in page
└── .github/
    └── workflows/
        └── sync_to_hf.yml       # Auto-sync to HF Spaces on push
```

**Step 3 — GitHub Action: `.github/workflows/sync_to_hf.yml`**
```yaml
name: Sync to HuggingFace Spaces

on:
  push:
    branches: [main]

jobs:
  sync-asr-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to ASR Eval Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config user.email "ci@family-care.ai"
          git config user.name "Family Care CI"
          git remote add hf_eval https://user:${HF_TOKEN}@huggingface.co/spaces/build-small-hackathon/family-care-asr-eval
          git push hf_eval main --force

  sync-main-app:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to Main App Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config user.email "ci@family-care.ai"
          git config user.name "Family Care CI"
          git remote add hf_main https://user:${HF_TOKEN}@huggingface.co/spaces/build-small-hackathon/family-care-network
          git push hf_main main --force
```

**Step 4 — Add HF_TOKEN to GitHub secrets**
- GitHub repo → Settings → Secrets and variables → Actions → New repository secret
- Name: `HF_TOKEN` | Value: your HF write token (HF → Settings → Access Tokens → New token, role: Write)

From here: every `git push origin main` automatically deploys both Spaces.

---

### 9.3 HF Agent CLI — Install & Space Management

The HF CLI Skill lets any AI agent (Claude Code, etc.) create, configure, and manage Spaces without a browser. Install once per machine.

**Installation**
```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
# Paste your HF write token when prompted
# Store token: yes
```

**Verify membership**
```bash
huggingface-cli whoami
# Confirms you're authenticated and can act on build-small-hackathon org
```

**Agent-usable commands for this project**

```bash
# List all Spaces in the org
huggingface-cli repo list --repo-type space --organization build-small-hackathon

# Create a new Space mid-hackathon if needed
huggingface-cli repo create family-care-new-feature \
  --repo-type space \
  --organization build-small-hackathon \
  --space-sdk gradio

# Upload a single updated file directly (bypass GitHub sync)
huggingface-cli upload build-small-hackathon/family-care-network \
  app.py app.py --repo-type space

# Upload entire local folder
huggingface-cli upload build-small-hackathon/family-care-network \
  . . --repo-type space

# Check Space status
huggingface-cli repo info build-small-hackathon/family-care-network --repo-type space
```

**Python SDK commands (for hardware and secrets — CLI doesn't cover these)**

```python
from huggingface_hub import HfApi
api = HfApi()

# Switch Space to ZeroGPU (do this after initial setup)
api.request_space_hardware(
    repo_id="build-small-hackathon/family-care-network",
    hardware="zero-a10g"
)

# Add all required secrets before first real run
secrets = {
    "TWILIO_SID":            "your_twilio_sid",
    "TWILIO_AUTH_TOKEN":     "your_twilio_auth_token",
    "TWILIO_WHATSAPP_FROM":  "whatsapp:+14155238886",  # Twilio sandbox number
    "MODAL_TOKEN_ID":        "your_modal_token_id",
    "MODAL_TOKEN_SECRET":    "your_modal_token_secret",
    "HF_TOKEN":              "your_hf_write_token",
}
for key, val in secrets.items():
    api.add_space_secret("build-small-hackathon/family-care-network", key, val)
    print(f"Added secret: {key}")

# Restart Space after config or secret changes
api.restart_space("build-small-hackathon/family-care-network")

# Delete a Space (e.g. clean up eval space after submission)
api.delete_repo(
    repo_id="build-small-hackathon/family-care-asr-eval",
    repo_type="space"
)
```

**Attach storage bucket via SDK (if not done at creation)**
```python
api.add_space_secret(
    "build-small-hackathon/family-care-network",
    "BUCKET_NAME",
    "family-care-storage"
)
# Then mount the bucket in Space settings UI, or via:
api.request_space_storage(
    repo_id="build-small-hackathon/family-care-network",
    storage="small"  # "small" = 50GB, "medium" = 150GB, "large" = 1TB
)
```

---

## 10. Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| ASR (primary) | `facebook/mms-1b-all` | Native Twi + Fante, swappable |
| ASR (your fine-tune ⭐) | `teckedd/whisper_small-waxal_akan-asr-v1` | 34.28% WER, WaxalNLP data, qualifies for Well-Tuned badge |
| ASR (community fallback) | `GiftMark/akan-whisper-model` | Twi-only, ~29% WER |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | Multilingual understanding |
| TTS | `facebook/mms-tts` (twi / fat) | Native Twi & Fante speech |
| Voice calls | Twilio Programmable Voice | Real calls |
| WhatsApp | Twilio WhatsApp API sandbox | Free for testing |
| Scheduler | Modal Cron | Per-second billing, no idle cost |
| GPU inference | Modal A10G | $0.000306/sec |
| Check-in pages | FastAPI on HF Space | Lightweight |
| Dashboard | Gradio on HF Space | Hackathon requirement |
| Database | SQLite + HF Space persistent storage | Simple, free |

**Total parameter count: ~9.5B** (well under 32B limit)

---

## 10. Datasets

| Dataset | HF ID | Size | Access | Use |
|---|---|---|---|---|
| Twi speech-text pairs | `ghananlpcommunity/twi-speech-text-multispeaker-16k` | 21,138 pairs | Free, direct HF | **Primary fine-tuning data** |
| Mozilla Common Voice Twi | cv-corpus-25.0 (Twi) | 341 clips / 0.29h validated | ⚠️ **NOT on HuggingFace anymore.** Download from [datacollective.mozillafoundation.org](https://datacollective.mozillafoundation.org) — free account required | Validation set only (too small to train on) |
| Akan audio processed | `Lagyamfi/akan_audio_processed` | Small | Free HF | Supplementary Akan |

**Common Voice access steps (manual):**
1. Create account at `datacollective.mozillafoundation.org`
2. Search "Common Voice Scripted Speech" → filter by Twi
3. Download `cv-corpus-25.0-2026-03-09-tw.tar.gz`
4. Extract and convert to 16kHz WAV for training use
5. Use only as a **validation/test set** given its tiny size (341 clips)

---

## 11. Budget Plan

| Resource | Use | Est. Spend |
|---|---|---|
| Modal A10G ($250 credits) | ASR + LLM inference, cron jobs, call processing, fine-tune run if needed | **~$30–50 total** |
| HF Space ($20 credits) | ZeroGPU for dashboard, Space hosting | ~$5–10 |
| Twilio (free trial ~$15) | Calls + WhatsApp sandbox | $0 (free trial covers demo) |
| **Total** | | **Well under $80, likely under $50** |

**A10G math:** $0.000306/sec × 17 sec per check-in = $0.005 per check-in. $50 covers 10,000 check-ins. You're safe for all development, testing, and demo.

---

## 12. Build Plan — Day by Day

### ✅ Day 1 (June 5) — Foundation
- [x] Repo created, HF Space shell up
- [x] Modal account ready
- [x] Twilio account + WhatsApp sandbox

### Day 2 (June 6) — TODAY
- [ ] SQLite schema created (see Section 13)
- [ ] Gradio dashboard shell: member list, add member form
- [ ] **Deploy ASR Eval Space** (`adwuma-pa/asr-eval`) — Section 4
- [ ] Record 5–10 voice samples from yourself in Twi and run through eval space
- [ ] Pick ASR model based on real results

### Days 3–4 (June 7–8) — Core Pipeline
- [ ] Check-in web page (FastAPI, mobile-friendly, Twi/Fante/English)
- [ ] Wire chosen ASR model on Modal
- [ ] Wire Qwen concern scoring on Modal
- [ ] End-to-end: voice note → transcription → concern score → DB

### Days 5–6 (June 9–10) — Notifications & Relay
- [ ] WhatsApp link sending via Twilio
- [ ] Silence detection cron job on Modal
- [ ] First-party contact relay — nudge + field report page
- [ ] Dashboard: live status, alert feed

### Days 7–8 (June 11–12) — Voice Call Engine
- [ ] Outbound Twilio call
- [ ] MMS TTS for Twi/Fante greeting + close
- [ ] Call transcript → dashboard
- [ ] Manual call trigger from dashboard

### Days 9–10 (June 13–14) — Real Testing
- [ ] Live test with dad, aunties, uncles
- [ ] Tune concern scoring thresholds on real responses
- [ ] Tune Twi/Fante message phrasing
- [ ] Loop tracker + resolved states
- [ ] Polish dashboard UI

### Day 10 + buffer (June 14–15) — Submit
- [ ] Demo video (60 seconds — see script below)
- [ ] Social post
- [ ] HF Space submission

---

## 13. SQLite Schema

```sql
CREATE TABLE members (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  phone TEXT NOT NULL,
  whatsapp TEXT NOT NULL,
  location_city TEXT,
  location_region TEXT,
  language TEXT DEFAULT 'twi',       -- twi | fat | eng
  checkin_url_token TEXT UNIQUE,
  active INTEGER DEFAULT 1,
  escalation_days_amber INTEGER DEFAULT 7,
  escalation_days_red INTEGER DEFAULT 14,
  call_enabled INTEGER DEFAULT 1,
  created_at TEXT
);

CREATE TABLE first_party_contacts (
  id TEXT PRIMARY KEY,
  elder_id TEXT REFERENCES members(id),
  contact_id TEXT REFERENCES members(id),
  priority INTEGER DEFAULT 1
);

CREATE TABLE checkins (
  id TEXT PRIMARY KEY,
  member_id TEXT REFERENCES members(id),
  submitted_at TEXT,
  input_type TEXT,                   -- text | voice
  raw_input TEXT,
  transcript TEXT,
  asr_model_used TEXT,               -- tracks which model was used
  asr_confidence REAL,
  summary TEXT,
  concern_level INTEGER,
  flags TEXT,                        -- JSON array
  language_detected TEXT,
  source TEXT                        -- self | field_report
);

CREATE TABLE alerts (
  id TEXT PRIMARY KEY,
  member_id TEXT REFERENCES members(id),
  alert_type TEXT,                   -- silence | concern | call_no_answer
  created_at TEXT,
  resolved INTEGER DEFAULT 0,
  resolved_at TEXT,
  resolved_by TEXT,
  notes TEXT
);

CREATE TABLE calls (
  id TEXT PRIMARY KEY,
  member_id TEXT REFERENCES members(id),
  initiated_at TEXT,
  duration_seconds INTEGER,
  transcript TEXT,
  asr_model_used TEXT,
  summary TEXT,
  concern_level INTEGER,
  twilio_call_sid TEXT,
  status TEXT                        -- initiated | completed | no_answer | failed
);

CREATE TABLE nudges (
  id TEXT PRIMARY KEY,
  elder_id TEXT REFERENCES members(id),
  contact_id TEXT REFERENCES members(id),
  sent_at TEXT,
  responded_at TEXT,
  checkin_id TEXT REFERENCES checkins(id)
);
```

---

## 14. Hackathon Bonus Badges Targeted

| Badge | How |
|---|---|
| 🎨 **Off-Brand** | Custom Gradio CSS — warm Ghanaian earth tones, not default blue |
| 📓 **Field Notes** | Write-up: "Building multilingual AI for African families with small models" |
| 🎯 **Well-Tuned** | Already qualified — `teckedd/whisper_small-waxal_akan-asr-v1` is your published fine-tune. Wire it into the app and you're done. |

---

## 15. Demo Video Script (60 seconds)

1. *"My uncle passed away. We didn't know he was sick until it was too late."* — 5s, face to camera
2. Dashboard — 8 family members, all green — 8s
3. One member goes amber: silence detected → WhatsApp nudge sent to nearest relative — 10s
4. Voice call triggers — phone rings — AI speaks in Twi — 10s
5. Transcript appears on dashboard, coordinator sees summary — 8s
6. Nearest relative checks in, field report closes the loop — 8s
7. *"This is Adwuma Pa. Good work — so no family member slips away unnoticed."* — 5s, logo

---

*Version 2 — Updated June 6, 2026*
*Key changes from V1: Whisper removed as primary ASR · MMS-1b-all confirmed as primary · ASR eval space added · Model swap architecture added · Fine-tuning script added · Dataset section updated with Common Voice access change*
