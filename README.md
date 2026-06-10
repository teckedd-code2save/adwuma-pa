---
title: Ani Kɛse
emoji: 🫶
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered family wellness network for Ghanaian elders
models:
  - facebook/mms-1b-all
  - ninte/twi-en-nllb-v2
  - Qwen/Qwen2.5-7B-Instruct
  - facebook/mms-tts-aka
  - facebook/mms-tts-eng
  - teckedd/whisper_small-waxal_akan-asr-v1
  - GiftMark/akan-whisper-model
---

# Ani Kɛse

Ani Kɛse is a small-model family care network for Ghanaian elders. It creates real checkup requests, collects text or voice responses in Twi, Fante, or English, translates Akan-family responses to English, analyzes concern with Qwen, routes follow-up to nearby relatives, and gives the family coordinator a live Gradio dashboard.

Built for the Build Small Hackathon, Backyard AI track.

## Built With OpenAI Codex

OpenAI Codex is being used as the coding agent for this build. Codex created and patched the ASR eval Space, the main family care Space, SQLite persistence, configurable silence escalation, and the community voting workflow. See `CODEX_BUILD_LOG.md` and `HACKATHON_TODO.md`.

## Why This Should Be Competitive

- Specific real user: a Ghanaian family coordinator checking on elders across cities.
- Small-model compliant: ASR, concern scoring, and TTS are each under the 32B parameter cap.
- Real workflow: tokenized checkup requests, silence detection, first-party relay, alerts, and loop closure.
- Bonus badges targeted: custom Gradio UI, field notes, published fine-tuned Akan ASR model, and shared build trace.
- OpenAI track angle: Codex-assisted build process, documented agent trace, and a practical agentic care workflow where the AI routes work to the right human.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL.

## Hugging Face Space

Use the main app Space:

```bash
huggingface-cli upload build-small-hackathon/family-care-network . . --repo-type space
```

For the ASR evaluation Space, set `app_file: asr_eval.py` in that Space README or upload `asr_eval.py` as `app.py`.

## Files

- `app.py`: main Gradio coordinator dashboard and request-backed check-in workflow.
- `asr_eval.py`: standalone ASR model comparison Space.
- `config/models.py`: model IDs and parameter accounting.
- `db/database.py`: SQLite persistence.
- `services/asr.py`: lazy ASR service.
- `services/modal_client.py`: cost-safe Modal API client; unavailable inference returns `needs_review`.
- `services/pipeline.py`: ASR -> translation -> Qwen concern pipeline.
- `services/relay.py`: silence detection, request creation, and contact routing.
- `modal_backend/adwuma_modal.py`: Modal endpoints for health, translation, ASR, Qwen analysis, and TTS.
- `modal_backend/cron.py`: deploy-only-when-needed Modal cron skeleton.
- `finetune/finetune_mms_twi.py`: cost-capped Modal harness for Twi MMS ASR fine-tuning on `ghananlpcommunity/twi-speech-text-multispeaker-16k`.
- `docs/asr-finetune-pipeline.md`: complete ASR fine-tune, evaluation, push, cleanup, and testing-space pipeline.
- `SUBMISSION.md`: demo script, social copy, and judging checklist.
- `FIELD_NOTES.md`: report draft for the Field Notes badge.
