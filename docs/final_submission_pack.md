# Ani Kese Final Submission Pack

## Primary Links

- Live Space: https://huggingface.co/spaces/build-small-hackathon/family-care-network
- Demo video: to be uploaded to YouTube after rendering
- Hugging Face article: https://huggingface.co/blog/build-small-hackathon/ani-kese-small-model-care-loop
- GitHub repository: https://github.com/teckedd-code2save/adwuma-pa
- ASR evaluation Space: https://huggingface.co/spaces/build-small-hackathon/family-care-asr-eval
- Hackathon page: https://huggingface.co/build-small-hackathon

## Short Description

Ani Kese is a small-model AI care network for Ghanaian families. It watches for missed check-ins, sends WhatsApp links, routes urgent follow-up to the assigned relative, accepts text or voice replies, translates Akan/Twi/Fante responses, scores concern with evidence, and keeps the care loop open until a coordinator closes it.

## What The Demo Shows

1. A coordinator registers family members and affiliations.
2. Autopilot monitors silence using per-person care policies.
3. When Auntie Afia misses her window, Ani Kese opens one care case.
4. It sends a direct check-in to Auntie Afia and a relative update request to Edward.
5. Edward receives the WhatsApp nudge, opens the secure link, and submits a field report.
6. The reply is linked to the exact request, analyzed, and shown with evidence.
7. The coordinator closes the loop, proving the family action was completed.

## Small Model Stack

- ASR: `facebook/mms-1b-all` using Akan `aka` routing for Twi/Fante.
- ASR fine-tune alternative: `teckedd/whisper_small-waxal_akan-asr-v1`.
- MMS fine-tune harness: `finetune/finetune_mms_twi.py`, built for Modal with GhanaNLP Twi speech.
- MMS fine-tune target: `teckedd/mms-akan-ani-kese-v1`.
- Translation: `ninte/twi-en-nllb-v2`.
- Structured analysis: `Qwen/Qwen2.5-7B-Instruct`.
- TTS: `facebook/mms-tts-aka` and `facebook/mms-tts-eng`.
- Delivery: Twilio WhatsApp Sandbox for demo links and replies.
- Hosting: Hugging Face Spaces with Gradio.
- Inference and scheduling: Modal.

Total small-model budget in the UI: 9.8B / 32B.

## Modal Work

Modal runs the metered pieces only when needed:

- `modal_backend/adwuma_modal.py` exposes health, translation, ASR, Qwen analysis, and TTS endpoints.
- `modal_backend/cron.py` wakes the Space autopilot on a 10-minute demo cadence.
- The Space owns the app-level scan interval, exclusions, and frequency caps.
- Containers use scale-down settings and are stopped after testing to control cost.
- Failed or unavailable model calls are stored as `needs_review`; the app does not fake scores.

## Fine-Tuning Work

The project includes a cost-capped Modal fine-tuning harness for Akan/Twi ASR:

- Primary dataset: `ghananlpcommunity/twi-speech-text-multispeaker-16k`.
- Supplemental/eval dataset explored: `AfriSpeech/youversion-african-speech`.
- Existing published comparison model: `teckedd/whisper_small-waxal_akan-asr-v1`.
- Evaluation Space includes MMS, the Whisper fine-tune, the MMS fine-tune target, and a community Akan Whisper fallback.

The fine-tune work surfaced the core product risk honestly: Ghanaian-family care needs better Twi/Fante ASR and translation, so the app keeps transcripts, translations, evidence, and review states visible instead of pretending the model is always right.

## YouTube Title

Ani Kese: Small-Model AI Care Loop for Ghanaian Families

## YouTube Description

Ani Kese is a small-model AI care network for Ghanaian families. It monitors missed check-ins, sends WhatsApp links, routes urgent follow-up to the right relative, accepts field reports, translates Akan/Twi/Fante responses, analyzes concern with evidence, and keeps the loop open until a coordinator closes it.

Built for the Hugging Face Build Small Hackathon.

Links:
- Space: https://huggingface.co/spaces/build-small-hackathon/family-care-network
- Article: https://huggingface.co/blog/build-small-hackathon/ani-kese-small-model-care-loop
- GitHub: https://github.com/teckedd-code2save/adwuma-pa
- ASR eval Space: https://huggingface.co/spaces/build-small-hackathon/family-care-asr-eval

Stack:
Hugging Face Spaces, Gradio, Modal, Twilio WhatsApp, MMS ASR/TTS, NLLB translation, Qwen structured analysis, and Akan ASR fine-tuning experiments.

## Social Post

I built Ani Kese for the Hugging Face Build Small Hackathon: a small-model care loop for Ghanaian families.

It detects missed check-ins, sends WhatsApp links, routes follow-up to the right relative, accepts text/voice replies, translates Akan/Twi/Fante, analyzes concern with evidence, and keeps the loop open until the family closes it.

Built with HF Spaces, Modal, Twilio, MMS ASR/TTS, NLLB translation, Qwen, and Akan ASR fine-tuning work.

Space: https://huggingface.co/spaces/build-small-hackathon/family-care-network
Article: https://huggingface.co/blog/build-small-hackathon/ani-kese-small-model-care-loop
GitHub: https://github.com/teckedd-code2save/adwuma-pa
