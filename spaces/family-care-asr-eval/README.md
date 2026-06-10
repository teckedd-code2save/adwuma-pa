---
title: Ani Kɛse ASR Eval
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Twi and Fante ASR comparison
---

# Ani Kɛse ASR Eval

This Space is the first build step for Ani Kɛse. It tests small ASR models on real Twi, Fante, and Ghanaian English family recordings before choosing the production voice path.

Community testers can vote for the model that best preserves the meaning of each sample. Rough WER is only shown when exact reference text is provided, so votes are useful when people can judge the transcript by ear.

## Models

- `facebook/mms-1b-all`: primary recommendation for Twi and Fante coverage.
- `teckedd/whisper_small-waxal_akan-asr-v1`: published Akan fine-tune for the Well-Tuned badge.
- `GiftMark/akan-whisper-model`: community Akan fallback.

## Test Protocol

1. Record 5 to 10 natural samples from the intended family users.
2. Test Twi first, then Fante, then Ghanaian English.
3. Add the reference text when possible to compare rough WER.
4. Choose the model that best captures concern signals, not perfect spelling.
5. Keep text fallback in the main app for low-confidence or garbled output.

## Voting

After comparing outputs, pick the model that best captured the care signal. Add a short note such as "caught walking pain" or "missed the isolation phrase." These votes help decide whether the next step should be fine-tuning.
