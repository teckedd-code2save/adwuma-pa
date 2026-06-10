# Field Notes: Building Small AI For Ghanaian Family Care

Ani Kɛse started from a specific family failure mode: an elder can become sick, nobody nearby knows, the coordinator is far away, and the group chat never turns concern into accountable action.

The product decision was to avoid a generic "elder care assistant" and build the smallest useful loop:

1. An elder or nearby relative submits a text or voice check-in.
2. A small model extracts health, mobility, isolation, food, or urgent medical signals.
3. The dashboard turns that into green, amber, or red status.
4. If silence or concern appears, the system routes follow-up to the nearest relative.
5. The coordinator closes the loop only when a field report arrives.

## Small-Model Constraint

The planned production stack remains under the 32B cap:

- ASR: `facebook/mms-1b-all` for Twi/Fante, with `teckedd/whisper_small-waxal_akan-asr-v1` as the published fine-tuned alternative.
- LLM: `Qwen/Qwen2.5-7B-Instruct` for concern scoring and summarization.
- TTS: `facebook/mms-tts` for Twi/Fante call prompts.

The Space ships with a deterministic concern scorer so judges can test the full care loop without waiting for large model cold starts. The service boundaries are explicit, so the GPU-backed model can replace the fallback without changing the product workflow.

## What Makes It Backyard AI

This is not a broad productivity tool. It is built for one family coordinator, elders in Ghana, and nearby relatives who can physically check in. The value is not the transcript; the value is that someone becomes accountable for the next step.

## What Is Still Honest Future Work

Twi ASR can work well enough for concern detection, but Fante remains harder. The app therefore keeps text input available and treats low-confidence voice transcription as a reason to ask for typed confirmation. That is less flashy than pretending the model is perfect, but it is the safer care product.

## What I Learned

For low-resource language apps, the best demo is not perfect transcription. The best demo is a resilient workflow around imperfect transcription: confidence checks, fallback input, human relay, and loop closure.

