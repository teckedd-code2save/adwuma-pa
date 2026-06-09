# Modal Runbook

Use Modal only for targeted validation and demo recording. The HF Space is the lightweight UI; Modal is the metered inference and cron backend.

## Install And Login

```bash
python -m pip install modal
modal setup
```

Confirm auth:

```bash
modal token info
```

If this reports `Token missing`, run:

```bash
modal token new
```

## Deploy For Testing

The Modal backend is a single FastAPI gateway exposed through `@modal.asgi_app`.
The HF Space expects one base URL with these paths:

- `/health`
- `/translate`
- `/transcribe`
- `/analyze`
- `/speak`

Deploy the app:

```bash
modal deploy modal_backend/adwuma_modal.py
```

Copy the `api` web URL from the deploy output or Modal dashboard. It should be the base URL for the ASGI app, not a per-function URL.

Set the deployed Modal base URL in the HF Space. The URL is not secret, but putting it in a secret is also fine:

```bash
hf spaces variables set build-small-hackathon/family-care-network MODAL_API_BASE_URL=<modal-web-base-url>
```

Or set it in the Space UI under **Settings -> Variables and secrets**:

```text
MODAL_API_BASE_URL=<modal-web-base-url>
```

## Validation Order

Run exactly one validation per endpoint, stopping when something fails:

1. `/health`
2. `/translate` with one Twi phrase
3. `/transcribe` with one short audio sample
4. `/analyze` with one translated check-in
5. `/speak` with one short prompt

Suggested low-cost checks:

```bash
python scripts/modal_smoke.py --base-url "$MODAL_API_BASE_URL" --health
python scripts/modal_smoke.py --base-url "$MODAL_API_BASE_URL" --translate "Me ho ye"
python scripts/modal_smoke.py --base-url "$MODAL_API_BASE_URL" --analyze-sample
python scripts/modal_smoke.py --base-url "$MODAL_API_BASE_URL" --transcribe-audio /path/to/short-real-akan.wav --language twi
python scripts/modal_smoke.py --base-url "$MODAL_API_BASE_URL" --speak "Me pe se me hwe wo ho." --language twi
```

Do not run ASR/Qwen/TTS repeatedly during UI iteration. Those endpoints are GPU-backed.
Use a real short elder-style audio sample for ASR; synthetic or silent audio is not a useful validation.

## Cron

Do not deploy cron during development. Use the dashboard button "Run silence scan now".

Only for demo recording:

```bash
modal deploy modal_backend/cron.py
```

## Stop After Testing

Stop the inference app and cron app after validation/demo:

```bash
modal app stop adwuma-pa-inference --yes
modal app stop adwuma-pa-cron --yes
```

Then remove or blank the HF Space variable if you want the public app to return `needs_review` instead of calling Modal:

```bash
hf spaces variables delete build-small-hackathon/family-care-network MODAL_API_BASE_URL
```

## Cost Policy

- Modal functions use `min_containers=0`.
- Modal functions use `max_containers=1`.
- The public API gateway is CPU-only.
- Translation is CPU-only.
- GPU functions use short `scaledown_window=10`.
- Cron is not deployed until final demo validation.
- Failed or unavailable inference becomes `needs_review`; never fake a concern score.

## Last Validation Session

2026-06-09:

- Modal auth confirmed for workspace `createdliving1000`.
- Deployed `adwuma-pa-inference`.
- API base URL: `https://createdliving1000--api.modal.run`.
- `/health` returned HTTP 200.
- `/translate` returned HTTP 200 for `Me ho ye, na me nsa aka aduan. Meda wo ase.`
- Translation output: `I am well, I have had food. Thank you.`
- Translation model: `ninte/twi-en-nllb-v2`.
- No ASR, Qwen, TTS, or cron endpoints were tested in this session.
- App was stopped after validation; `modal app list` showed state `stopped` and `0` tasks.

2026-06-09 Qwen validation:

- Redeployed `adwuma-pa-inference`.
- API base URL: `https://createdliving1000--api.modal.run`.
- `/health` returned HTTP 200.
- `/analyze` returned HTTP 200 for a translated routine Twi check-in.
- Qwen model: `Qwen/Qwen2.5-7B-Instruct`.
- Strict JSON result included `summary`, `concern_level`, `flags`, `sentiment`, `evidence`, `recommended_action`, and `confidence`.
- Result summary: `The elder reported being well and having eaten.`
- Result concern level: `0`.
- Result recommended action: `normal`.
- No ASR, TTS, or cron endpoints were tested in this session.
- App was stopped after validation; `modal app list` showed state `stopped` and `0` tasks.

2026-06-09 TTS validation:

- Redeployed `adwuma-pa-inference`.
- API base URL: `https://createdliving1000--api.modal.run`.
- `/speak` returned HTTP 200 for `Me pe se me hwe wo ho.`
- TTS model: `facebook/mms-tts-aka`.
- Output audio: WAV payload, 16 kHz, 73,104 base64 characters.
- No ASR or cron endpoints were tested in this session.
- App was stopped after validation; `modal app list` showed state `stopped` and `0` tasks.

## Current Secret Checklist

HF Space:

```text
MODAL_API_BASE_URL=<modal-api-base-url>
PUBLIC_BASE_URL=https://build-small-hackathon-family-care-network.hf.space
TWILIO_ACCOUNT_SID=<twilio-account-sid>
TWILIO_AUTH_TOKEN=<twilio-auth-token>
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

Twilio Sandbox:

```text
When a message comes in: https://build-small-hackathon-family-care-network.hf.space/twilio/whatsapp
Method: POST
Status callback URL: leave blank; the app attaches /twilio/status per outbound send when PUBLIC_BASE_URL is configured.
```
