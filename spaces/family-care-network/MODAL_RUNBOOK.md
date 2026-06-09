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
```

Do not run ASR/Qwen/TTS repeatedly during UI iteration. Those endpoints are GPU-backed.

## Cron

Do not deploy cron during development. Use the dashboard button "Run silence scan now".

Only for demo recording:

```bash
modal deploy modal_backend/cron.py
```

## Stop After Testing

Stop the inference app and cron app after validation/demo:

```bash
modal app stop adwuma-pa-inference
modal app stop adwuma-pa-cron
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
