# Modal Runbook

Use Modal only for targeted validation and demo recording. The HF Space is the lightweight UI; Modal is the metered inference and cron backend.

## Install And Login

```bash
python -m pip install modal
modal setup
```

Confirm auth:

```bash
modal token current
```

## Deploy For Testing

Start with health only:

```bash
modal deploy modal_backend/adwuma_modal.py
```

Set the deployed Modal base URL in the HF Space:

```bash
hf spaces variables set build-small-hackathon/family-care-network MODAL_API_BASE_URL=<modal-web-base-url>
```

Run exactly one validation per endpoint:

1. `/health`
2. `/translate` with one Twi phrase
3. `/transcribe` with one short audio sample
4. `/analyze` with one translated check-in
5. `/speak` with one short prompt

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
- GPU functions use short `scaledown_window=10`.
- Cron is not deployed until final demo validation.
- Failed or unavailable inference becomes `needs_review`; never fake a concern score.
