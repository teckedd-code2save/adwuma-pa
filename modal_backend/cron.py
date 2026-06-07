from __future__ import annotations

import os

import modal
import requests


app = modal.App("adwuma-pa-cron")

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")


@app.function(
    image=image,
    schedule=modal.Cron("0 */6 * * *", timezone="Africa/Accra"),
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=5,
    timeout=120,
)
def autopilot_scan() -> dict:
    """
    Deploy this only for final validation/demo.

    During normal development, run the dashboard "Run silence scan now" button instead so
    Modal cron is not consuming any of the $50 credit budget.
    """
    base_url = os.getenv("ADWUMA_PA_SPACE_URL", "").rstrip("/")
    secret = os.getenv("ADWUMA_PA_AUTOPILOT_SECRET", "")
    if not base_url:
        return {"status": "skipped", "reason": "ADWUMA_PA_SPACE_URL is not configured"}
    response = requests.post(
        f"{base_url}/api/autopilot/scan",
        headers={"X-Adwuma-Secret": secret},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
