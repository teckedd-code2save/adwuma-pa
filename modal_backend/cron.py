from __future__ import annotations

import os

import modal
import requests


app = modal.App("ani-kese-cron")

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("adwuma-pa-autopilot")],
    schedule=modal.Cron("*/15 * * * *", timezone="Africa/Accra"),
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=5,
    timeout=120,
)
def autopilot_scan() -> dict:
    """
    Deploy this only for final validation/demo.

    During normal development, keep this undeployed and use the dashboard button instead.
    The Space owns the real scan interval and skips work until its configured cadence is due.
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
    result = response.json()
    print(result)
    return result
