from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from db import database as db
from services.relay import scan_silence
from services import twilio_client


def run_autopilot_scan(force: bool = False, actor: str = "Ani Kɛse autopilot") -> dict[str, Any]:
    db.init_db()
    settings = db.autopilot_settings()
    if not settings["enabled"] and not force:
        return {
            "status": "skipped",
            "reason": "Autopilot is off.",
            "settings": settings,
            "actions": [],
            "deliveries": [],
        }
    if not force and not scan_due(settings):
        return {
            "status": "skipped",
            "reason": "Autopilot scan interval has not elapsed.",
            "settings": settings,
            "actions": [],
            "deliveries": [],
        }

    actions = scan_silence()
    deliveries = send_autopilot_whatsapp() if settings["send_whatsapp"] else []
    result = {
        "status": "complete",
        "actor": actor,
        "settings": db.autopilot_settings(),
        "actions": actions,
        "deliveries": deliveries,
    }
    db.set_setting("autopilot.last_scan_at", db.now_iso())
    db.set_setting("autopilot.last_scan_result", result)
    return result


def scan_due(settings: dict[str, Any]) -> bool:
    last_scan_at = settings.get("last_scan_at")
    if not last_scan_at:
        return True
    then = datetime.fromisoformat(last_scan_at)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    elapsed_minutes = int((datetime.now(timezone.utc) - then).total_seconds() / 60)
    return elapsed_minutes >= int(settings.get("scan_interval_minutes") or 360)


def send_autopilot_whatsapp() -> list[str]:
    pending = db.rows(
        """
        SELECT id
        FROM checkup_requests
        WHERE requester IN ('Ani Kɛse autopilot', 'Adwuma Pa autopilot')
          AND channel = 'whatsapp'
          AND status = 'pending'
        ORDER BY
          CASE priority WHEN 'red' THEN 0 WHEN 'amber' THEN 1 ELSE 2 END,
          created_at ASC
        LIMIT 20
        """
    )
    deliveries = []
    for row in pending:
        result = twilio_client.send_request_link(row["id"])
        if result.sid:
            deliveries.append(f"{row['id']}: {result.message} SID: {result.sid}")
        else:
            deliveries.append(f"{row['id']}: {result.message}")
    if not deliveries:
        deliveries.append("No pending autopilot WhatsApp messages to send.")
    return deliveries


def autopilot_summary_html() -> str:
    settings = db.autopilot_settings()
    enabled = "On" if settings["enabled"] else "Off"
    delivery = "Auto-send WhatsApp" if settings["send_whatsapp"] else "Queue only"
    last = settings.get("last_scan_at") or "Never"
    result = settings.get("last_scan_result") or {}
    actions = result.get("actions") if isinstance(result, dict) else []
    action_count = len(actions or [])
    return f"""
<div class="ap-autopilot">
  <div><strong>Status:</strong> {enabled}</div>
  <div><strong>Cadence:</strong> every {settings['scan_interval_minutes']} minutes</div>
  <div><strong>Delivery:</strong> {delivery}</div>
  <div><strong>Last scan:</strong> {last}</div>
  <div><strong>Last actions:</strong> {action_count}</div>
</div>
"""
