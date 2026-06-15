from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from db import database as db
from services.relay import scan_silence
from services import twilio_client


def run_autopilot_scan(force: bool = False, actor: str = "Ani Kɛse autopilot") -> dict[str, Any]:
    db.init_db()
    settings = db.autopilot_settings()
    if not settings["enabled"] and not force:
        result = {
            "status": "skipped",
            "reason": "Autopilot is off.",
            "settings": settings,
            "actions": [],
            "deliveries": [],
        }
        db.add_autopilot_run(actor, result["status"], result["reason"], settings=settings)
        return result
    if not force and not scan_due(settings):
        result = {
            "status": "skipped",
            "reason": "Autopilot scan interval has not elapsed.",
            "settings": settings,
            "actions": [],
            "deliveries": [],
        }
        db.add_autopilot_run(actor, result["status"], result["reason"], settings=settings)
        return result

    try:
        actions = scan_silence(settings.get("excluded_member_ids") or [])
        deliveries = send_autopilot_whatsapp() if settings["send_whatsapp"] else ["WhatsApp delivery is set to queue only."]
        reason = scan_reason(actions, deliveries)
        result = {
            "status": "complete",
            "actor": actor,
            "reason": reason,
            "settings": db.autopilot_settings(),
            "actions": actions,
            "deliveries": deliveries,
        }
        db.set_setting("autopilot.last_scan_at", db.now_iso())
        db.set_setting("autopilot.last_scan_result", compact_last_scan_result(result))
        db.add_autopilot_run(actor, result["status"], reason, actions, deliveries, result["settings"])
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "reason": "Autopilot scan failed.",
            "settings": settings,
            "actions": [],
            "deliveries": [],
            "error": str(exc),
        }
        db.add_autopilot_run(actor, result["status"], result["reason"], settings=settings, error=str(exc))
        raise


def scan_due(settings: dict[str, Any]) -> bool:
    last_scan_at = settings.get("last_scan_at")
    if not last_scan_at:
        return True
    then = datetime.fromisoformat(last_scan_at)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    elapsed_seconds = (datetime.now(timezone.utc) - then).total_seconds()
    interval_seconds = int(settings.get("scan_interval_minutes") or 360) * 60
    return elapsed_seconds >= max(0, interval_seconds - 90)


def send_autopilot_whatsapp() -> list[str]:
    open_requests = db.rows(
        """
        SELECT id, member_id, priority, request_type, status
        FROM checkup_requests
        WHERE requester IN ('Ani Kɛse autopilot', 'Adwuma Pa autopilot')
          AND channel = 'whatsapp'
          AND status IN ('pending', 'sent')
        ORDER BY
          CASE priority WHEN 'red' THEN 0 WHEN 'amber' THEN 1 ELSE 2 END,
          CASE status WHEN 'pending' THEN 0 ELSE 1 END,
          created_at ASC
        LIMIT 20
        """
    )
    deliveries = []
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
    for row in open_requests:
        cap = db.member_frequency_cap(row["member_id"], row["priority"] or "routine")
        sent = db.outbound_count_for_member_priority(
            row["member_id"],
            row["priority"] or "routine",
            since,
            row.get("request_type"),
        )
        if sent >= cap:
            deliveries.append(f"{row['id']}: Frequency cap reached; no WhatsApp sent.")
            continue
        result = twilio_client.send_request_link(row["id"])
        action = "resent" if row["status"] == "sent" else "sent"
        if result.sid:
            deliveries.append(f"{row['id']}: WhatsApp {action}. SID: {result.sid}")
        else:
            deliveries.append(f"{row['id']}: {result.message}")
    if not deliveries:
        deliveries.append("No pending autopilot WhatsApp messages to send.")
    return deliveries


def scan_reason(actions: list[str], deliveries: list[str]) -> str:
    excluded_actions = [item for item in actions if item.startswith("Excluded from autopilot")]
    recently_closed_actions = [item for item in actions if item.startswith("Recently closed care loop")]
    meaningful_actions = [
        item
        for item in actions
        if not item.startswith("No silence escalations")
        and not item.startswith("Excluded from autopilot")
        and not item.startswith("Recently closed care loop")
    ]
    meaningful_deliveries = [
        item
        for item in deliveries
        if not item.startswith("No pending autopilot WhatsApp messages")
        and not item.startswith("WhatsApp delivery is set to queue only")
        and "Frequency cap reached" not in item
    ]
    if meaningful_deliveries:
        return f"Sent or attempted {len(meaningful_deliveries)} WhatsApp notification(s)."
    if any("Frequency cap reached" in item for item in deliveries):
        return "Frequency cap reached; no WhatsApp sent."
    if excluded_actions and not meaningful_actions:
        return "Excluded from autopilot."
    if recently_closed_actions and not meaningful_actions:
        return "Recently closed care loop; no new WhatsApp sent."
    if meaningful_actions:
        if all(item.startswith("Existing open") for item in meaningful_actions):
            return "Existing open request reused; no new WhatsApp sent."
        return f"Created or updated {len(meaningful_actions)} care action(s), but no WhatsApp was sent."
    return "No due family members; no WhatsApp sent."


def compact_last_scan_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "actor": result.get("actor"),
        "reason": result.get("reason"),
        "settings": {
            "enabled": result.get("settings", {}).get("enabled"),
            "scan_interval_minutes": result.get("settings", {}).get("scan_interval_minutes"),
            "send_whatsapp": result.get("settings", {}).get("send_whatsapp"),
            "excluded_member_count": len(result.get("settings", {}).get("excluded_member_ids") or []),
        },
        "actions": result.get("actions") or [],
        "deliveries": result.get("deliveries") or [],
    }


def autopilot_summary_html() -> str:
    settings = db.autopilot_settings()
    enabled = "On" if settings["enabled"] else "Off"
    delivery = "Auto-send WhatsApp" if settings["send_whatsapp"] else "Queue only"
    last = settings.get("last_scan_at") or "Never"
    result = settings.get("last_scan_result") or {}
    actions = result.get("actions") if isinstance(result, dict) else []
    action_count = len([item for item in actions or [] if not str(item).startswith("No silence escalations")])
    return f"""
<div class="ap-autopilot">
  <div><strong>Status:</strong> {enabled}</div>
  <div><strong>Cadence:</strong> every {settings['scan_interval_minutes']} minutes</div>
  <div><strong>Delivery:</strong> {delivery}</div>
  <div><strong>Last scan:</strong> {last}</div>
  <div><strong>Last actions:</strong> {action_count}</div>
</div>
"""
