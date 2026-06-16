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
        SELECT r.id,
               r.member_id,
               r.priority,
               r.request_type,
               r.status,
               m.name AS watched_name,
               CASE WHEN r.request_type = 'field_report' THEN COALESCE(c.id, m.id) ELSE m.id END AS cap_member_id,
               CASE WHEN r.request_type = 'field_report' THEN COALESCE(c.name, m.name) ELSE m.name END AS cap_member_name,
               COALESCE(c.name, m.name) AS recipient_name,
               COALESCE(c.whatsapp, m.whatsapp, c.phone, m.phone) AS recipient_phone
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        LEFT JOIN nudges n ON n.id = r.related_nudge_id
        LEFT JOIN members c ON c.id = n.contact_id
        WHERE r.requester IN ('Ani Kɛse autopilot', 'Adwuma Pa autopilot')
          AND r.channel = 'whatsapp'
          AND r.status IN ('pending', 'sent')
        ORDER BY
          CASE r.priority WHEN 'red' THEN 0 WHEN 'amber' THEN 1 ELSE 2 END,
          CASE r.status WHEN 'pending' THEN 0 ELSE 1 END,
          r.created_at ASC
        LIMIT 20
        """
    )
    deliveries = []
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
    recent_attempt_since = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(timespec="seconds")
    for row in open_requests:
        cap_member_id = row.get("cap_member_id") or row["member_id"]
        cap = db.member_frequency_cap(cap_member_id, row["priority"] or "routine")
        sent = db.outbound_count_for_recipient_priority(
            cap_member_id,
            row["priority"] or "routine",
            since,
            row.get("request_type"),
            recent_attempt_since,
        )
        label = request_delivery_label(row)
        if sent >= cap:
            deliveries.append(
                f"{row['id']} ({label}): Frequency cap reached for {row.get('cap_member_name') or 'this recipient'} "
                f"({sent}/{cap} today); no WhatsApp sent."
            )
            continue
        result = twilio_client.send_request_link(row["id"])
        action = "resent" if row["status"] == "sent" else "sent"
        if result.sid:
            deliveries.append(f"{row['id']} ({label}): WhatsApp {action}. SID: {result.sid}")
        else:
            deliveries.append(f"{row['id']} ({label}): {result.message}")
    if not deliveries:
        deliveries.append("No pending autopilot WhatsApp messages to send.")
    return deliveries


def request_delivery_label(row: dict[str, Any]) -> str:
    watched = row.get("watched_name") or "unknown family member"
    recipient = row.get("recipient_name") or "unknown recipient"
    phone = mask_contact(row.get("recipient_phone")) or "number hidden"
    if row.get("request_type") == "field_report":
        return f"{recipient} checking on {watched}, to {phone}"
    return f"{watched} direct check-in, to {phone}"


def mask_contact(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    prefix = "whatsapp:" if text.lower().startswith("whatsapp:") else ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        return f"{prefix}+***{digits[-4:]}"
    return "hidden"


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
