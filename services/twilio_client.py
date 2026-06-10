from __future__ import annotations

import os
import re
from dataclasses import dataclass

from db import database as db


@dataclass
class TwilioResult:
    ok: bool
    status: str
    message: str
    sid: str | None = None


def configured() -> bool:
    return bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN") and os.getenv("TWILIO_WHATSAPP_FROM"))


def normalize_e164(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = raw.replace("whatsapp:", "").strip()
    if raw.startswith("+"):
        return "+" + re.sub(r"\D", "", raw)
    if raw.startswith("00"):
        return "+" + re.sub(r"\D", "", raw[2:])
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("0") and len(digits) == 10:
        return f"+233{digits[1:]}"
    if digits.startswith("233"):
        return f"+{digits}"
    return f"+{digits}" if digits else ""


def normalize_whatsapp_to(value: str | None) -> str:
    phone = normalize_e164(value)
    return f"whatsapp:{phone}" if phone else ""


def configured_from() -> str:
    return normalize_whatsapp_to(os.getenv("TWILIO_WHATSAPP_FROM"))


def public_url(path: str) -> str | None:
    base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not base_url:
        return None
    return f"{base_url}{path}"


def send_whatsapp(to: str, body: str) -> TwilioResult:
    recipient = normalize_whatsapp_to(to)
    if not recipient:
        return TwilioResult(False, "failed", "No valid WhatsApp recipient number was provided.")
    if not configured():
        return TwilioResult(False, "not_configured", "Twilio credentials are not configured; no WhatsApp was sent.")
    sender = configured_from()
    if not sender:
        return TwilioResult(False, "failed", "TWILIO_WHATSAPP_FROM must be a WhatsApp sender such as whatsapp:+14155238886.")
    try:
        from twilio.rest import Client

        client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        kwargs = {"from_": sender, "to": recipient, "body": body}
        status_callback = public_url("/twilio/status")
        if status_callback:
            kwargs["status_callback"] = status_callback
        message = client.messages.create(**kwargs)
        return TwilioResult(True, "sent", "WhatsApp sent.", message.sid)
    except Exception as exc:
        error = str(exc)
        if "could not find a Channel with the specified From address" in error:
            error = (
                f"{error}\n\n"
                f"Using From={sender}. In Twilio, this must be the WhatsApp Sandbox sender "
                "whatsapp:+14155238886 or an approved WhatsApp sender connected to this account."
            )
        return TwilioResult(False, "failed", error)


def checkin_url(token: str) -> str:
    base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    path = f"/checkin/{token}"
    return f"{base_url}{path}" if base_url else path


def send_request_link(request_id: str) -> TwilioResult:
    request = db.one(
        """
        SELECT r.*,
               m.name AS elder_name,
               m.whatsapp AS elder_whatsapp,
               m.phone AS elder_phone,
               COALESCE(c.id, m.id) AS recipient_member_id,
               COALESCE(c.name, m.name) AS recipient_name,
               COALESCE(c.whatsapp, m.whatsapp) AS recipient_whatsapp,
               COALESCE(c.phone, m.phone) AS recipient_phone,
               n.contact_id AS nudge_contact_id
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        LEFT JOIN nudges n ON n.id = r.related_nudge_id
        LEFT JOIN members c ON c.id = n.contact_id
        WHERE r.id = ?
        """,
        (request_id,),
    )
    if not request:
        return TwilioResult(False, "failed", "No request found.")
    if request["request_type"] == "field_report" and not request.get("nudge_contact_id"):
        return TwilioResult(False, "failed", "Field report has no assigned relative contact. Add a valid affiliation first.")
    body = request_message_body(request)
    recipient = request.get("recipient_whatsapp") or request.get("recipient_phone")
    result = send_whatsapp(recipient, body)
    db.add_outbound_message(
        request_id=request["id"],
        recipient_member_id=request.get("recipient_member_id") or request["member_id"],
        recipient=recipient,
        body=body,
        status=result.status,
        provider_sid=result.sid,
        error=None if result.ok else result.message,
    )
    return result


def request_message_body(request: dict) -> str:
    reason = (request.get("reason_code") or "check-in").replace("_", " ")
    if request.get("request_type") == "field_report":
        return (
            f"Ani Kɛse needs your help checking on {request['elder_name']}. "
            f"Reason: {reason}. "
            f"Please send a short family report here: {checkin_url(request['token'])}"
        )
    return (
        f"Ani Kɛse check-in for {request['elder_name']}. "
        f"Reason: {reason}. "
        f"Please respond here: {checkin_url(request['token'])}"
    )


def record_inbound(sender: str, body: str, channel: str = "whatsapp") -> str:
    normalized_sender = normalize_whatsapp_to(sender)
    normalized_phone = normalized_sender.replace("whatsapp:", "")
    member = db.one(
        """
        SELECT id FROM members
        WHERE whatsapp = ? OR phone = ? OR whatsapp = ? OR phone = ?
        LIMIT 1
        """,
        (sender, sender, normalized_sender, normalized_phone),
    )
    status = "matched" if member else "unmatched"
    return db.add_inbound_message(sender, body, channel, matched_member_id=member["id"] if member else None, status=status)


def receive_whatsapp_reply(sender: str, body: str) -> dict:
    normalized_sender = normalize_whatsapp_to(sender)
    normalized_phone = normalized_sender.replace("whatsapp:", "")
    member = db.one(
        """
        SELECT * FROM members
        WHERE whatsapp = ? OR phone = ? OR whatsapp = ? OR phone = ?
        LIMIT 1
        """,
        (sender, sender, normalized_sender, normalized_phone),
    )
    message_id = db.add_inbound_message(
        sender=normalized_sender or sender,
        body=body,
        channel="whatsapp",
        matched_member_id=member["id"] if member else None,
        status="matched" if member else "unmatched",
    )
    if not member:
        return {"ok": True, "message_id": message_id, "status": "unmatched", "detail": "No member matched this sender."}

    request = db.one(
        """
        SELECT r.token, r.request_type, r.member_id
        FROM checkup_requests r
        LEFT JOIN nudges n ON n.id = r.related_nudge_id
        WHERE r.status IN ('pending', 'sent', 'processing')
          AND r.completed_at IS NULL
          AND (
            r.member_id = ?
            OR (r.request_type = 'field_report' AND n.contact_id = ?)
          )
        ORDER BY
          CASE
            WHEN r.request_type = 'field_report' AND n.contact_id = ? THEN 0
            WHEN r.member_id = ? THEN 1
            ELSE 2
          END,
          CASE r.priority
            WHEN 'red' THEN 0
            WHEN 'amber' THEN 1
            ELSE 2
          END,
          r.created_at DESC
        LIMIT 1
        """,
        (member["id"], member["id"], member["id"], member["id"]),
    )
    if not request:
        return {"ok": True, "message_id": message_id, "status": "matched_no_open_request", "member_id": member["id"]}

    from services import pipeline

    result = pipeline.submit_request_response(
        token=request["token"],
        text=body,
        language=member.get("language") or "twi",
        input_type="text",
        source="field_report" if request["request_type"] == "field_report" else "self",
    )
    return {
        "ok": True,
        "message_id": message_id,
        "status": "processed",
        "member_id": member["id"],
        "request_member_id": request["member_id"],
        "request_type": request["request_type"],
        "request_token": request["token"],
        "pipeline": result,
    }


def record_status_callback(message_sid: str, message_status: str, error_code: str = "", error_message: str = "") -> dict:
    if not message_sid:
        return {"ok": False, "message": "No MessageSid supplied."}
    error = " ".join(part for part in [error_code, error_message] if part).strip() or None
    db.update_outbound_status(message_sid, message_status or "status_callback", error)
    return {"ok": True, "sid": message_sid, "status": message_status, "error": error}
