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
        message = client.messages.create(
            from_=sender,
            to=recipient,
            body=body,
        )
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
        SELECT r.*, m.name, m.whatsapp, m.phone
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        WHERE r.id = ?
        """,
        (request_id,),
    )
    if not request:
        return TwilioResult(False, "failed", "No request found.")
    body = request_message_body(request)
    result = send_whatsapp(request.get("whatsapp") or request.get("phone"), body)
    db.add_outbound_message(
        request_id=request["id"],
        recipient_member_id=request["member_id"],
        recipient=request.get("whatsapp") or request.get("phone"),
        body=body,
        status=result.status,
        provider_sid=result.sid,
        error=None if result.ok else result.message,
    )
    return result


def request_message_body(request: dict) -> str:
    reason = (request.get("reason_code") or "check-in").replace("_", " ")
    return (
        f"Adwuma Pa check-in for {request['name']}. "
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
        SELECT token
        FROM checkup_requests
        WHERE member_id = ?
          AND status IN ('pending', 'sent', 'processing')
          AND completed_at IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (member["id"],),
    )
    if not request:
        return {"ok": True, "message_id": message_id, "status": "matched_no_open_request", "member_id": member["id"]}

    from services import pipeline

    result = pipeline.submit_request_response(
        token=request["token"],
        text=body,
        language=member.get("language") or "twi",
        input_type="text",
        source="self",
    )
    return {
        "ok": True,
        "message_id": message_id,
        "status": "processed",
        "member_id": member["id"],
        "request_token": request["token"],
        "pipeline": result,
    }
