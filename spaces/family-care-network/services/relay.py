from __future__ import annotations

import os
from datetime import datetime, timezone

from db import database as db

ROUTE_ROLE_PRIORITY = {
    "first_party_contact": 0,
    "nearby_relative": 1,
    "emergency_contact": 2,
    "caregiver": 3,
    "backup_coordinator": 4,
    "primary_coordinator": 5,
}


def dashboard_rows() -> list[dict]:
    members = db.rows(
        """
        SELECT m.*,
               c.submitted_at AS last_checkin_at,
               c.summary AS last_summary,
               c.concern_level AS last_concern,
               ra.resolved_at AS last_resolved_at,
               c.analysis_status AS last_analysis_status
        FROM members m
        LEFT JOIN checkins c ON c.id = (
          SELECT id FROM checkins WHERE member_id = m.id ORDER BY submitted_at DESC LIMIT 1
        )
        LEFT JOIN alerts ra ON ra.id = (
          SELECT id FROM alerts
          WHERE member_id = m.id AND resolved = 1 AND resolved_at IS NOT NULL
          ORDER BY resolved_at DESC LIMIT 1
        )
        WHERE m.active = 1
        ORDER BY COALESCE(c.concern_level, 0) DESC, m.name ASC
        """
    )
    return [with_status(member) for member in members]


def with_status(member: dict) -> dict:
    concern = member.get("last_concern") or 0
    minutes_silent = minutes_since(latest_care_timestamp(member))
    reminder_minutes = member.get("reminder_minutes") or 10080
    amber_minutes = member.get("escalation_minutes_amber") or 14400
    red_minutes = member.get("escalation_minutes_red") or 20160
    if concern >= 7 or minutes_silent >= red_minutes:
        status = "Urgent follow-up"
        next_action = "Call this person or ask the assigned relative to confirm they are okay"
    elif concern >= 4 or minutes_silent >= amber_minutes:
        status = "Needs attention"
        next_action = "Ask the assigned relative to check in and report back"
    elif minutes_silent >= reminder_minutes:
        status = "Check soon"
        next_action = "Send a warm check-in reminder"
    else:
        status = "Routine"
        next_action = "Normal schedule"
    contact = route_contact(member["id"])
    care_route = "No care contact assigned"
    if contact:
        role = (contact.get("care_role") or "family").replace("_", " ")
        care_route = f"{contact['name']} ({role})"
    return {
        "Name": member["name"],
        "City": member.get("location_city") or "",
        "Region": member.get("location_region") or "",
        "Language": member.get("language") or "",
        "Role": member.get("family_role") or "relative",
        "Coordinator": "Yes" if member.get("is_coordinator") else "No",
        "Status": status,
        "Concern": concern,
        "Minutes silent": minutes_silent,
        "Reminder min": reminder_minutes,
        "Amber min": amber_minutes,
        "Red min": red_minutes,
        "Last summary": member.get("last_summary") or ("Care loop recently closed" if member.get("last_resolved_at") else "No check-in yet"),
        "Analysis": member.get("last_analysis_status") or "none",
        "Next action": next_action,
        "Care route": care_route,
        "Token": member.get("checkin_url_token") or "",
    }


def days_since(timestamp: str | None) -> int:
    minutes = minutes_since(timestamp)
    return int(minutes / 1440)


def minutes_since(timestamp: str | None) -> int:
    if not timestamp:
        return 9999
    then = datetime.fromisoformat(timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - then).total_seconds() / 60))


def latest_care_timestamp(member: dict) -> str | None:
    candidates = [member.get("last_checkin_at"), member.get("last_resolved_at")]
    parsed = []
    for value in candidates:
        if not value:
            continue
        try:
            timestamp = datetime.fromisoformat(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            parsed.append((timestamp, value))
        except Exception:
            continue
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def closure_grace_minutes() -> int:
    return max(0, int(os.getenv("ANI_KESE_CLOSURE_GRACE_MINUTES", "60") or 0))


def recently_closed(member: dict) -> bool:
    resolved_at = member.get("last_resolved_at")
    return bool(resolved_at and latest_care_timestamp(member) == resolved_at and minutes_since(resolved_at) < closure_grace_minutes())


def human_duration(minutes: int) -> str:
    minutes = max(0, int(minutes))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    rem_minutes = minutes % 60
    if hours < 24:
        suffix = f" {rem_minutes} min" if rem_minutes else ""
        return f"{hours} hour{'s' if hours != 1 else ''}{suffix}"
    days = hours // 24
    rem_hours = hours % 24
    suffix = f" {rem_hours} hr" if rem_hours else ""
    return f"{days} day{'s' if days != 1 else ''}{suffix}"


def route_contact(elder_id: str) -> dict | None:
    contact = db.one(
        """
        SELECT c.*, a.care_role, a.relationship, a.priority AS affiliation_priority,
               a.can_coordinate
        FROM member_affiliations a
        JOIN members c ON c.id = a.related_member_id
        WHERE a.subject_member_id = ?
          AND c.active = 1
          AND a.care_role IN (
            'first_party_contact',
            'nearby_relative',
            'emergency_contact',
            'caregiver',
            'backup_coordinator',
            'primary_coordinator'
          )
        ORDER BY
          CASE a.care_role
            WHEN 'first_party_contact' THEN 0
            WHEN 'nearby_relative' THEN 1
            WHEN 'emergency_contact' THEN 2
            WHEN 'caregiver' THEN 3
            WHEN 'backup_coordinator' THEN 4
            WHEN 'primary_coordinator' THEN 5
            ELSE 9
          END,
          a.priority DESC,
          c.is_coordinator DESC,
          c.name ASC
        LIMIT 1
        """,
        (elder_id,),
    )
    if contact:
        return contact
    return db.one(
        """
        SELECT c.*, 'first_party_contact' AS care_role, '' AS relationship,
               f.priority AS affiliation_priority, 0 AS can_coordinate
        FROM first_party_contacts f
        JOIN members c ON c.id = f.contact_id
        WHERE f.elder_id = ?
        ORDER BY f.priority ASC
        LIMIT 1
        """,
        (elder_id,),
    )


def simulate_nudge(elder_id: str) -> str:
    elder = db.one("SELECT * FROM members WHERE id = ?", (elder_id,))
    contact = route_contact(elder_id)
    if not elder:
        return "No family member found."
    if not contact:
        return f"No care contact assigned for {elder['name']}. Add an affiliation in Members."
    role = (contact.get("care_role") or "family").replace("_", " ")
    return (
        f"WhatsApp draft to {contact['name']} ({role}): Hi {contact['name']}, we have not heard from "
        f"{elder['name']} in {elder.get('location_city') or 'their town'} recently. Could you check on them today "
        f"and report back through their Ani Kɛse link?"
    )


def checkin_link_for_request(request_id: str) -> str:
    request = db.one("SELECT token FROM checkup_requests WHERE id = ?", (request_id,))
    if not request:
        return ""
    return f"/checkin/{request['token']}"


def scan_silence(excluded_member_ids: list[str] | None = None) -> list[str]:
    actions: list[str] = []
    excluded = set(excluded_member_ids or [])
    members = db.rows(
        """
        SELECT m.*,
               c.submitted_at AS last_checkin_at,
               c.concern_level AS last_concern,
               ra.resolved_at AS last_resolved_at
        FROM members m
        LEFT JOIN checkins c ON c.id = (
          SELECT id FROM checkins WHERE member_id = m.id ORDER BY submitted_at DESC LIMIT 1
        )
        LEFT JOIN alerts ra ON ra.id = (
          SELECT id FROM alerts
          WHERE member_id = m.id AND resolved = 1 AND resolved_at IS NOT NULL
          ORDER BY resolved_at DESC LIMIT 1
        )
        WHERE m.active = 1
        ORDER BY m.name ASC
        """
    )
    for member in members:
        if member["id"] in excluded:
            actions.append(f"Excluded from autopilot: {member['name']}.")
            continue
        if recently_closed(member):
            actions.append(f"Recently closed care loop for {member['name']}; no new request.")
            continue
        silent = minutes_since(latest_care_timestamp(member))
        no_recorded_checkin = not member.get("last_checkin_at") and not member.get("last_resolved_at")
        amber = member.get("escalation_minutes_amber") or 14400
        red = member.get("escalation_minutes_red") or 20160
        contact = route_contact(member["id"])
        reminder = member.get("reminder_minutes") or 10080
        if silent >= red:
            silent_text = "no recorded check-in yet" if no_recorded_checkin else human_duration(silent)
            red_text = human_duration(red)
            elder_detail = (
                f"We have {silent_text} for {member['name']}. This is past their urgent follow-up window of {red_text}."
                if no_recorded_checkin
                else f"We have not heard from {member['name']} for {silent_text}. This is past their urgent follow-up window of {red_text}."
            )
            alert_id = db.create_alert(
                member["id"],
                "red_silence",
                elder_detail,
            )
            existing = db.open_checkup_request(member["id"], "red_silence", "elder_checkin")
            request_id = db.create_checkup_request(
                member["id"],
                "red_silence",
                elder_detail,
                channel="whatsapp",
                priority="red",
                related_alert_id=alert_id,
            )
            if contact:
                field_existing = db.open_checkup_request(member["id"], "first_party_red_silence", "field_report")
                if not field_existing:
                    nudge_id = db.add_nudge(member["id"], contact["id"])
                    db.create_checkup_request(
                        member["id"],
                        "first_party_red_silence",
                        f"{member['name']} needs urgent follow-up. Please check on them and send a short family update.",
                        request_type="field_report",
                        channel="whatsapp",
                        requester="Ani Kɛse autopilot",
                        priority="red",
                        related_alert_id=alert_id,
                        related_nudge_id=nudge_id,
                    )
            routed = f" Routed to {contact['name']}." if contact else " No care contact assigned."
            state = "Existing open urgent request reused" if existing else "Urgent check-in request queued"
            actions.append(f"{state} for {member['name']}: case {alert_id}.{routed}")
        elif silent >= amber:
            silent_text = "no recorded check-in yet" if no_recorded_checkin else human_duration(silent)
            amber_text = human_duration(amber)
            elder_detail = (
                f"We have {silent_text} for {member['name']}. This is past their check-soon window of {amber_text}."
                if no_recorded_checkin
                else f"We have not heard from {member['name']} for {silent_text}. This is past their check-soon window of {amber_text}."
            )
            alert_id = db.create_alert(
                member["id"],
                "amber_silence",
                elder_detail,
            )
            existing = db.open_checkup_request(member["id"], "amber_silence", "elder_checkin")
            request_id = db.create_checkup_request(
                member["id"],
                "amber_silence",
                elder_detail,
                channel="whatsapp",
                priority="amber",
                related_alert_id=alert_id,
            )
            if contact:
                field_existing = db.open_checkup_request(member["id"], "first_party_amber_silence", "field_report")
                if not field_existing:
                    nudge_id = db.add_nudge(member["id"], contact["id"])
                    db.create_checkup_request(
                        member["id"],
                        "first_party_amber_silence",
                        f"We have not heard from {member['name']} for a while. Please check on them and send a short family update.",
                        request_type="field_report",
                        channel="whatsapp",
                        requester="Ani Kɛse autopilot",
                        priority="amber",
                        related_alert_id=alert_id,
                        related_nudge_id=nudge_id,
                    )
            routed = f" Routed to {contact['name']}." if contact else " No care contact assigned."
            state = "Existing open check-soon request reused" if existing else "Check-soon request queued"
            actions.append(f"{state} for {member['name']}: case {alert_id}.{routed}")
        elif silent >= reminder:
            silent_text = "no recorded check-in yet" if no_recorded_checkin else human_duration(silent)
            reminder_text = human_duration(reminder)
            elder_detail = (
                f"We have {silent_text} for {member['name']}. Their routine check-in window is {reminder_text}."
                if no_recorded_checkin
                else f"We have not heard from {member['name']} for {silent_text}. Their routine check-in window is {reminder_text}."
            )
            alert_id = db.create_alert(
                member["id"],
                "reminder_silence",
                elder_detail,
            )
            existing = db.open_checkup_request(member["id"], "reminder_silence", "elder_checkin")
            request_id = db.create_checkup_request(
                member["id"],
                "reminder_silence",
                elder_detail,
                channel="whatsapp",
                priority="routine",
                related_alert_id=alert_id,
            )
            state = "Existing open routine request reused" if existing else "Routine check-in request queued"
            actions.append(f"{state} for {member['name']}: case {alert_id}.")
    if not actions:
        actions.append("No silence escalations. All active family members are inside their configured thresholds.")
    return actions
