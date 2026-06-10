from __future__ import annotations

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
               c.analysis_status AS last_analysis_status
        FROM members m
        LEFT JOIN checkins c ON c.id = (
          SELECT id FROM checkins WHERE member_id = m.id ORDER BY submitted_at DESC LIMIT 1
        )
        WHERE m.active = 1
        ORDER BY COALESCE(c.concern_level, 0) DESC, m.name ASC
        """
    )
    return [with_status(member) for member in members]


def with_status(member: dict) -> dict:
    concern = member.get("last_concern") or 0
    minutes_silent = minutes_since(member.get("last_checkin_at"))
    reminder_minutes = member.get("reminder_minutes") or 10080
    amber_minutes = member.get("escalation_minutes_amber") or 14400
    red_minutes = member.get("escalation_minutes_red") or 20160
    if concern >= 7 or minutes_silent >= red_minutes:
        status = "Red"
        next_action = "Call this person and nudge the assigned relative"
    elif concern >= 4 or minutes_silent >= amber_minutes:
        status = "Amber"
        next_action = "Ask the assigned relative to check in"
    elif minutes_silent >= reminder_minutes:
        status = "Reminder"
        next_action = "Send this person a check-in reminder link"
    else:
        status = "Green"
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
        "Last summary": member.get("last_summary") or "No check-in yet",
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
          a.priority ASC,
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
        return "No elder found."
    if not contact:
        return f"No care contact assigned for {elder['name']}. Add an affiliation in Members."
    role = (contact.get("care_role") or "family").replace("_", " ")
    return (
        f"WhatsApp draft to {contact['name']} ({role}): Hi {contact['name']}, we have not heard from "
        f"{elder['name']} in {elder.get('location_city') or 'their town'} recently. Could you check on them today "
        f"and report back through their Adwuma Pa link?"
    )


def checkin_link_for_request(request_id: str) -> str:
    request = db.one("SELECT token FROM checkup_requests WHERE id = ?", (request_id,))
    if not request:
        return ""
    return f"/checkin/{request['token']}"


def scan_silence() -> list[str]:
    actions: list[str] = []
    members = db.rows(
        """
        SELECT m.*,
               c.submitted_at AS last_checkin_at,
               c.concern_level AS last_concern
        FROM members m
        LEFT JOIN checkins c ON c.id = (
          SELECT id FROM checkins WHERE member_id = m.id ORDER BY submitted_at DESC LIMIT 1
        )
        WHERE m.active = 1
        ORDER BY m.name ASC
        """
    )
    for member in members:
        silent = minutes_since(member.get("last_checkin_at"))
        amber = member.get("escalation_minutes_amber") or 14400
        red = member.get("escalation_minutes_red") or 20160
        contact = route_contact(member["id"])
        reminder = member.get("reminder_minutes") or 10080
        if silent >= red:
            silent_text = human_duration(silent)
            red_text = human_duration(red)
            alert_id = db.create_alert(
                member["id"],
                "red_silence",
                f"No check-in for {silent_text}. Red threshold is {red_text}.",
            )
            request_id = db.create_checkup_request(
                member["id"],
                "red_silence",
                f"We have not heard from {member['name']} for {silent_text}, which is beyond the red threshold of {red_text}.",
                channel="whatsapp",
                priority="red",
                related_alert_id=alert_id,
            )
            if contact:
                nudge_id = db.add_nudge(member["id"], contact["id"])
                db.create_checkup_request(
                    member["id"],
                    "first_party_red_silence",
                    f"Ask {contact['name']} ({contact.get('care_role', 'family')}) to check on {member['name']} after red silence.",
                    request_type="field_report",
                    channel="whatsapp",
                    requester="Adwuma Pa autopilot",
                    priority="red",
                    related_alert_id=alert_id,
                    related_nudge_id=nudge_id,
                )
            routed = f" Routed to {contact['name']}." if contact else " No care contact assigned."
            actions.append(f"RED {member['name']}: alert {alert_id}. Request {checkin_link_for_request(request_id)} queued.{routed}")
        elif silent >= amber:
            silent_text = human_duration(silent)
            amber_text = human_duration(amber)
            alert_id = db.create_alert(
                member["id"],
                "amber_silence",
                f"No check-in for {silent_text}. Amber threshold is {amber_text}.",
            )
            request_id = db.create_checkup_request(
                member["id"],
                "amber_silence",
                f"We have not heard from {member['name']} for {silent_text}, which is beyond the amber threshold of {amber_text}.",
                channel="whatsapp",
                priority="amber",
                related_alert_id=alert_id,
            )
            if contact:
                nudge_id = db.add_nudge(member["id"], contact["id"])
                db.create_checkup_request(
                    member["id"],
                    "first_party_amber_silence",
                    f"Ask {contact['name']} ({contact.get('care_role', 'family')}) to check on {member['name']} after amber silence.",
                    request_type="field_report",
                    channel="whatsapp",
                    requester="Adwuma Pa autopilot",
                    priority="amber",
                    related_alert_id=alert_id,
                    related_nudge_id=nudge_id,
                )
            routed = f" Routed to {contact['name']}." if contact else " No care contact assigned."
            actions.append(f"AMBER {member['name']}: alert {alert_id}. Request {checkin_link_for_request(request_id)} queued.{routed}")
        elif silent >= reminder:
            silent_text = human_duration(silent)
            reminder_text = human_duration(reminder)
            alert_id = db.create_alert(
                member["id"],
                "reminder_silence",
                f"No check-in for {silent_text}. Reminder threshold is {reminder_text}.",
            )
            request_id = db.create_checkup_request(
                member["id"],
                "reminder_silence",
                f"We have not heard from {member['name']} for {silent_text}, so Adwuma Pa is sending a reminder.",
                channel="whatsapp",
                priority="routine",
                related_alert_id=alert_id,
            )
            actions.append(f"REMINDER {member['name']}: alert {alert_id}. Request {checkin_link_for_request(request_id)} queued.")
    if not actions:
        actions.append("No silence escalations. All active family members are inside their configured thresholds.")
    return actions
