from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("ADWUMA_DATA_DIR", "/data" if os.getenv("SPACE_ID") else str(ROOT / "data")))
DB_PATH = DATA_DIR / "adwuma_pa.sqlite3"
SCHEMA_PATH = ROOT / "db" / "schema.sql"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_phone(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = raw.replace("whatsapp:", "").strip()
    if raw.startswith("+"):
        digits = "+" + re.sub(r"\D", "", raw)
    elif raw.startswith("00"):
        digits = "+" + re.sub(r"\D", "", raw[2:])
    else:
        clean = re.sub(r"\D", "", raw)
        if clean.startswith("0") and len(clean) == 10:
            digits = f"+233{clean[1:]}"
        elif clean.startswith("233"):
            digits = f"+{clean}"
        else:
            digits = f"+{clean}" if clean else ""
    return digits


def normalize_whatsapp(value: str | None) -> str:
    phone = normalize_phone(value)
    return f"whatsapp:{phone}" if phone else ""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        migrate_schema(conn)
    migrate_autopilot_defaults()


def migrate_schema(conn: sqlite3.Connection) -> None:
    ensure_columns(
        conn,
        "members",
        {
            "reminder_minutes": "INTEGER DEFAULT 10080",
            "escalation_minutes_amber": "INTEGER DEFAULT 14400",
            "escalation_minutes_red": "INTEGER DEFAULT 20160",
            "routine_messages_per_day": "INTEGER DEFAULT 1",
            "amber_messages_per_day": "INTEGER DEFAULT 1",
            "red_messages_per_day": "INTEGER DEFAULT 2",
            "call_enabled": "INTEGER DEFAULT 1",
            "family_role": "TEXT DEFAULT 'relative'",
            "is_coordinator": "INTEGER DEFAULT 0",
        },
    )
    ensure_columns(
        conn,
        "checkins",
        {
            "request_id": "TEXT REFERENCES checkup_requests(id)",
            "translation": "TEXT",
            "analysis_status": "TEXT DEFAULT 'needs_review'",
            "analysis_json": "TEXT",
            "processing_error": "TEXT",
        },
    )
    ensure_columns(conn, "nudges", {"request_id": "TEXT REFERENCES checkup_requests(id)"})
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
          key TEXT PRIMARY KEY,
          value TEXT,
          updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS autopilot_runs (
          id TEXT PRIMARY KEY,
          started_at TEXT NOT NULL,
          completed_at TEXT,
          actor TEXT,
          status TEXT NOT NULL,
          reason TEXT,
          actions_json TEXT DEFAULT '[]',
          deliveries_json TEXT DEFAULT '[]',
          settings_json TEXT DEFAULT '{}',
          error TEXT
        )
        """
    )


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def seed_demo_data() -> None:
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM members").fetchone()["n"]
        if count:
            return
        members = [
            ("elder_kwame", "Uncle Kwame", "+233000000001", "whatsapp:+233000000001", "Obuasi", "Ashanti", "twi", 1),
            ("elder_esi", "Auntie Esi", "+233000000002", "whatsapp:+233000000002", "Cape Coast", "Central", "fat", 1),
            ("elder_akua", "Auntie Akua", "+233000000003", "whatsapp:+233000000003", "Kumasi", "Ashanti", "twi", 1),
            ("contact_ama", "Ama", "+233000000004", "whatsapp:+233000000004", "Obuasi", "Ashanti", "eng", 0),
            ("contact_kojo", "Kojo", "+233000000005", "whatsapp:+233000000005", "Cape Coast", "Central", "eng", 0),
        ]
        for member in members:
            conn.execute(
                """
                INSERT INTO members
                (id, name, phone, whatsapp, location_city, location_region, language, call_enabled,
                 checkin_url_token, reminder_minutes, escalation_minutes_amber, escalation_minutes_red, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*member, member[0].replace("_", "-"), 10080, 14400, 20160, now_iso()),
            )
        conn.execute(
            "INSERT INTO first_party_contacts (id, elder_id, contact_id, priority) VALUES (?, ?, ?, ?)",
            (new_id("fpc"), "elder_kwame", "contact_ama", 1),
        )
        conn.execute(
            "INSERT INTO first_party_contacts (id, elder_id, contact_id, priority) VALUES (?, ?, ?, ?)",
            (new_id("fpc"), "elder_esi", "contact_kojo", 1),
        )
    add_checkin(
        "elder_kwame",
        "self",
        "Me ho ye, nanso me nantew kakra nnansa yi.",
        "Uncle says he is okay but has walked less over the last few days.",
        4,
        ["reduced_mobility"],
        "twi",
    )
    add_checkin(
        "elder_esi",
        "self",
        "I feel fine. Kojo came by yesterday.",
        "Auntie Esi reports feeling fine and had a recent visit.",
        1,
        [],
        "eng",
    )


def clear_all_data() -> None:
    with connect() as conn:
        conn.execute("UPDATE checkup_requests SET related_alert_id = NULL, related_nudge_id = NULL")
        conn.execute("UPDATE nudges SET request_id = NULL, checkin_id = NULL")
        for table in [
            "inbound_messages",
            "outbound_messages",
            "autopilot_runs",
            "model_runs",
            "nudges",
            "calls",
            "checkins",
            "checkup_requests",
            "alerts",
            "member_affiliations",
            "first_party_contacts",
            "members",
        ]:
            conn.execute(f"DELETE FROM {table}")


def rows(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def add_member(
    name: str,
    phone: str,
    whatsapp: str,
    city: str,
    region: str,
    language: str,
    call_enabled: bool = True,
    family_role: str = "relative",
    is_coordinator: bool = False,
) -> str:
    if family_role == "coordinator":
        is_coordinator = True
    phone = normalize_phone(phone)
    whatsapp = normalize_whatsapp(whatsapp or phone)
    member_id = new_id("member")
    token = f"{name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:5]}"
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO members
            (id, name, phone, whatsapp, location_city, location_region, language, family_role,
             is_coordinator, checkin_url_token, reminder_minutes, escalation_minutes_amber,
             escalation_minutes_red, routine_messages_per_day, amber_messages_per_day,
             red_messages_per_day, call_enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                member_id,
                name,
                phone,
                whatsapp,
                city,
                region,
                language,
                family_role,
                int(is_coordinator),
                token,
                10080,
                14400,
                20160,
                1,
                1,
                2,
                int(call_enabled),
                now_iso(),
            ),
        )
    return member_id


def update_member(
    member_id: str,
    name: str,
    phone: str,
    whatsapp: str,
    city: str,
    region: str,
    language: str,
    call_enabled: bool = True,
    family_role: str = "relative",
    is_coordinator: bool = False,
) -> None:
    if family_role == "coordinator":
        is_coordinator = True
    with connect() as conn:
        conn.execute(
            """
            UPDATE members
            SET name = ?,
                phone = ?,
                whatsapp = ?,
                location_city = ?,
                location_region = ?,
                language = ?,
                family_role = ?,
                is_coordinator = ?,
                call_enabled = ?
            WHERE id = ?
            """,
            (
                name,
                normalize_phone(phone),
                normalize_whatsapp(whatsapp or phone),
                city,
                region,
                language,
                family_role,
                int(is_coordinator),
                int(call_enabled),
                member_id,
            ),
        )


def update_member_role(member_id: str, family_role: str, is_coordinator: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE members SET family_role = ?, is_coordinator = ? WHERE id = ?",
            (family_role, int(is_coordinator), member_id),
        )


def add_affiliation(
    subject_member_id: str,
    related_member_id: str,
    relationship: str,
    care_role: str,
    priority: int = 5,
    can_coordinate: bool = False,
    notes: str = "",
) -> str:
    if subject_member_id == related_member_id:
        raise ValueError("A member cannot be affiliated with themselves.")
    affiliation_id = new_id("affil")
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO member_affiliations
            (id, subject_member_id, related_member_id, relationship, care_role, priority,
             can_coordinate, notes, created_at)
            VALUES (
              COALESCE((SELECT id FROM member_affiliations
                        WHERE subject_member_id = ? AND related_member_id = ? AND relationship = ?), ?),
              ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                subject_member_id,
                related_member_id,
                relationship,
                affiliation_id,
                subject_member_id,
                related_member_id,
                relationship,
                care_role,
                max(1, int(priority or 5)),
                int(can_coordinate),
                notes,
                now_iso(),
            ),
        )
        if care_role in {"first_party_contact", "nearby_relative", "emergency_contact"}:
            existing = conn.execute(
                """
                SELECT id FROM first_party_contacts
                WHERE elder_id = ? AND contact_id = ?
                LIMIT 1
                """,
                (subject_member_id, related_member_id),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE first_party_contacts SET priority = ? WHERE id = ?",
                    (max(1, int(priority or 5)), existing["id"]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO first_party_contacts (id, elder_id, contact_id, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    (new_id("fpc"), subject_member_id, related_member_id, max(1, int(priority or 5))),
                )
    return affiliation_id


def affiliation_rows(member_id: str | None = None) -> list[dict[str, Any]]:
    where = ""
    params: tuple[Any, ...] = ()
    if member_id:
        where = "WHERE a.subject_member_id = ? OR a.related_member_id = ?"
        params = (member_id, member_id)
    return rows(
        f"""
        SELECT a.id AS Affiliation,
               s.name AS Subject,
               r.name AS Related,
               a.relationship AS Relationship,
               a.care_role AS "Care role",
               a.priority AS Priority,
               CASE WHEN a.can_coordinate = 1 THEN 'Yes' ELSE 'No' END AS Coordinator,
               COALESCE(a.notes, '') AS Notes
        FROM member_affiliations a
        JOIN members s ON s.id = a.subject_member_id
        JOIN members r ON r.id = a.related_member_id
        {where}
        ORDER BY a.priority ASC, s.name ASC, r.name ASC
        """,
        params,
    )


def update_escalation(
    member_id: str,
    reminder_minutes: int,
    amber_minutes: int,
    red_minutes: int,
    routine_messages_per_day: int = 1,
    amber_messages_per_day: int = 1,
    red_messages_per_day: int = 2,
) -> None:
    reminder_minutes = max(1, int(reminder_minutes))
    amber_minutes = max(1, int(amber_minutes))
    if amber_minutes <= reminder_minutes:
        amber_minutes = reminder_minutes + 1
    red_minutes = max(amber_minutes + 1, int(red_minutes))
    routine_messages_per_day = max(0, int(routine_messages_per_day or 0))
    amber_messages_per_day = max(0, int(amber_messages_per_day or 0))
    red_messages_per_day = max(0, int(red_messages_per_day or 0))
    with connect() as conn:
        conn.execute(
            """
            UPDATE members
            SET reminder_minutes = ?, escalation_minutes_amber = ?, escalation_minutes_red = ?,
                routine_messages_per_day = ?, amber_messages_per_day = ?, red_messages_per_day = ?
            WHERE id = ?
            """,
            (
                reminder_minutes,
                amber_minutes,
                red_minutes,
                routine_messages_per_day,
                amber_messages_per_day,
                red_messages_per_day,
                member_id,
            ),
        )


def storage_status() -> dict[str, Any]:
    with connect() as conn:
        member_count = conn.execute("SELECT COUNT(*) AS n FROM members").fetchone()["n"]
        affiliation_count = conn.execute("SELECT COUNT(*) AS n FROM member_affiliations").fetchone()["n"]
        request_count = conn.execute("SELECT COUNT(*) AS n FROM checkup_requests").fetchone()["n"]
        outbound_count = conn.execute("SELECT COUNT(*) AS n FROM outbound_messages").fetchone()["n"]
        autopilot_run_count = conn.execute("SELECT COUNT(*) AS n FROM autopilot_runs").fetchone()["n"]
    return {
        "db_path": str(DB_PATH),
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists(),
        "db_exists": DB_PATH.exists(),
        "persistent_storage": DATA_DIR == Path("/data"),
        "member_count": member_count,
        "affiliation_count": affiliation_count,
        "request_count": request_count,
        "outbound_count": outbound_count,
        "autopilot_run_count": autopilot_run_count,
    }


def get_setting(key: str, default: Any = None) -> Any:
    row = one("SELECT value FROM app_settings WHERE key = ?", (key,))
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]


def set_setting(key: str, value: Any) -> None:
    encoded = json.dumps(value)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, encoded, now_iso()),
        )


def autopilot_settings() -> dict[str, Any]:
    return {
        "enabled": bool(get_setting("autopilot.enabled", False)),
        "scan_interval_minutes": int(get_setting("autopilot.scan_interval_minutes", 30)),
        "send_whatsapp": bool(get_setting("autopilot.send_whatsapp", False)),
        "excluded_member_ids": list(get_setting("autopilot.excluded_member_ids", [])),
        "last_scan_at": get_setting("autopilot.last_scan_at", None),
        "last_scan_result": get_setting("autopilot.last_scan_result", []),
    }


def save_autopilot_settings(
    enabled: bool,
    scan_interval_minutes: int,
    send_whatsapp: bool,
    excluded_member_ids: list[str] | None = None,
) -> dict[str, Any]:
    interval = max(1, int(scan_interval_minutes or 30))
    set_setting("autopilot.enabled", bool(enabled))
    set_setting("autopilot.scan_interval_minutes", interval)
    set_setting("autopilot.send_whatsapp", bool(send_whatsapp))
    set_setting("autopilot.excluded_member_ids", list(excluded_member_ids or []))
    return autopilot_settings()


def migrate_autopilot_defaults() -> None:
    marker = get_setting("autopilot.migrated_interval_30", False)
    if marker:
        return
    current = get_setting("autopilot.scan_interval_minutes", None)
    if current in (None, 15):
        set_setting("autopilot.scan_interval_minutes", 30)
    set_setting("autopilot.migrated_interval_30", True)


def add_autopilot_run(
    actor: str,
    status: str,
    reason: str = "",
    actions: list[Any] | None = None,
    deliveries: list[Any] | None = None,
    settings: dict[str, Any] | None = None,
    error: str = "",
    started_at: str | None = None,
) -> str:
    run_id = new_id("autorun")
    now = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO autopilot_runs
            (id, started_at, completed_at, actor, status, reason, actions_json, deliveries_json, settings_json, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                started_at or now,
                now,
                actor,
                status,
                reason,
                json.dumps(actions or []),
                json.dumps(deliveries or []),
                json.dumps(settings or {}),
                error,
            ),
        )
    return run_id


def autopilot_run_rows(limit: int = 20) -> list[dict[str, Any]]:
    rows_ = rows(
        """
        SELECT started_at, completed_at, actor, status, reason, actions_json, deliveries_json, error
        FROM autopilot_runs
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    output = []
    for row in rows_:
        actions = _decode_json_list(row.get("actions_json"))
        deliveries = _decode_json_list(row.get("deliveries_json"))
        output.append(
            {
                "Started": row.get("started_at") or "",
                "Actor": row.get("actor") or "",
                "Status": row.get("status") or "",
                "Reason": row.get("reason") or row.get("error") or "",
                "Actions": len([item for item in actions if not _is_noop_action(item)]),
                "Deliveries": len([item for item in deliveries if not _is_noop_delivery(item)]),
                "Details": _compact_run_details(actions, deliveries, row.get("error") or ""),
            }
        )
    return output


def _decode_json_list(value: Any) -> list[Any]:
    try:
        decoded = json.loads(value or "[]")
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def _is_noop_action(value: Any) -> bool:
    text = str(value)
    return (
        text.startswith("No silence escalations")
        or text.startswith("Excluded from autopilot")
        or text.startswith("Recently closed care loop")
    )


def _is_noop_delivery(value: Any) -> bool:
    text = str(value)
    return text.startswith("No pending autopilot WhatsApp messages") or "Frequency cap reached" in text


def _compact_run_details(actions: list[Any], deliveries: list[Any], error: str) -> str:
    parts = []
    if error:
        parts.append(f"Error: {error}")
    meaningful_actions = [str(item) for item in actions if not _is_noop_action(item)]
    meaningful_deliveries = [str(item) for item in deliveries if not _is_noop_delivery(item)]
    if meaningful_actions:
        parts.append("Actions: " + " | ".join(meaningful_actions[:3]))
    elif actions:
        parts.append(str(actions[0]))
    if meaningful_deliveries:
        parts.append("Delivery: " + " | ".join(meaningful_deliveries[:3]))
    elif deliveries:
        parts.append(str(deliveries[0]))
    return "\n".join(parts)


def create_alert(member_id: str, alert_type: str, notes: str) -> str:
    existing = one(
        """
        SELECT id FROM alerts
        WHERE member_id = ? AND alert_type = ? AND resolved = 0
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (member_id, alert_type),
    )
    if existing:
        with connect() as conn:
            conn.execute(
                "UPDATE alerts SET notes = ?, created_at = ? WHERE id = ?",
                (notes, now_iso(), existing["id"]),
            )
        return existing["id"]
    alert_id = new_id("alert")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO alerts (id, member_id, alert_type, created_at, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (alert_id, member_id, alert_type, now_iso(), notes),
        )
    return alert_id


def add_nudge(elder_id: str, contact_id: str | None) -> str:
    existing = one(
        """
        SELECT id FROM nudges
        WHERE elder_id = ? AND COALESCE(contact_id, '') = COALESCE(?, '') AND responded_at IS NULL
        ORDER BY sent_at DESC
        LIMIT 1
        """,
        (elder_id, contact_id),
    )
    if existing:
        return existing["id"]
    nudge_id = new_id("nudge")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO nudges (id, elder_id, contact_id, sent_at)
            VALUES (?, ?, ?, ?)
            """,
            (nudge_id, elder_id, contact_id, now_iso()),
        )
    return nudge_id


def age_latest_checkin(member_id: str, minutes_ago: int) -> None:
    checkin = one(
        "SELECT id FROM checkins WHERE member_id = ? ORDER BY submitted_at DESC LIMIT 1",
        (member_id,),
    )
    if not checkin:
        return
    from datetime import timedelta

    aged_at = (datetime.now(timezone.utc) - timedelta(minutes=max(0, int(minutes_ago)))).isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute("UPDATE checkins SET submitted_at = ? WHERE id = ?", (aged_at, checkin["id"]))


def add_checkin(
    member_id: str,
    source: str,
    transcript: str,
    summary: str,
    concern_level: int | None,
    flags: list[str],
    language_detected: str,
    input_type: str = "text",
    raw_input: str | None = None,
    asr_model_used: str | None = None,
    asr_confidence: float | None = None,
    request_id: str | None = None,
    translation: str | None = None,
    analysis_status: str = "needs_review",
    analysis_json: dict[str, Any] | None = None,
    processing_error: str | None = None,
) -> str:
    checkin_id = new_id("checkin")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO checkins
            (id, member_id, request_id, submitted_at, input_type, raw_input, transcript, translation,
             analysis_status, analysis_json, processing_error, asr_model_used, asr_confidence,
             summary, concern_level, flags, language_detected, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkin_id,
                member_id,
                request_id,
                now_iso(),
                input_type,
                raw_input or transcript,
                transcript,
                translation,
                analysis_status,
                json.dumps(analysis_json or {}),
                processing_error,
                asr_model_used,
                asr_confidence,
                summary,
                concern_level,
                json.dumps(flags),
                language_detected,
                source,
            ),
        )
        if request_id:
            status = "complete" if analysis_status == "complete" else "needs_review"
            conn.execute(
                "UPDATE checkup_requests SET status = ?, completed_at = ? WHERE id = ?",
                (status, now_iso(), request_id),
            )
            request = conn.execute(
                "SELECT related_nudge_id FROM checkup_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if request and request["related_nudge_id"]:
                conn.execute(
                    "UPDATE nudges SET responded_at = ?, checkin_id = ? WHERE id = ?",
                    (now_iso(), checkin_id, request["related_nudge_id"]),
                )
    if concern_level is not None:
        maybe_create_concern_alert(
            member_id,
            concern_level,
            summary=summary,
            flags=flags,
            transcript=transcript,
            translation=translation,
            analysis_json=analysis_json,
        )
    elif analysis_status == "needs_review":
        create_alert(member_id, "needs_review", processing_error or "Check-in requires human review.")
    return checkin_id


def maybe_create_concern_alert(
    member_id: str,
    concern_level: int,
    summary: str = "",
    flags: list[str] | None = None,
    transcript: str = "",
    translation: str | None = None,
    analysis_json: dict[str, Any] | None = None,
) -> None:
    if concern_level < 4:
        return
    alert_type = "red_concern" if concern_level >= 7 else "amber_concern"
    create_alert(member_id, alert_type, concern_alert_notes(concern_level, summary, flags or [], transcript, translation, analysis_json or {}))


def concern_alert_notes(
    concern_level: int,
    summary: str,
    flags: list[str],
    transcript: str,
    translation: str | None,
    analysis_json: dict[str, Any],
) -> str:
    evidence = analysis_json.get("evidence") or []
    recommendation = analysis_json.get("recommended_action") or ""
    confidence = analysis_json.get("confidence") or ""
    lines = [f"Concern score {concern_level} from latest check-in."]
    if summary:
        lines.append(f"Summary: {summary}")
    if evidence:
        lines.append("Evidence: " + "; ".join(str(item) for item in evidence[:3]))
    if flags:
        lines.append("Flags: " + ", ".join(str(flag) for flag in flags[:5]))
    if recommendation:
        lines.append(f"Recommended action: {recommendation}")
    if confidence:
        lines.append(f"Confidence: {confidence}")
    excerpt = (translation or transcript or "").strip()
    if excerpt:
        lines.append(f"Reply excerpt: {excerpt[:220]}")
    return "\n".join(lines)


def create_checkup_request(
    member_id: str,
    reason_code: str,
    reason_detail: str,
    request_type: str = "elder_checkin",
    channel: str = "web",
    requester: str = "Ani Kɛse autopilot",
    priority: str = "routine",
    related_alert_id: str | None = None,
    related_nudge_id: str | None = None,
    expires_minutes: int | None = 4320,
) -> str:
    existing = one(
        """
        SELECT id FROM checkup_requests
        WHERE member_id = ? AND reason_code = ? AND request_type = ? AND status IN ('pending', 'sent', 'processing', 'needs_review')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (member_id, reason_code, request_type),
    )
    if existing:
        return existing["id"]
    request_id = new_id("request")
    token = f"{request_id[8:]}-{uuid.uuid4().hex[:8]}"
    expires_at = None
    if expires_minutes:
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=int(expires_minutes))).isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO checkup_requests
            (id, token, member_id, requester, request_type, reason_code, reason_detail, channel,
             status, priority, created_at, expires_at, related_alert_id, related_nudge_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                token,
                member_id,
                requester,
                request_type,
                reason_code,
                reason_detail,
                channel,
                "pending",
                priority,
                now_iso(),
                expires_at,
                related_alert_id,
                related_nudge_id,
            ),
        )
    return request_id


def open_checkup_request(member_id: str, reason_code: str, request_type: str = "elder_checkin") -> dict[str, Any] | None:
    return one(
        """
        SELECT * FROM checkup_requests
        WHERE member_id = ? AND reason_code = ? AND request_type = ?
          AND status IN ('pending', 'sent', 'processing', 'needs_review')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (member_id, reason_code, request_type),
    )


def get_request_by_token(token: str) -> dict[str, Any] | None:
    return one(
        """
        SELECT r.*, m.name AS member_name, m.language, m.location_city, m.location_region,
               m.whatsapp, m.phone
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        WHERE r.token = ?
        """,
        (token,),
    )


def request_rows(limit: int = 30) -> list[dict[str, Any]]:
    return rows(
        """
        SELECT r.id AS Request, r.token AS Token, m.name AS Member, r.request_type AS Type,
               r.reason_code AS Reason, r.priority AS Priority, r.status AS Status,
               r.created_at AS Created, COALESCE(r.completed_at, '') AS Completed
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        ORDER BY
          CASE r.status
            WHEN 'pending' THEN 0
            WHEN 'sent' THEN 1
            WHEN 'needs_review' THEN 2
            WHEN 'processing' THEN 3
            ELSE 4
          END,
          r.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def pending_request_for_member(member_id: str, reason_code: str) -> dict[str, Any] | None:
    return one(
        """
        SELECT * FROM checkup_requests
        WHERE member_id = ? AND reason_code = ? AND status IN ('pending', 'sent', 'processing', 'needs_review')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (member_id, reason_code),
    )


def outbound_count_for_member_priority(
    member_id: str,
    priority: str,
    since_iso: str,
    request_type: str | None = None,
    recent_attempt_since_iso: str | None = None,
) -> int:
    type_filter = "AND r.request_type = ?" if request_type else ""
    params_list: list[Any] = [member_id, priority, since_iso]
    if request_type:
        params_list.append(request_type)
    params_list.append(recent_attempt_since_iso or since_iso)
    row = one(
        f"""
        SELECT COUNT(*) AS n
        FROM outbound_messages o
        JOIN checkup_requests r ON r.id = o.request_id
        WHERE r.member_id = ?
          AND r.priority = ?
          AND o.created_at >= ?
          {type_filter}
          AND (
            COALESCE(o.status, '') IN ('delivered', 'read')
            OR (
              COALESCE(o.status, '') IN ('sent', 'queued')
              AND o.created_at >= ?
            )
          )
        """,
        tuple(params_list),
    )
    return int(row["n"] if row else 0)


def outbound_count_for_recipient_priority(
    recipient_member_id: str,
    priority: str,
    since_iso: str,
    request_type: str | None = None,
    recent_attempt_since_iso: str | None = None,
) -> int:
    type_filter = "AND r.request_type = ?" if request_type else ""
    params_list: list[Any] = [recipient_member_id, priority, since_iso]
    if request_type:
        params_list.append(request_type)
    params_list.append(recent_attempt_since_iso or since_iso)
    row = one(
        f"""
        SELECT COUNT(*) AS n
        FROM outbound_messages o
        JOIN checkup_requests r ON r.id = o.request_id
        WHERE o.recipient_member_id = ?
          AND r.priority = ?
          AND o.created_at >= ?
          {type_filter}
          AND (
            COALESCE(o.status, '') IN ('delivered', 'read')
            OR (
              COALESCE(o.status, '') IN ('sent', 'queued')
              AND o.created_at >= ?
            )
          )
        """,
        tuple(params_list),
    )
    return int(row["n"] if row else 0)


def member_frequency_cap(member_id: str, priority: str) -> int:
    member = one(
        """
        SELECT routine_messages_per_day, amber_messages_per_day, red_messages_per_day
        FROM members
        WHERE id = ?
        """,
        (member_id,),
    )
    if not member:
        return 0
    if priority == "red":
        return int(member.get("red_messages_per_day") or 0)
    if priority == "amber":
        return int(member.get("amber_messages_per_day") or 0)
    return int(member.get("routine_messages_per_day") or 0)


def record_model_run(
    checkin_id: str | None,
    run_type: str,
    model_id: str | None,
    status: str,
    input_summary: str = "",
    output_json: dict[str, Any] | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
) -> str:
    run_id = new_id("run")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO model_runs
            (id, checkin_id, run_type, model_id, status, latency_ms, input_summary, output_json, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                checkin_id,
                run_type,
                model_id,
                status,
                latency_ms,
                input_summary,
                json.dumps(output_json or {}),
                error,
                now_iso(),
            ),
        )
    return run_id


def add_inbound_message(
    sender: str,
    body: str,
    channel: str = "whatsapp",
    matched_member_id: str | None = None,
    matched_contact_id: str | None = None,
    status: str = "unmatched",
) -> str:
    message_id = new_id("inbound")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO inbound_messages
            (id, sender, channel, body, matched_member_id, matched_contact_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, sender, channel, body, matched_member_id, matched_contact_id, status, now_iso()),
        )
    return message_id


def add_outbound_message(
    request_id: str | None,
    recipient_member_id: str | None,
    recipient: str,
    body: str,
    status: str,
    channel: str = "whatsapp",
    provider_sid: str | None = None,
    error: str | None = None,
) -> str:
    message_id = new_id("outbound")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO outbound_messages
            (id, request_id, recipient_member_id, channel, recipient, body, provider_sid, status, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, request_id, recipient_member_id, channel, recipient, body, provider_sid, status, error, now_iso()),
        )
        if request_id and status in {"sent", "queued"}:
            conn.execute("UPDATE checkup_requests SET status = 'sent' WHERE id = ?", (request_id,))
    return message_id


def update_outbound_status(provider_sid: str, status: str, error: str | None = None) -> None:
    if not provider_sid:
        return
    with connect() as conn:
        conn.execute(
            """
            UPDATE outbound_messages
            SET status = ?,
                error = COALESCE(?, error)
            WHERE provider_sid = ?
            """,
            (status, error, provider_sid),
        )


def update_checkin_translation(checkin_id: str, translation: str, reviewer: str = "Coordinator") -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE checkins
            SET translation = ?,
                analysis_status = 'needs_review',
                concern_level = NULL,
                processing_error = ?
            WHERE id = ?
            """,
            (
                translation,
                f"English translation corrected by {reviewer}. Re-run analysis before using an automated concern score.",
                checkin_id,
            ),
        )


def outbound_rows(limit: int = 20) -> list[dict[str, Any]]:
    return rows(
        """
        SELECT o.created_at AS Created,
               COALESCE(m.name, 'Unknown') AS Recipient,
               o.channel AS Channel,
               o.status AS Status,
               COALESCE(o.provider_sid, '') AS SID,
               COALESCE(o.error, '') AS Error,
               o.body AS Body
        FROM outbound_messages o
        LEFT JOIN members m ON m.id = o.recipient_member_id
        ORDER BY o.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def resolve_alert(alert_id: str, resolved_by: str, notes: str) -> dict[str, Any] | None:
    with connect() as conn:
        alert = conn.execute("SELECT id, member_id FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if not alert:
            return None
        conn.execute(
            """
            UPDATE alerts
            SET resolved = 1, resolved_at = ?, resolved_by = ?, notes = COALESCE(notes, '') || CHAR(10) || ?
            WHERE id = ?
            """,
            (now_iso(), resolved_by, notes, alert_id),
        )
        open_requests = conn.execute(
            """
            SELECT id, related_nudge_id
            FROM checkup_requests
            WHERE status IN ('pending', 'sent', 'processing', 'needs_review')
              AND (
                related_alert_id = ?
                OR (
                  related_alert_id IS NULL
                  AND member_id = ?
                  AND requester IN ('Ani Kɛse autopilot', 'Adwuma Pa autopilot')
                )
              )
            """,
            (alert_id, alert["member_id"]),
        ).fetchall()
        linked_nudges = conn.execute(
            """
            SELECT DISTINCT related_nudge_id
            FROM checkup_requests
            WHERE related_nudge_id IS NOT NULL
              AND (
                related_alert_id = ?
                OR (
                  related_alert_id IS NULL
                  AND member_id = ?
                  AND requester IN ('Ani Kɛse autopilot', 'Adwuma Pa autopilot')
                )
              )
            """,
            (alert_id, alert["member_id"]),
        ).fetchall()
        request_ids = [row["id"] for row in open_requests]
        nudge_ids = [row["related_nudge_id"] for row in linked_nudges]
        if request_ids:
            placeholders = ",".join("?" for _ in request_ids)
            conn.execute(
                f"UPDATE checkup_requests SET status = 'complete' WHERE id IN ({placeholders})",
                tuple(request_ids),
            )
        if nudge_ids:
            placeholders = ",".join("?" for _ in nudge_ids)
            conn.execute(
                f"UPDATE nudges SET responded_at = COALESCE(responded_at, ?) WHERE id IN ({placeholders})",
                (now_iso(), *nudge_ids),
            )
        return {"alert_id": alert_id, "closed_requests": len(request_ids), "closed_nudges": len(nudge_ids)}
