from __future__ import annotations

import html
import json
import os
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, Request, Response
import gradio as gr

from config.models import ASR_CONFIG, LLM_CONFIG, TRANSLATION_CONFIG, TTS_CONFIG, total_parameter_budget_b
from db import database as db
from services.relay import dashboard_rows, scan_silence, simulate_nudge
from services import modal_client, pipeline, twilio_client

os.environ.setdefault("GRADIO_SSR_MODE", "False")

FAMILY_HEADERS = [
    "Name",
    "City",
    "Region",
    "Language",
    "Status",
    "Concern",
    "Minutes silent",
    "Reminder min",
    "Amber min",
    "Red min",
    "Last summary",
    "Analysis",
    "Next action",
    "Token",
]
ALERT_HEADERS = ["Alert", "Member", "Type", "Created", "State", "Notes"]
OPEN_LOOP_HEADERS = ["Member", "Type", "Created", "Notes"]
CHECKIN_HEADERS = ["Submitted", "Source", "Input", "Status", "Concern", "Summary", "Translation", "Transcript", "Error"]
REQUEST_HEADERS = ["Request", "Token", "Member", "Type", "Reason", "Priority", "Status", "Created", "Completed"]
NUDGE_HEADERS = ["Sent", "Contact", "Request", "Responded", "Check-in"]
AFFILIATION_HEADERS = ["Subject", "Related", "Relationship", "Care role", "Priority", "Coordinator", "Notes"]
OUTBOUND_HEADERS = ["Created", "Recipient", "Channel", "Status", "SID", "Error", "Body"]
ASR_MODEL_CHOICES = [
    ("MMS-1B-all (Akan)", "primary"),
    ("Adwuma Pa Akan Whisper fine-tune", "fine_tuned"),
    ("GiftMark Akan Whisper", "fallback"),
]
ROLE_CHOICES = [
    ("Elder / care recipient", "elder"),
    ("Coordinator", "coordinator"),
    ("Relative", "relative"),
    ("Nearby contact", "nearby_contact"),
    ("Caregiver", "caregiver"),
]
RELATIONSHIP_CHOICES = [
    ("Daughter", "daughter"),
    ("Son", "son"),
    ("Mother", "mother"),
    ("Father", "father"),
    ("Spouse", "spouse"),
    ("Sibling", "sibling"),
    ("Auntie", "auntie"),
    ("Uncle", "uncle"),
    ("Niece", "niece"),
    ("Nephew", "nephew"),
    ("Cousin", "cousin"),
    ("Grandchild", "grandchild"),
    ("In-law", "in_law"),
    ("Neighbor", "neighbor"),
    ("Family coordinator", "family_coordinator"),
    ("Caregiver", "caregiver"),
    ("Friend", "friend"),
]
CARE_ROLE_CHOICES = [
    ("Family", "family"),
    ("Primary coordinator", "primary_coordinator"),
    ("Backup coordinator", "backup_coordinator"),
    ("First-party contact", "first_party_contact"),
    ("Nearby relative", "nearby_relative"),
    ("Emergency contact", "emergency_contact"),
    ("Caregiver", "caregiver"),
]
GHANA_REGIONS = [
    "Ahafo",
    "Ashanti",
    "Bono",
    "Bono East",
    "Central",
    "Eastern",
    "Greater Accra",
    "North East",
    "Northern",
    "Oti",
    "Savannah",
    "Upper East",
    "Upper West",
    "Volta",
    "Western",
    "Western North",
]
TTS_PROMPT_TYPES = [
    ("Check-in reminder", "reminder"),
    ("Outbound call greeting", "call_greeting"),
    ("Warm call close", "call_close"),
]
APP_THEME = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    text_size="md",
    spacing_size="md",
    radius_size="sm",
)

CUSTOM_CSS = """
:root {
  --ap-bg: #0f172a;
  --ap-surface: #ffffff;
  --ap-panel: #ffffff;
  --ap-panel-soft: #f8fafc;
  --ap-ink: #0f172a;
  --ap-muted: #334155;
  --ap-border: #94a3b8;
  --ap-palm: #047857;
  --ap-palm-dark: #064e3b;
  --ap-gold: #b45309;
  --ap-clay: #b91c1c;
}
.gradio-container {
  background: #e2e8f0;
  color: var(--ap-ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 1240px !important;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container span,
.gradio-container p {
  color: var(--ap-ink) !important;
}
.ap-header {
  background: #0f172a;
  border-radius: 8px;
  border: 1px solid #1e293b;
  color: #f8fafc;
  margin: 0 0 12px;
  padding: 22px 24px;
}
.ap-title {
  color: #ffffff;
  font-size: 34px;
  line-height: 1.05;
  font-weight: 800;
}
.ap-subtitle {
  color: #cbd5e1;
  font-size: 15px;
  max-width: 760px;
  margin-top: 8px;
}
.ap-pill {
  display: inline-block;
  border: 1px solid #047857;
  background: #ecfdf5;
  border-radius: 6px;
  padding: 6px 10px;
  margin: 4px 6px 12px 0;
  color: #064e3b !important;
  font-size: 13px;
  font-weight: 800;
}
button.primary {
  background: var(--ap-palm-dark) !important;
  border-color: var(--ap-palm-dark) !important;
  color: #ffffff !important;
}
button {
  font-weight: 700 !important;
}
button:not([role="tab"]):not(.primary) {
  background: #ffffff !important;
  border: 1px solid #64748b !important;
  color: #0f172a !important;
}
.ap-note {
  color: var(--ap-muted);
  font-size: 13px;
}
.block,
.form,
.panel {
  background: var(--ap-surface) !important;
  border-color: var(--ap-border) !important;
}
.tabitem,
.block,
.form {
  border-radius: 8px !important;
}
button[role="tab"] {
  color: #0f172a !important;
  background: #cbd5e1 !important;
  border: 1px solid #94a3b8 !important;
  border-radius: 6px !important;
  font-weight: 800 !important;
}
button[role="tab"][aria-selected="true"] {
  color: #ffffff !important;
  background: #0f172a !important;
  border-color: #0f172a !important;
}
.wrap label,
.wrap .label-wrap,
.form label,
.block label {
  color: #0f172a !important;
  font-weight: 800 !important;
  opacity: 1 !important;
}
input,
textarea,
select {
  background: #ffffff !important;
  border-color: #64748b !important;
  color: #0f172a !important;
}
.table-container,
.table-wrap,
.virtual-table-viewport {
  background: #ffffff !important;
  border: 1px solid #64748b !important;
  border-radius: 6px !important;
}
.header-table,
.dataframe table {
  font-size: 13px;
  color: var(--ap-ink) !important;
  background: #ffffff !important;
  border-collapse: collapse !important;
}
.header-cell,
.cell-wrap,
.header-table .header-cell,
.header-table th,
.header-table td,
.dataframe th {
  background: #1e293b !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  border-color: #334155 !important;
}
.header-cell *,
.cell-wrap *,
.header-table th *,
.header-table td *,
.header-content,
.header-content *,
.header-menu,
.header-menu *,
.dataframe th span {
  color: #ffffff !important;
  background: #1e293b !important;
}
.table-container tbody tr,
.table-container tbody td,
.table-container td,
.table-container td *,
.cell,
.cell *,
.dataframe td,
.dataframe td span {
  color: var(--ap-ink) !important;
  background: #ffffff !important;
  border-color: #cbd5e1 !important;
}
.table-container tbody tr:nth-child(even) td,
.table-container tbody tr:nth-child(even) td * {
  background: #f8fafc !important;
}
.table-container .wrap,
.table-container .text,
.table-container span {
  opacity: 1 !important;
}
.ap-status-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  margin: 10px 0 14px;
}
.ap-status-card {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .08);
}
.ap-status-label {
  color: #1e293b !important;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.ap-status-value {
  color: #0f172a !important;
  font-size: 28px;
  font-weight: 800;
  line-height: 1;
  margin-top: 6px;
}
.ap-green { border-left: 6px solid #047857; }
.ap-reminder { border-left: 6px solid #b45309; }
.ap-amber { border-left: 6px solid #d97706; }
.ap-red { border-left: 6px solid #b91c1c; }
.ap-section-title {
  color: #0f172a !important;
  font-size: 18px;
  font-weight: 900;
  margin: 18px 0 8px;
}
.ap-list {
  display: grid;
  gap: 10px;
  margin-bottom: 12px;
}
.ap-item {
  align-items: center;
  background: #ffffff;
  border: 1px solid #94a3b8;
  border-left: 6px solid #047857;
  border-radius: 8px;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  padding: 12px 14px;
}
.ap-item code {
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  color: #0f172a;
  font-size: 12px;
  padding: 7px 8px;
  white-space: nowrap;
}
.ap-item-title {
  color: #0f172a !important;
  font-size: 15px;
  font-weight: 900;
}
.ap-item-meta,
.ap-item-note,
.ap-family-foot {
  color: #334155 !important;
  font-size: 13px;
}
.ap-item-note {
  margin-top: 3px;
}
.ap-red,
.ap-item.ap-red {
  border-left-color: #b91c1c;
}
.ap-amber,
.ap-item.ap-amber {
  border-left-color: #d97706;
}
.ap-routine,
.ap-item.ap-routine {
  border-left-color: #047857;
}
.ap-alert {
  border-left-color: #b45309;
}
.ap-state {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  color: #0f172a !important;
  font-size: 12px;
  font-weight: 800;
  padding: 5px 9px;
  text-transform: uppercase;
}
.ap-family-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  margin-bottom: 12px;
}
.ap-family-card {
  background: #ffffff;
  border: 1px solid #94a3b8;
  border-left: 6px solid #047857;
  border-radius: 8px;
  padding: 12px;
}
.ap-family-top {
  align-items: center;
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
.ap-family-top strong {
  color: #0f172a !important;
  font-size: 15px;
}
.ap-family-top span {
  color: #0f172a !important;
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
}
.ap-empty {
  background: #ffffff;
  border: 1px dashed #94a3b8;
  border-radius: 8px;
  color: #334155 !important;
  padding: 16px;
}
.ap-profile {
  background: #ffffff;
  border: 1px solid #64748b;
  border-left: 6px solid #047857;
  border-radius: 8px;
  color: #0f172a !important;
  padding: 16px;
}
.ap-profile h3 {
  color: #0f172a !important;
  font-size: 22px;
  font-weight: 900;
  margin: 0 0 12px;
}
.ap-profile-grid {
  display: grid;
  gap: 8px 14px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.ap-profile-row {
  color: #0f172a !important;
  font-size: 14px;
}
.ap-profile-row strong,
.ap-profile-section strong {
  color: #0f172a !important;
  font-weight: 900;
}
.ap-profile-section {
  border-top: 1px solid #cbd5e1;
  color: #0f172a !important;
  margin-top: 14px;
  padding-top: 12px;
}
.ap-profile-section ul {
  margin: 8px 0 0 18px;
}
.ap-storage {
  background: #f8fafc;
  border: 1px solid #64748b;
  border-radius: 8px;
  color: #0f172a !important;
  padding: 12px;
}
.ap-storage strong {
  color: #0f172a !important;
}
.gradio-container {
  max-width: 1160px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 18px 18px 32px !important;
}
.tabs {
  gap: 14px !important;
}
.tabitem {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
.block,
.form {
  box-shadow: none !important;
}
.ap-header {
  align-items: flex-start;
  display: flex;
  justify-content: space-between;
  gap: 18px;
}
.ap-title {
  letter-spacing: 0;
}
.ap-subtitle {
  font-size: 16px;
  line-height: 1.5;
}
.ap-header-side {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  color: #0f172a !important;
  min-width: 260px;
  padding: 12px 14px;
}
.ap-header-side strong,
.ap-header-side span {
  color: #0f172a !important;
}
.ap-header-side strong {
  display: block;
  font-size: 12px;
  letter-spacing: .04em;
  margin-bottom: 4px;
  text-transform: uppercase;
}
.ap-hero-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: 1.25fr .75fr;
  margin: 8px 0 14px;
}
.ap-story {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 16px;
}
.ap-story h2 {
  color: #0f172a !important;
  font-size: 22px;
  line-height: 1.2;
  margin: 0 0 10px;
}
.ap-story p {
  color: #334155 !important;
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
}
.ap-flow {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-top: 14px;
}
.ap-flow-step {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 10px;
}
.ap-flow-step strong {
  color: #0f172a !important;
  display: block;
  font-size: 13px;
  margin-bottom: 3px;
}
.ap-flow-step span {
  color: #475569 !important;
  display: block;
  font-size: 12px;
  line-height: 1.35;
}
.ap-recorder {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  color: #f8fafc !important;
  padding: 16px;
}
.ap-recorder strong,
.ap-recorder li,
.ap-recorder p {
  color: #f8fafc !important;
}
.ap-recorder ul {
  margin: 10px 0 0 18px;
  padding: 0;
}
.ap-recorder li {
  font-size: 13px;
  line-height: 1.5;
}
.ap-status-card,
.ap-family-card,
.ap-item,
.ap-profile,
.ap-story,
.ap-storage {
  box-shadow: 0 1px 2px rgba(15, 23, 42, .06);
}
.ap-status-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.ap-section-title {
  align-items: center;
  display: flex;
  gap: 8px;
  letter-spacing: 0;
}
.ap-section-title::before {
  background: #047857;
  border-radius: 999px;
  content: "";
  display: inline-block;
  height: 9px;
  width: 9px;
}
.ap-family-grid {
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.ap-family-card,
.ap-item {
  min-height: 96px;
}
.ap-family-foot,
.ap-item-note,
.ap-item-meta {
  line-height: 1.4;
}
details,
.accordion {
  border-color: #94a3b8 !important;
}
details > summary {
  background: #ffffff !important;
  border: 1px solid #64748b !important;
  border-radius: 6px !important;
  color: #0f172a !important;
  font-weight: 800 !important;
  padding: 10px 12px !important;
}
details[open] > summary {
  background: #f8fafc !important;
  border-bottom-left-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
}
.prose h3,
.markdown h3 {
  color: #0f172a !important;
}
@media (max-width: 820px) {
  .ap-header,
  .ap-hero-grid {
    display: block;
  }
  .ap-header-side,
  .ap-recorder {
    margin-top: 12px;
  }
  .ap-flow,
  .ap-status-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
"""


def refresh_dashboard():
    return (
        status_cards_html(),
        active_requests_html(),
        family_overview_html(),
        care_routes_html(),
        alert_overview_html(),
        modal_health_markdown(),
        model_budget_markdown(),
    )


def table_value(rows, headers):
    return [[row.get(header, "") for header in headers] for row in rows]


def family_table_value():
    return table_value(dashboard_rows(), FAMILY_HEADERS)


def alert_table_value():
    return table_value(alert_rows(), ALERT_HEADERS)


def open_loop_table_value():
    return table_value(open_loop_rows(), OPEN_LOOP_HEADERS)


def request_table_value():
    return table_value(db.request_rows(), REQUEST_HEADERS)


def outbound_table_value():
    return table_value(db.outbound_rows(), OUTBOUND_HEADERS)


def storage_status_html():
    status = db.storage_status()
    persistence = "persistent /data storage detected" if status["persistent_storage"] else "ephemeral app filesystem"
    warning = (
        "Records should survive Space restarts."
        if status["persistent_storage"]
        else "Records can disappear when the Space rebuilds or restarts. Attach HF persistent storage or external DB before real use."
    )
    return f"""
<div class="ap-storage">
  <strong>Storage:</strong> {html.escape(persistence)}<br>
  <strong>Members saved:</strong> {status['member_count']}<br>
  <strong>Check-ins queued:</strong> {status['request_count']}<br>
  <strong>Database:</strong> <code>{html.escape(status['db_path'])}</code><br>
  {html.escape(warning)}
</div>
"""


def demo_story_html():
    return """
<section>
  <div class="ap-story">
    <h2>Autopilot keeps the family loop moving.</h2>
    <p>
      A coordinator registers the family, Adwuma Pa detects who is due or silent,
      sends a secure check-in, processes the response, routes a relative when needed,
      and keeps the loop open until someone confirms the next action.
    </p>
    <div class="ap-flow">
      <div class="ap-flow-step"><strong>1. Watch</strong><span>Configurable silence thresholds per elder.</span></div>
      <div class="ap-flow-step"><strong>2. Ask</strong><span>Tokenized web or WhatsApp check-ins.</span></div>
      <div class="ap-flow-step"><strong>3. Understand</strong><span>ASR, translation, Qwen JSON analysis.</span></div>
      <div class="ap-flow-step"><strong>4. Close</strong><span>Nudge the right relative and resolve the loop.</span></div>
    </div>
  </div>
</section>
"""


def active_requests_html(limit=8):
    rows = db.rows(
        """
        SELECT r.token, r.request_type, r.reason_code, r.reason_detail, r.priority, r.status,
               r.created_at, m.name, m.location_city
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        WHERE r.status IN ('pending', 'sent', 'processing', 'needs_review')
        ORDER BY
          CASE r.priority WHEN 'red' THEN 0 WHEN 'amber' THEN 1 ELSE 2 END,
          r.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    if not rows:
        return '<div class="ap-empty">No active check-ins. Add family members, then run Autopilot or create a check-in.</div>'
    cards = []
    for row in rows:
        priority = row["priority"] or "routine"
        detail = row["reason_detail"] or friendly_reason(row["reason_code"])
        link = f"/checkin/{row['token']}"
        label = "Relative report" if row["request_type"] == "field_report" else "Elder check-in"
        cards.append(
            f"""
            <article class="ap-item ap-{priority}">
              <div>
                <div class="ap-item-title">{row['name']}</div>
                <div class="ap-item-meta">{label} · {friendly_reason(row['reason_code'])} · {row['status']}</div>
                <div class="ap-item-note">{detail}</div>
              </div>
              <code>{link}</code>
            </article>
            """
        )
    return '<section class="ap-list">' + "\n".join(cards) + "</section>"


def family_overview_html(limit=12):
    rows = dashboard_rows()[:limit]
    if not rows:
        return '<div class="ap-empty">No family members yet. Add the first elder or relative in Members.</div>'
    cards = []
    for row in rows:
        status = row["Status"].lower()
        cards.append(
            f"""
            <article class="ap-family-card ap-{status}">
              <div class="ap-family-top">
                <strong>{row['Name']}</strong>
                <span>{row['Status']}</span>
              </div>
              <div class="ap-item-meta">{row['City'] or 'Unknown city'} · {row.get('Role') or 'relative'} · {row['Language'] or 'language unset'}</div>
              <div class="ap-item-note">{row['Next action']}</div>
              <div class="ap-family-foot">Route: {row.get('Care route') or 'No care contact assigned'}</div>
              <div class="ap-family-foot">Last: {row['Last summary']}</div>
            </article>
            """
        )
    return '<section class="ap-family-grid">' + "\n".join(cards) + "</section>"


def care_routes_html(limit=10):
    rows = dashboard_rows()[:limit]
    if not rows:
        return '<div class="ap-empty">No care routes yet.</div>'
    items = []
    for row in rows:
        items.append(
            f"""
            <article class="ap-item">
              <div>
                <div class="ap-item-title">{row['Name']}</div>
                <div class="ap-item-meta">Next contact: {row.get('Care route') or 'No care contact assigned'}</div>
              </div>
              <span class="ap-state">{row['Status']}</span>
            </article>
            """
        )
    return '<section class="ap-list">' + "\n".join(items) + "</section>"


def member_registry_html():
    rows = db.rows(
        """
        SELECT name, phone, whatsapp, location_city, location_region, language,
               COALESCE(family_role, 'relative') AS family_role,
               COALESCE(is_coordinator, 0) AS is_coordinator,
               active
        FROM members
        ORDER BY is_coordinator DESC, name ASC
        """
    )
    if not rows:
        return '<div class="ap-empty">No family members registered yet.</div>'
    cards = []
    for row in rows:
        coordinator = " · coordinator" if row["is_coordinator"] else ""
        active = "Active" if row["active"] else "Inactive"
        cards.append(
            f"""
            <article class="ap-family-card">
              <div class="ap-family-top">
                <strong>{row['name']}</strong>
                <span>{active}</span>
              </div>
              <div class="ap-item-meta">{row['family_role']}{coordinator} · {row['location_city'] or 'city unset'}, {row['location_region'] or 'region unset'}</div>
              <div class="ap-item-note">{row['phone']} · {row['whatsapp'] or 'WhatsApp unset'} · {row['language']}</div>
            </article>
            """
        )
    return '<section class="ap-family-grid">' + "\n".join(cards) + "</section>"


def alert_overview_html(limit=8):
    rows = alert_rows()[:limit]
    if not rows:
        return '<div class="ap-empty">No open alerts or review items.</div>'
    items = []
    for row in rows:
        state_label = "Needs closure" if row["State"].lower() == "open" else row["State"]
        action = alert_next_action(row["Type"])
        items.append(
            f"""
            <article class="ap-item ap-alert">
              <div>
                <div class="ap-item-title">{row['Member']}</div>
                <div class="ap-item-meta">{row['Type']} · {state_label}</div>
                <div class="ap-item-note">{row['Notes'] or 'No notes yet.'}</div>
                <div class="ap-item-note"><strong>Next:</strong> {action}</div>
              </div>
              <span class="ap-state">{state_label.lower()}</span>
            </article>
            """
        )
    return '<section class="ap-list">' + "\n".join(items) + "</section>"


def alert_next_action(alert_type):
    alert_type = (alert_type or "").lower()
    if alert_type.startswith("red"):
        return "Call the elder or nudge the assigned relative, then resolve when someone confirms the next action."
    if alert_type.startswith("amber"):
        return "Ask the assigned relative to check in, then resolve once the family has a confirmed update."
    if alert_type == "needs_review":
        return "Review the saved response and model error, then resolve or create a follow-up request."
    if alert_type.startswith("reminder"):
        return "Send or resend a check-in link, then resolve after a response arrives."
    return "Use Resolve latest alert after the coordinator confirms the next family action."


def friendly_reason(reason):
    return {
        "coordinator_request": "Coordinator requested check-in",
        "routine_check": "Routine check-in",
        "reminder_silence": "Reminder after silence",
        "amber_silence": "Needs relative follow-up",
        "red_silence": "Urgent silence escalation",
        "first_party_amber_silence": "Relative asked to check in",
        "first_party_red_silence": "Urgent relative report",
    }.get(reason or "", (reason or "Check-in").replace("_", " ").title())


def status_cards_html():
    rows = dashboard_rows()
    counts = {status: 0 for status in ["Green", "Reminder", "Amber", "Red", "Support"]}
    for row in rows:
        counts[row["Status"]] = counts.get(row["Status"], 0) + 1
    return f"""
<div class="ap-status-grid">
  <div class="ap-status-card ap-green"><div class="ap-status-label">Green</div><div class="ap-status-value">{counts.get("Green", 0)}</div></div>
  <div class="ap-status-card ap-reminder"><div class="ap-status-label">Reminder</div><div class="ap-status-value">{counts.get("Reminder", 0)}</div></div>
  <div class="ap-status-card ap-amber"><div class="ap-status-label">Amber</div><div class="ap-status-value">{counts.get("Amber", 0)}</div></div>
  <div class="ap-status-card ap-red"><div class="ap-status-label">Red</div><div class="ap-status-value">{counts.get("Red", 0)}</div></div>
  <div class="ap-status-card"><div class="ap-status-label">Support</div><div class="ap-status-value">{counts.get("Support", 0)}</div></div>
</div>
"""


def alert_rows():
    return db.rows(
        """
        SELECT a.id AS Alert, m.name AS Member, a.alert_type AS Type, a.created_at AS Created,
               CASE WHEN a.resolved = 1 THEN 'Resolved' ELSE 'Open' END AS State,
               COALESCE(a.notes, '') AS Notes
        FROM alerts a
        JOIN members m ON m.id = a.member_id
        ORDER BY a.resolved ASC, a.created_at DESC
        LIMIT 30
        """
    )


def open_loop_rows():
    return db.rows(
        """
        SELECT m.name AS Member, a.alert_type AS Type, a.created_at AS Created, COALESCE(a.notes, '') AS Notes
        FROM alerts a
        JOIN members m ON m.id = a.member_id
        WHERE a.resolved = 0
        ORDER BY
          CASE
            WHEN a.alert_type LIKE 'red%' THEN 0
            WHEN a.alert_type LIKE 'amber%' THEN 1
            WHEN a.alert_type LIKE 'reminder%' THEN 2
            ELSE 3
          END,
          a.created_at DESC
        LIMIT 10
        """
    )


def esc(value):
    return html.escape("" if value is None else str(value))


def member_profile_html(member_id):
    if not member_id:
        return '<div class="ap-empty">Choose a family member.</div>'
    member = db.one("SELECT * FROM members WHERE id = ?", (member_id,))
    if not member:
        return '<div class="ap-empty">Member not found.</div>'
    contact_rows = db.rows(
        """
        SELECT c.name, c.whatsapp, c.location_city
        FROM first_party_contacts f
        JOIN members c ON c.id = f.contact_id
        WHERE f.elder_id = ?
        ORDER BY f.priority ASC
        """,
        (member_id,),
    )
    contacts = ", ".join(f"{row['name']} ({row['location_city']})" for row in contact_rows) or "None assigned"
    affiliations = db.affiliation_rows(member_id)
    affiliation_lines = []
    for row in affiliations[:8]:
        affiliation_lines.append(
            f"<li>{esc(row['Subject'])} -> {esc(row['Related'])}: {esc(row['Relationship'])} ({esc(row['Care role'])}, priority {esc(row['Priority'])})</li>"
        )
    affiliation_text = "\n".join(affiliation_lines) or "<li>None yet</li>"
    pending = db.rows(
        """
        SELECT token, reason_code, status
        FROM checkup_requests
        WHERE member_id = ? AND status IN ('pending', 'sent', 'needs_review', 'processing')
        ORDER BY created_at DESC
        LIMIT 3
        """,
        (member_id,),
    )
    pending_lines = "\n".join(
        f"<li><code>/checkin/{esc(row['token'])}</code> - {esc(row['reason_code'])} ({esc(row['status'])})</li>" for row in pending
    ) or "<li>None</li>"
    return f"""
<div class="ap-profile">
  <h3>{esc(member['name'])}</h3>
  <div class="ap-profile-grid">
    <div class="ap-profile-row"><strong>Location</strong><br>{esc(member.get('location_city') or 'Unknown')}, {esc(member.get('location_region') or '')}</div>
    <div class="ap-profile-row"><strong>Role</strong><br>{esc(member.get('family_role') or 'relative')}</div>
    <div class="ap-profile-row"><strong>Coordinator</strong><br>{'Yes' if member.get('is_coordinator') else 'No'}</div>
    <div class="ap-profile-row"><strong>Language</strong><br>{esc(member.get('language') or 'Unknown')}</div>
    <div class="ap-profile-row"><strong>Phone</strong><br>{esc(member.get('phone') or '')}</div>
    <div class="ap-profile-row"><strong>WhatsApp</strong><br>{esc(member.get('whatsapp') or member.get('phone') or '')}</div>
    <div class="ap-profile-row"><strong>First-party contacts</strong><br>{esc(contacts)}</div>
    <div class="ap-profile-row"><strong>Policy</strong><br>reminder {esc(member.get('reminder_minutes'))} min, amber {esc(member.get('escalation_minutes_amber'))} min, red {esc(member.get('escalation_minutes_red'))} min</div>
  </div>
  <div class="ap-profile-section"><strong>Affiliations</strong><ul>{affiliation_text}</ul></div>
  <div class="ap-profile-section"><strong>Open request links</strong><ul>{pending_lines}</ul></div>
</div>
"""


def member_checkin_rows(member_id):
    if not member_id:
        return []
    rows = db.rows(
        """
        SELECT submitted_at AS Submitted, source AS Source, input_type AS Input,
               analysis_status AS Status, COALESCE(concern_level, '') AS Concern,
               summary AS Summary, COALESCE(translation, '') AS Translation,
               transcript AS Transcript, COALESCE(processing_error, '') AS Error
        FROM checkins
        WHERE member_id = ?
        ORDER BY submitted_at DESC
        LIMIT 20
        """,
        (member_id,),
    )
    return table_value(rows, CHECKIN_HEADERS)


def member_alert_rows(member_id):
    if not member_id:
        return []
    rows = db.rows(
        """
        SELECT a.id AS Alert, m.name AS Member, a.alert_type AS Type, a.created_at AS Created,
               CASE WHEN a.resolved = 1 THEN 'Resolved' ELSE 'Open' END AS State,
               COALESCE(a.notes, '') AS Notes
        FROM alerts a
        JOIN members m ON m.id = a.member_id
        WHERE a.member_id = ?
        ORDER BY a.resolved ASC, a.created_at DESC
        LIMIT 20
        """,
        (member_id,),
    )
    return table_value(rows, ALERT_HEADERS)


def member_nudge_rows(member_id):
    if not member_id:
        return []
    rows = db.rows(
        """
        SELECT n.sent_at AS Sent, COALESCE(c.name, 'Unassigned') AS Contact,
               COALESCE(r.token, '') AS Request,
               COALESCE(n.responded_at, '') AS Responded, COALESCE(n.checkin_id, '') AS "Check-in"
        FROM nudges n
        LEFT JOIN members c ON c.id = n.contact_id
        LEFT JOIN checkup_requests r ON r.related_nudge_id = n.id
        WHERE n.elder_id = ?
        ORDER BY n.sent_at DESC
        LIMIT 20
        """,
        (member_id,),
    )
    return table_value(rows, NUDGE_HEADERS)


def member_affiliation_rows(member_id):
    if not member_id:
        return []
    rows = db.affiliation_rows(member_id)
    return table_value(rows, AFFILIATION_HEADERS)


def load_member_detail(member_id):
    return (
        member_profile_html(member_id),
        member_checkin_rows(member_id),
        member_alert_rows(member_id),
        member_nudge_rows(member_id),
        member_affiliation_rows(member_id),
    )


def member_choices():
    return [(f"{row['name']} - {row['location_city']}", row["id"]) for row in db.rows("SELECT * FROM members ORDER BY name")]


def member_dropdown():
    return gr.Dropdown(choices=member_choices())


def add_member(name, phone, whatsapp, city, region, language, family_role, is_coordinator, call_enabled):
    if not name or not phone:
        raise gr.Error("Name and phone are required.")
    member_id = db.add_member(name, phone, whatsapp or phone, city, region, language, call_enabled, family_role, is_coordinator)
    choices = member_dropdown()
    member = db.one("SELECT phone, whatsapp FROM members WHERE id = ?", (member_id,))
    return (
        f"Saved {name}. Phone: {member['phone']}. WhatsApp: {member['whatsapp']}.",
        member_registry_html(),
        storage_status_html(),
        status_cards_html(),
        family_overview_html(),
        care_routes_html(),
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
    )


def load_member_for_edit(member_id):
    if not member_id:
        raise gr.Error("Choose a member to edit.")
    member = db.one("SELECT * FROM members WHERE id = ?", (member_id,))
    if not member:
        raise gr.Error("Member not found.")
    return (
        member["name"],
        member["phone"],
        member["whatsapp"],
        member.get("location_city") or "",
        member.get("location_region") or "Greater Accra",
        member.get("language") or "twi",
        member.get("family_role") or "relative",
        bool(member.get("is_coordinator")),
        bool(member.get("call_enabled")),
    )


def save_member_edits(member_id, name, phone, whatsapp, city, region, language, family_role, is_coordinator, call_enabled):
    if not member_id:
        raise gr.Error("Choose a member to edit.")
    if not name or not phone:
        raise gr.Error("Name and phone are required.")
    db.update_member(member_id, name, phone, whatsapp or phone, city, region, language, call_enabled, family_role, is_coordinator)
    choices = member_dropdown()
    member = db.one("SELECT phone, whatsapp FROM members WHERE id = ?", (member_id,))
    return (
        f"Updated {name}. Phone: {member['phone']}. WhatsApp: {member['whatsapp']}.",
        member_registry_html(),
        storage_status_html(),
        member_profile_html(member_id),
        status_cards_html(),
        family_overview_html(),
        care_routes_html(),
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
    )


def add_affiliation(subject_member_id, related_member_id, relationship, care_role, priority, can_coordinate, notes):
    if not subject_member_id or not related_member_id:
        raise gr.Error("Choose both family members.")
    if not relationship:
        raise gr.Error("Relationship is required.")
    try:
        affiliation_id = db.add_affiliation(
            subject_member_id,
            related_member_id,
            relationship,
            care_role,
            priority,
            can_coordinate,
            notes or "",
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return (
        f"Saved affiliation {affiliation_id}.",
        member_affiliation_rows(subject_member_id),
        member_profile_html(subject_member_id),
        family_overview_html(),
        care_routes_html(),
    )


def load_sample_data():
    db.seed_demo_data()
    choices = member_dropdown()
    return (
        "Sample data loaded.",
        status_cards_html(),
        active_requests_html(),
        family_overview_html(),
        care_routes_html(),
        alert_overview_html(),
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
    )


def clear_data():
    db.clear_all_data()
    choices = member_dropdown()
    return (
        "All members, check-ins, alerts, nudges, and calls cleared.",
        status_cards_html(),
        active_requests_html(),
        family_overview_html(),
        care_routes_html(),
        alert_overview_html(),
        member_registry_html(),
        storage_status_html(),
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
        choices,
    )


def transcribe_voice(audio, language, model_key):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    result = modal_client.transcribe_audio(audio, language, model_key)
    if not result.ok:
        return "", f"Needs review: {result.error}", "", 0.0
    status = (
        f"Transcript from {result.data.get('model_used', 'Modal ASR')}. "
        f"Confidence: {float(result.data.get('confidence') or 0):.2f}."
    )
    return result.data.get("text", ""), status, result.data.get("model_used", ""), float(result.data.get("confidence") or 0)


def load_request_context(token):
    request = db.get_request_by_token(normalize_token(token))
    if not request:
        return "No check-in found for that link.", "twi", "", "", "self"
    source = "field_report" if request["request_type"] == "field_report" else "self"
    return (
        request_context_markdown(request),
        request["language"],
        request["member_name"],
        request["reason_code"],
        source,
    )


def request_context_markdown(request):
    reporter = "nearby relative" if request["request_type"] == "field_report" else "elder"
    return f"""
### Response for {request['member_name']}

Requested by: **{request.get('requester') or 'Adwuma Pa'}**  
Expected responder: **{reporter}**  
Reason: **{friendly_reason(request['reason_code'])}**  
Details: {request.get('reason_detail') or 'No extra details.'}  
Status: **{request['status']}**

If AI processing is unavailable, the update will be saved for family review instead of being scored.
"""


def submit_checkin_by_token(token, language, text, audio, input_mode, source):
    token = normalize_token(token)
    if not token:
        raise gr.Error("Paste the secure check-in link first.")
    if input_mode == "text" and not (text or "").strip():
        raise gr.Error("Enter a response before submitting.")
    if input_mode == "voice" and audio is None:
        raise gr.Error("Record or upload audio before submitting.")
    result = pipeline.submit_request_response(
        token=token,
        text=text or "",
        language=language,
        input_type=input_mode,
        audio=audio,
        source=source,
    )
    return (
        checkin_receipt(result),
        json.dumps(result, indent=2),
        status_cards_html(),
        active_requests_html(),
        family_overview_html(),
        care_routes_html(),
        alert_overview_html(),
    )


def normalize_token(value):
    clean = (value or "").strip()
    if "/checkin/" in clean:
        clean = clean.rsplit("/checkin/", 1)[-1]
    return clean.strip().strip("/")


def checkin_receipt(result):
    if result.get("status") == "complete":
        return (
            f"Saved check-in `{result['checkin_id']}` with concern level "
            f"**{result['concern_level']}**.\n\nTranslation: {result.get('translation') or 'English input'}"
        )
    return (
        f"Saved for human review as `{result.get('checkin_id', 'unknown')}`.\n\n"
        f"Reason: {result.get('error') or result.get('message') or 'No automated analysis was produced.'}"
    )


def resolve_first_open_alert(resolved_by, notes):
    alert = db.one("SELECT id FROM alerts WHERE resolved = 0 ORDER BY created_at DESC LIMIT 1")
    if not alert:
        return "No open alerts.", alert_overview_html(), family_overview_html(), care_routes_html(), status_cards_html()
    db.resolve_alert(alert["id"], resolved_by or "Coordinator", notes or "Loop closed.")
    return f"Resolved {alert['id']}.", alert_overview_html(), family_overview_html(), care_routes_html(), status_cards_html()


def nudge(member_id):
    if not member_id:
        raise gr.Error("Choose an elder.")
    return simulate_nudge(member_id)


def create_manual_request(member_id, reason_code, reason_detail, request_type, priority):
    if not member_id:
        raise gr.Error("Choose a family member.")
    request_id = db.create_checkup_request(
        member_id,
        reason_code or "coordinator_request",
        reason_detail or "Coordinator requested a check-in.",
        request_type=request_type,
        channel="web",
        requester="Coordinator",
        priority=priority,
    )
    request = db.one("SELECT token FROM checkup_requests WHERE id = ?", (request_id,))
    return (
        f"Created secure check-in link:\n\n`/checkin/{request['token']}`",
        status_cards_html(),
        active_requests_html(),
        family_overview_html(),
        care_routes_html(),
        alert_overview_html(),
        gr.Dropdown(choices=pending_request_choices()),
    )


def pending_request_choices():
    rows = db.rows(
        """
        SELECT r.id, m.name, r.reason_code, r.priority, r.status
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        WHERE r.status IN ('pending', 'needs_review')
        ORDER BY
          CASE r.priority WHEN 'red' THEN 0 WHEN 'amber' THEN 1 ELSE 2 END,
          r.created_at DESC
        LIMIT 30
        """
    )
    return [(f"{row['name']} - {friendly_reason(row['reason_code'])} ({row['priority']}, {row['status']})", row["id"]) for row in rows]


def send_checkin_whatsapp(request_id):
    if not request_id:
        raise gr.Error("Choose a pending check-in request.")
    result = twilio_client.send_request_link(request_id)
    choices = gr.Dropdown(choices=pending_request_choices())
    message = result.message
    if result.sid:
        message = f"{message} SID: {result.sid}"
    return (
        message,
        status_cards_html(),
        active_requests_html(),
        family_overview_html(),
        care_routes_html(),
        alert_overview_html(),
        outbound_table_value(),
        choices,
    )


def twilio_status_markdown():
    if twilio_client.configured():
        sender = twilio_client.configured_from()
        return f"WhatsApp delivery: **ready** from `{sender}`."
    return "WhatsApp delivery: **off**. Create secure links now; enable Twilio when the family is ready for live messages."


def public_checkin_page(token: str, request: dict, message: str = "") -> str:
    checked = {
        "twi": "selected" if request.get("language") == "twi" else "",
        "fat": "selected" if request.get("language") == "fat" else "",
        "eng": "selected" if request.get("language") == "eng" else "",
    }
    expected = "relative or caregiver" if request["request_type"] == "field_report" else "elder"
    submit_disabled = "disabled" if request["status"] == "complete" else ""
    status_note = (
        "This check-in has already been completed."
        if request["status"] == "complete"
        else "Write a short update. If AI is unavailable, your response is still saved for family review."
    )
    escaped_message = f'<div class="notice">{html.escape(message)}</div>' if message else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adwuma Pa Check-in</title>
  <style>
    body {{
      background: #e2e8f0;
      color: #0f172a;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 22px;
    }}
    main {{
      background: #ffffff;
      border: 1px solid #94a3b8;
      border-radius: 10px;
      margin: 0 auto;
      max-width: 680px;
      overflow: hidden;
    }}
    header {{
      background: #0f172a;
      color: #f8fafc;
      padding: 22px;
    }}
    h1 {{
      font-size: 28px;
      line-height: 1.1;
      margin: 0 0 8px;
    }}
    header p {{
      color: #cbd5e1;
      line-height: 1.45;
      margin: 0;
    }}
    section {{
      padding: 20px 22px 22px;
    }}
    .meta {{
      border-bottom: 1px solid #cbd5e1;
      margin-bottom: 16px;
      padding-bottom: 14px;
    }}
    .meta strong, label {{
      color: #0f172a;
      display: block;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .meta div {{
      color: #334155;
      margin: 0 0 10px;
    }}
    select, textarea {{
      background: #ffffff;
      border: 1px solid #64748b;
      border-radius: 8px;
      box-sizing: border-box;
      color: #0f172a;
      font: inherit;
      margin-bottom: 14px;
      padding: 12px;
      width: 100%;
    }}
    textarea {{
      min-height: 150px;
      resize: vertical;
    }}
    button {{
      background: #064e3b;
      border: 1px solid #064e3b;
      border-radius: 8px;
      color: #ffffff;
      cursor: pointer;
      font: inherit;
      font-weight: 800;
      padding: 13px 16px;
      width: 100%;
    }}
    button:disabled {{
      background: #94a3b8;
      border-color: #94a3b8;
      cursor: not-allowed;
    }}
    .notice {{
      background: #ecfdf5;
      border: 1px solid #047857;
      border-radius: 8px;
      color: #064e3b;
      font-weight: 700;
      line-height: 1.4;
      margin-bottom: 14px;
      padding: 12px;
    }}
    .hint {{
      color: #475569;
      font-size: 14px;
      line-height: 1.45;
      margin: 12px 0 0;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Adwuma Pa</h1>
      <p>Secure family check-in for {html.escape(request["member_name"])}.</p>
    </header>
    <section>
      {escaped_message}
      <div class="meta">
        <strong>Person being checked on</strong>
        <div>{html.escape(request["member_name"])} · {html.escape(request.get("location_city") or "Location not set")}</div>
        <strong>Expected responder</strong>
        <div>{html.escape(expected)}</div>
        <strong>Reason</strong>
        <div>{html.escape(friendly_reason(request["reason_code"]))}</div>
        <strong>Details</strong>
        <div>{html.escape(request.get("reason_detail") or "No extra details.")}</div>
      </div>
      <form method="post" action="/checkin/{html.escape(token)}">
        <label for="language">Response language</label>
        <select id="language" name="language">
          <option value="twi" {checked["twi"]}>Twi</option>
          <option value="fat" {checked["fat"]}>Fante</option>
          <option value="eng" {checked["eng"]}>English</option>
        </select>
        <label for="text">Response</label>
        <textarea id="text" name="text" required placeholder="Type the update here."></textarea>
        <button type="submit" {submit_disabled}>Send update</button>
      </form>
      <p class="hint">{html.escape(status_note)}</p>
    </section>
  </main>
</body>
</html>"""


def public_checkin_result_page(request: dict, result: dict) -> str:
    if result.get("status") == "complete":
        message = "Thank you. Your update was received and shared with the family coordinator."
    elif result.get("status") == "needs_review":
        message = "Thank you. Your update was received and saved for family review."
    else:
        message = result.get("message") or "Your response could not be saved. Please contact the family coordinator."
    return public_checkin_page(request["token"], {**request, "status": "complete"}, message)


def process_public_checkin(token: str, text: str, language: str, source: str) -> None:
    pipeline.submit_request_response(
        token=token,
        text=text,
        language=language,
        input_type="text",
        source=source,
    )


def install_webhook_routes(server):
    @server.get("/checkin/{token}")
    async def public_checkin(token: str):
        db.init_db()
        normalized = normalize_token(token)
        request = db.get_request_by_token(normalized)
        if not request:
            return Response(
                content="<h1>Check-in not found</h1><p>This link is invalid or no longer exists.</p>",
                status_code=404,
                media_type="text/html",
            )
        return Response(content=public_checkin_page(normalized, request), media_type="text/html")

    @server.post("/checkin/{token}")
    async def submit_public_checkin(token: str, request: Request, background_tasks: BackgroundTasks):
        db.init_db()
        normalized = normalize_token(token)
        checkup = db.get_request_by_token(normalized)
        if not checkup:
            return Response(
                content="<h1>Check-in not found</h1><p>This link is invalid or no longer exists.</p>",
                status_code=404,
                media_type="text/html",
            )
        raw_body = (await request.body()).decode("utf-8")
        payload = {key: values[0] if values else "" for key, values in parse_qs(raw_body).items()}
        text = (payload.get("text") or "").strip()
        language = payload.get("language") or checkup.get("language") or "twi"
        if not text:
            return Response(
                content=public_checkin_page(normalized, checkup, "Please type a response before sending."),
                status_code=400,
                media_type="text/html",
            )
        background_tasks.add_task(
            process_public_checkin,
            normalized,
            text,
            language,
            "field_report" if checkup["request_type"] == "field_report" else "self",
        )
        return Response(
            content=public_checkin_page(
                normalized,
                checkup,
                "Thank you. Your update was received and is being processed for the family coordinator.",
            ),
            media_type="text/html",
        )

    @server.get("/twilio/health")
    async def twilio_health():
        return {"ok": True, "service": "adwuma-pa-twilio"}

    @server.get("/debug/storage")
    async def debug_storage():
        return db.storage_status()

    @server.post("/twilio/whatsapp")
    async def twilio_whatsapp(request: Request, background_tasks: BackgroundTasks):
        raw_body = (await request.body()).decode("utf-8")
        payload = {key: values[0] if values else "" for key, values in parse_qs(raw_body).items()}
        sender = payload.get("From", "")
        message_body = payload.get("Body", "")
        if sender and message_body:
            background_tasks.add_task(twilio_client.receive_whatsapp_reply, sender, message_body)
        xml = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
        return Response(content=xml, media_type="application/xml")

    @server.post("/twilio/status")
    async def twilio_status(request: Request):
        raw_body = (await request.body()).decode("utf-8")
        payload = {key: values[0] if values else "" for key, values in parse_qs(raw_body).items()}
        twilio_client.record_status_callback(
            payload.get("MessageSid", ""),
            payload.get("MessageStatus", ""),
            payload.get("ErrorCode", ""),
            payload.get("ErrorMessage", ""),
        )
        return {"ok": True}

    return server


def run_silence_scan():
    actions = scan_silence()
    return "\n".join(actions), status_cards_html(), active_requests_html(), family_overview_html(), care_routes_html(), alert_overview_html()


def update_escalation_settings(member_id, reminder_minutes, amber_minutes, red_minutes):
    if not member_id:
        raise gr.Error("Choose a family member.")
    db.update_escalation(member_id, reminder_minutes, amber_minutes, red_minutes)
    member = db.one("SELECT name, reminder_minutes, escalation_minutes_amber, escalation_minutes_red FROM members WHERE id = ?", (member_id,))
    return (
        f"Updated {member['name']}: reminder {member['reminder_minutes']} min, "
        f"amber {member['escalation_minutes_amber']} min, red {member['escalation_minutes_red']} min.",
        status_cards_html(),
        family_overview_html(),
    )


def model_budget_markdown():
    return f"""
<span class="ap-pill">ASR: {ASR_CONFIG["primary"]["label"]} ({ASR_CONFIG["primary"]["parameters_b"]}B)</span>
<span class="ap-pill">Translation: {TRANSLATION_CONFIG["label"]} ({TRANSLATION_CONFIG["parameters_b"]}B)</span>
<span class="ap-pill">LLM: {LLM_CONFIG["label"]} ({LLM_CONFIG["parameters_b"]}B)</span>
<span class="ap-pill">TTS: {TTS_CONFIG["label"]} ({TTS_CONFIG["parameters_b"]}B)</span>
<span class="ap-pill">Total: {total_parameter_budget_b():.1f}B / 32B</span>
"""


def modal_health_markdown():
    result = modal_client.modal_health()
    if result.ok:
        return f"Modal backend: **online** ({result.latency_ms} ms)\n\n```json\n{json.dumps(result.data, indent=2)}\n```"
    return f"Modal backend: **not configured/off**\n\n{result.error}"


def build_tts_prompt(member_id, prompt_type, language):
    member = db.one("SELECT * FROM members WHERE id = ?", (member_id,)) if member_id else None
    name = member["name"] if member else "Opanyin"
    if language == "eng":
        templates = {
            "reminder": f"Hello {name}. This is Adwuma Pa checking in. Please send a short update so your family knows how you are doing.",
            "call_greeting": f"Hello {name}. This is Adwuma Pa calling for your family. How are you feeling today?",
            "call_close": "Thank you. Your family will receive this update. We will follow up if anyone needs to check on you.",
        }
    else:
        templates = {
            "reminder": f"{name}, Adwuma Pa re bisa wo ho asɛm. Yɛsrɛ wo, kyerɛw anaa ka sɛnea wo ho te.",
            "call_greeting": f"{name}, Adwuma Pa na ɛrefrɛ wo ama abusua no. Ɛnnɛ wo ho te sɛn?",
            "call_close": "Meda wo ase. Yɛde wo nkra bɛkɔ ama abusua no, na sɛ ɛhia a obi bɛba abɛhwɛ wo.",
        }
    return templates.get(prompt_type, templates["reminder"])


def synthesize_tts_prompt(text, language):
    if not text or not text.strip():
        raise gr.Error("Generate or type TTS text first.")
    result = modal_client.synthesize_speech(text, language)
    if not result.ok:
        raise gr.Error(f"TTS needs review: {result.error}")
    audio = result.data.get("audio")
    status = f"Generated with {result.data.get('model_used', 'Modal TTS')}."
    return audio, status


def build_app():
    db.init_db()
    with gr.Blocks(title="Adwuma Pa - Family Care Network") as demo:
        gr.HTML(
            """
            <div class="ap-header">
              <div class="ap-title">Adwuma Pa</div>
              <div class="ap-subtitle">
                A small AI care network for Ghanaian families: multilingual check-ins, concern scoring,
                silence detection, nearest-relative nudges, and loop closure for elders who may not ask for help.
              </div>
            </div>
            """
        )

        budget = gr.HTML(model_budget_markdown())

        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.HTML(demo_story_html())
                status_cards = gr.HTML(status_cards_html())
                with gr.Row():
                    refresh = gr.Button("Refresh", variant="primary")
                    scan_btn = gr.Button("Run silence scan now")
                gr.HTML('<div class="ap-section-title">Active check-ins</div>')
                requests = gr.HTML(active_requests_html())
                gr.HTML('<div class="ap-section-title">Family overview</div>')
                family_table = gr.HTML(family_overview_html())
                gr.HTML('<div class="ap-section-title">Alerts and reviews</div>')
                alerts = gr.HTML(alert_overview_html())
                gr.HTML('<div class="ap-section-title">Close confirmed follow-up</div>')
                with gr.Row():
                    resolved_by = gr.Textbox(label="Confirmed by", value="Coordinator")
                    resolution_notes = gr.Textbox(label="What happened", value="Relative checked in and confirmed next action.")
                    resolve_btn = gr.Button("Close most urgent alert")
                resolve_output = gr.Textbox(label="Closure result", interactive=False)
                scan_output = gr.Textbox(label="Autopilot result", lines=5, interactive=False)
                care_routes = gr.HTML(care_routes_html(), visible=False)

            with gr.Tab("Family Setup"):
                member_storage = gr.HTML(storage_status_html())
                with gr.Accordion("Add family member", open=True):
                    with gr.Row():
                        new_name = gr.Textbox(label="Name")
                        new_phone = gr.Textbox(label="Phone")
                        new_whatsapp = gr.Textbox(label="WhatsApp")
                    with gr.Row():
                        new_city = gr.Textbox(label="City")
                        new_region = gr.Dropdown(choices=GHANA_REGIONS, value="Greater Accra", label="Region")
                        new_language = gr.Dropdown(
                            choices=[("Twi", "twi"), ("Fante", "fat"), ("English", "eng")],
                            value="twi",
                            label="Preferred language",
                        )
                    with gr.Row():
                        new_role = gr.Dropdown(choices=ROLE_CHOICES, value="relative", label="Family role")
                        new_is_coordinator = gr.Checkbox(label="Can coordinate care", value=False)
                        new_call = gr.Checkbox(label="Voice call enabled", value=True)
                    add_btn = gr.Button("Add member", variant="primary")
                    add_output = gr.Textbox(label="Result", interactive=False)

                with gr.Accordion("Add affiliation", open=True):
                    with gr.Row():
                        affiliation_subject = gr.Dropdown(choices=member_choices(), label="Person being cared for")
                        affiliation_related = gr.Dropdown(choices=member_choices(), label="Related family member")
                    with gr.Row():
                        affiliation_relationship = gr.Dropdown(choices=RELATIONSHIP_CHOICES, value="family_coordinator", label="Relationship")
                        affiliation_care_role = gr.Dropdown(choices=CARE_ROLE_CHOICES, value="first_party_contact", label="Care role")
                    with gr.Row():
                        affiliation_priority = gr.Number(label="Priority", value=5, precision=0)
                        affiliation_can_coordinate = gr.Checkbox(label="Can coordinate this person's care", value=False)
                    affiliation_notes = gr.Textbox(label="Notes", lines=2)
                    affiliation_btn = gr.Button("Save affiliation", variant="primary")
                    affiliation_output = gr.Textbox(label="Affiliation result", interactive=False)
                    affiliation_table = gr.Dataframe(headers=AFFILIATION_HEADERS, value=[], label="Saved affiliations", interactive=False, wrap=True)

                gr.HTML('<div class="ap-section-title">Registered family</div>')
                member_registry = gr.HTML(member_registry_html())

                with gr.Accordion("Edit and member history", open=False):
                    edit_member = gr.Dropdown(choices=member_choices(), label="Member to edit")
                    load_edit_member = gr.Button("Load member")
                    with gr.Row():
                        edit_name = gr.Textbox(label="Name")
                        edit_phone = gr.Textbox(label="Phone")
                        edit_whatsapp = gr.Textbox(label="WhatsApp")
                    with gr.Row():
                        edit_city = gr.Textbox(label="City")
                        edit_region = gr.Dropdown(choices=GHANA_REGIONS, value="Greater Accra", label="Region")
                        edit_language = gr.Dropdown(
                            choices=[("Twi", "twi"), ("Fante", "fat"), ("English", "eng")],
                            value="twi",
                            label="Preferred language",
                        )
                    with gr.Row():
                        edit_role = gr.Dropdown(choices=ROLE_CHOICES, value="relative", label="Family role")
                        edit_is_coordinator = gr.Checkbox(label="Can coordinate care", value=False)
                        edit_call = gr.Checkbox(label="Voice call enabled", value=True)
                    save_edit_member = gr.Button("Save member changes", variant="primary")
                    edit_output = gr.Textbox(label="Edit result", interactive=False)
                    gr.HTML('<div class="ap-section-title">Member detail</div>')
                    detail_member = gr.Dropdown(choices=member_choices(), label="Family member")
                    load_detail = gr.Button("Load member detail", variant="primary")
                    member_profile = gr.HTML(member_profile_html(None))
                    with gr.Accordion("Raw history tables", open=False):
                        member_affiliations = gr.Dataframe(headers=AFFILIATION_HEADERS, value=[], label="Affiliations", interactive=False, wrap=True)
                        member_checkins = gr.Dataframe(headers=CHECKIN_HEADERS, value=[], label="Check-in history", interactive=False, wrap=True)
                        member_alerts = gr.Dataframe(headers=ALERT_HEADERS, value=[], label="Member alerts", interactive=False, wrap=True)
                        member_nudges = gr.Dataframe(headers=NUDGE_HEADERS, value=[], label="Nudge history", interactive=False, wrap=True)

            with gr.Tab("Care Loop"):
                source_state = gr.State("self")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="ap-section-title">Create and send</div>')
                        request_member_picker = gr.Dropdown(choices=member_choices(), label="Person to check on")
                        with gr.Row():
                            manual_type = gr.Dropdown(
                                choices=[("Elder check-in", "elder_checkin"), ("Relative field report", "field_report")],
                                value="elder_checkin",
                                label="Request type",
                            )
                            manual_priority = gr.Dropdown(
                                choices=[("Routine", "routine"), ("Amber", "amber"), ("Red", "red")],
                                value="routine",
                                label="Priority",
                            )
                        manual_reason = gr.Dropdown(
                            choices=[
                                ("Coordinator request", "coordinator_request"),
                                ("Routine check", "routine_check"),
                                ("Reminder after silence", "reminder_silence"),
                                ("Needs relative follow-up", "amber_silence"),
                                ("Urgent silence escalation", "red_silence"),
                            ],
                            value="coordinator_request",
                            label="Why this exists",
                        )
                        manual_detail = gr.Textbox(label="Message", lines=2, value="Coordinator requested a check-in.")
                        create_request_btn = gr.Button("Create secure check-in link", variant="primary")
                        create_request_output = gr.Markdown()
                        twilio_status = gr.Markdown(twilio_status_markdown())
                        send_request_picker = gr.Dropdown(choices=pending_request_choices(), label="Pending request")
                        send_whatsapp_btn = gr.Button("Send selected link by WhatsApp", variant="primary")
                        send_whatsapp_output = gr.Textbox(label="Send result", interactive=False)
                    with gr.Column(scale=1):
                        gr.HTML('<div class="ap-section-title">Record response</div>')
                        with gr.Row():
                            request_token = gr.Textbox(label="Check-in link", placeholder="/checkin/...")
                            load_request = gr.Button("Find", variant="primary")
                        request_context = gr.Markdown("Find the request before saving a response.")
                        with gr.Row():
                            request_member = gr.Textbox(label="Person checked on", interactive=False)
                            request_reason = gr.Textbox(label="Reason", interactive=False)
                        with gr.Row():
                            language = gr.Dropdown(
                                choices=[("Twi", "twi"), ("Fante", "fat"), ("English", "eng")],
                                value="twi",
                                label="Language",
                            )
                            input_mode = gr.Radio([("Text", "text"), ("Voice", "voice")], value="text", label="Format")
                        text = gr.Textbox(
                            label="Received response",
                            lines=5,
                            placeholder="Paste the elder or relative response exactly as received.",
                        )
                        voice_audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Voice audio, only when format is Voice",
                        )
                        submit = gr.Button("Save received response", variant="primary")
                        receipt = gr.Textbox(label="Result", interactive=False)
                        ai_json = gr.Code(label="Care processing result", language="json", visible=False)

                with gr.Accordion("Loop tools and settings", open=False):
                    with gr.Row():
                        relay_member = gr.Dropdown(choices=member_choices(), label="Elder needing follow-up")
                        nudge_btn = gr.Button("Draft relative nudge", variant="primary")
                    nudge_output = gr.Textbox(label="WhatsApp nudge draft", lines=4, interactive=False)
                    gr.HTML('<div class="ap-section-title">Escalation timing</div>')
                    policy_member = gr.Dropdown(choices=member_choices(), label="Family member")
                    with gr.Row():
                        reminder_minutes = gr.Number(label="Reminder after minutes", value=10080, precision=0)
                        amber_minutes = gr.Number(label="Amber after minutes", value=14400, precision=0)
                        red_minutes = gr.Number(label="Red after minutes", value=20160, precision=0)
                    policy_btn = gr.Button("Save escalation policy", variant="primary")
                    policy_output = gr.Textbox(label="Policy update", interactive=False)
                    gr.HTML('<div class="ap-section-title">TTS prompt check</div>')
                    with gr.Row():
                        tts_member = gr.Dropdown(choices=member_choices(), label="Family member")
                        tts_language = gr.Dropdown(
                            choices=[("Twi/Akan", "twi"), ("Fante/Akan", "fat"), ("English", "eng")],
                            value="twi",
                            label="TTS language",
                        )
                        tts_prompt_type = gr.Dropdown(choices=TTS_PROMPT_TYPES, value="reminder", label="Prompt type")
                    tts_text = gr.Textbox(label="Prompt text", lines=3)
                    with gr.Row():
                        generate_tts_prompt = gr.Button("Generate prompt text")
                        synthesize_tts = gr.Button("Synthesize prompt", variant="primary")
                    tts_audio = gr.Audio(label="Generated prompt audio", type="numpy")
                    tts_status = gr.Textbox(label="TTS status", interactive=False)
                    with gr.Accordion("Delivery log and data controls", open=False):
                        outbound_messages = gr.Dataframe(headers=OUTBOUND_HEADERS, value=outbound_table_value(), label="Recent WhatsApp attempts", interactive=False, wrap=True)
                        gr.Markdown("Production data starts empty. This only clears records; it never loads dummy data.")
                        clear_data_btn = gr.Button("Clear all data", variant="stop")
                        admin_output = gr.Textbox(label="Admin action", interactive=False)

            with gr.Tab("Build Notes"):
                gr.Markdown(
                    """
### Submission positioning

This is a Backyard AI project: it solves one real family coordination problem instead of a generic SaaS problem.

The AI is load-bearing in four places: speech-to-text for Twi/Fante, Twi/Fante-to-English translation,
Qwen structured concern analysis, and routing the next human action. If Modal is unavailable, the app stores
the response as needs_review instead of producing a fake score.

### OpenAI track case

The project is Codex-built, includes an agent trace/report path, and demonstrates a practical agentic workflow:
monitor, interpret, choose the nearest responsible person, escalate, and close the loop.
                    """
                )
                gr.Markdown(
                    """
### Built with OpenAI Codex

Codex converted the product spec into two working Hugging Face Spaces: the ASR evaluation app and this family care network.

### Implemented by Codex in this repo

- ASR eval app with MMS, Adwuma Pa fine-tune, and GiftMark model comparison.
- Community ASR voting for Twi/Fante/Akan samples.
- Main Gradio care dashboard with SQLite persistence.
- Tokenized checkup requests, alerts, first-party nudge drafts, and loop resolution.
- Configurable reminder, amber, and red silence escalation intervals.
- Modal-safe client boundary for ASR, translation, Qwen analysis, and TTS.

### Current execution plan

Next: start Modal only for targeted endpoint validation, then stop it when live testing is complete.
                    """
                )
                modal_status = gr.Markdown(modal_health_markdown())

        refresh.click(refresh_dashboard, outputs=[status_cards, requests, family_table, care_routes, alerts, modal_status, budget])
        scan_btn.click(run_silence_scan, outputs=[scan_output, status_cards, requests, family_table, care_routes, alerts])
        load_request.click(
            load_request_context,
            inputs=[request_token],
            outputs=[request_context, language, request_member, request_reason, source_state],
        )
        submit.click(
            submit_checkin_by_token,
            inputs=[request_token, language, text, voice_audio, input_mode, source_state],
            outputs=[receipt, ai_json, status_cards, requests, family_table, care_routes, alerts],
        )
        resolve_btn.click(resolve_first_open_alert, inputs=[resolved_by, resolution_notes], outputs=[resolve_output, alerts, family_table, care_routes, status_cards])
        create_request_btn.click(
            create_manual_request,
            inputs=[request_member_picker, manual_reason, manual_detail, manual_type, manual_priority],
            outputs=[create_request_output, status_cards, requests, family_table, care_routes, alerts, send_request_picker],
        )
        send_whatsapp_btn.click(
            send_checkin_whatsapp,
            inputs=[send_request_picker],
            outputs=[send_whatsapp_output, status_cards, requests, family_table, care_routes, alerts, outbound_messages, send_request_picker],
        )
        nudge_btn.click(nudge, inputs=[relay_member], outputs=[nudge_output])
        generate_tts_prompt.click(build_tts_prompt, inputs=[tts_member, tts_prompt_type, tts_language], outputs=[tts_text])
        synthesize_tts.click(synthesize_tts_prompt, inputs=[tts_text, tts_language], outputs=[tts_audio, tts_status])
        load_detail.click(load_member_detail, inputs=[detail_member], outputs=[member_profile, member_checkins, member_alerts, member_nudges, member_affiliations])
        load_edit_member.click(
            load_member_for_edit,
            inputs=[edit_member],
            outputs=[edit_name, edit_phone, edit_whatsapp, edit_city, edit_region, edit_language, edit_role, edit_is_coordinator, edit_call],
        )
        save_edit_member.click(
            save_member_edits,
            inputs=[
                edit_member,
                edit_name,
                edit_phone,
                edit_whatsapp,
                edit_city,
                edit_region,
                edit_language,
                edit_role,
                edit_is_coordinator,
                edit_call,
            ],
            outputs=[
                edit_output,
                member_registry,
                member_storage,
                member_profile,
                status_cards,
                family_table,
                care_routes,
                request_member_picker,
                relay_member,
                tts_member,
                detail_member,
                policy_member,
                affiliation_subject,
                affiliation_related,
                edit_member,
            ],
        )
        affiliation_btn.click(
            add_affiliation,
            inputs=[
                affiliation_subject,
                affiliation_related,
                affiliation_relationship,
                affiliation_care_role,
                affiliation_priority,
                affiliation_can_coordinate,
                affiliation_notes,
            ],
            outputs=[affiliation_output, affiliation_table, member_profile, family_table, care_routes],
        )
        policy_btn.click(update_escalation_settings, inputs=[policy_member, reminder_minutes, amber_minutes, red_minutes], outputs=[policy_output, status_cards, family_table])
        add_btn.click(
            add_member,
            inputs=[new_name, new_phone, new_whatsapp, new_city, new_region, new_language, new_role, new_is_coordinator, new_call],
            outputs=[
                add_output,
                member_registry,
                member_storage,
                status_cards,
                family_table,
                care_routes,
                request_member_picker,
                relay_member,
                tts_member,
                detail_member,
                policy_member,
                affiliation_subject,
                affiliation_related,
                edit_member,
            ],
        )
        clear_data_btn.click(
            clear_data,
            outputs=[
                admin_output,
                status_cards,
                requests,
                family_table,
                care_routes,
                alerts,
                member_registry,
                member_storage,
                request_member_picker,
                relay_member,
                tts_member,
                detail_member,
                policy_member,
                affiliation_subject,
                affiliation_related,
                edit_member,
            ],
        )

    return demo


def build_server_app():
    server = install_webhook_routes(FastAPI())
    return gr.mount_gradio_app(server, build_app(), path="/", css=CUSTOM_CSS, theme=APP_THEME, ssr_mode=False)


app = build_server_app()
demo = app


if __name__ == "__main__":
    import uvicorn

    server_port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=server_port)
