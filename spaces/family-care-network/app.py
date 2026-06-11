from __future__ import annotations

import base64
import html
import io
import json
import os
import re
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, Request, Response
import gradio as gr

from config.models import ASR_CONFIG, LLM_CONFIG, TRANSLATION_CONFIG, TTS_CONFIG, total_parameter_budget_b
from db import database as db
from services.relay import dashboard_rows, route_contact, simulate_nudge
from services.autopilot import autopilot_summary_html, run_autopilot_scan
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
    ("Ani Kɛse Akan Whisper fine-tune", "fine_tuned"),
    ("GiftMark Akan Whisper", "fallback"),
]
ROLE_CHOICES = [
    ("Care recipient", "elder"),
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
.ap-build-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin-top: 12px;
}
.ap-build-panel {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  color: #0f172a;
  padding: 16px;
}
.ap-build-panel h3 {
  color: #0f172a !important;
  font-size: 18px;
  margin: 0 0 8px;
}
.ap-build-panel p,
.ap-build-panel li {
  color: #1e293b !important;
  font-size: 14px;
  line-height: 1.5;
}
.ap-build-panel ul {
  margin: 8px 0 0;
  padding-left: 18px;
}
.ap-autopilot {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  color: #0f172a;
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 10px 0 14px;
  padding: 12px;
}
.ap-autopilot,
.ap-autopilot * {
  color: #0f172a !important;
}
input[type="checkbox"] {
  accent-color: #047857 !important;
  min-height: 18px !important;
  min-width: 18px !important;
}
label:has(input[type="checkbox"]) {
  border: 1px solid #64748b !important;
  border-radius: 6px !important;
  color: #0f172a !important;
  padding: 8px 10px !important;
}
label:has(input[type="checkbox"]:checked) {
  background: #dcfce7 !important;
  border-color: #047857 !important;
}
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

CUSTOM_CSS += """
.ap-cockpit {
  display: grid;
  gap: 14px;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, .75fr);
  margin: 12px 0 16px;
}
.ap-cockpit-main,
.ap-cockpit-side,
.ap-action-panel,
.ap-composer-shell {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 14px;
}
.ap-action-row {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 10px 0 16px;
}
.ap-action-panel {
  min-width: 0;
}
.ap-panel-title {
  color: #0f172a !important;
  font-size: 16px;
  font-weight: 900;
  margin-bottom: 10px;
}
.ap-runbook {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  margin: 10px 0;
}
.ap-runbook article {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 12px;
}
.ap-runbook strong {
  color: #0f172a !important;
  display: block;
  font-size: 13px;
  margin-bottom: 8px;
}
.ap-runbook code {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  color: #0f172a !important;
  display: block;
  font-size: 12px;
  line-height: 1.35;
  overflow-wrap: anywhere;
  padding: 8px;
  white-space: pre-wrap;
}
.ap-cockpit-title {
  color: #0f172a !important;
  font-size: 18px;
  font-weight: 900;
  margin: 0 0 10px;
}
.ap-pulse-list {
  display: grid;
  gap: 8px;
}
.ap-pulse-row {
  align-items: center;
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-left: 6px solid #047857;
  border-radius: 8px;
  display: grid;
  gap: 10px;
  grid-template-columns: 1.2fr .85fr 1fr 1.25fr 1fr;
  min-height: 72px;
  padding: 10px 12px;
}
.ap-person-main strong,
.ap-timeline-person strong,
.ap-case-card strong {
  color: #0f172a !important;
  display: block;
  font-weight: 900;
}
.ap-person-main span,
.ap-timeline-person span,
.ap-case-card span {
  color: #475569 !important;
  display: block;
  font-size: 12px;
  line-height: 1.35;
}
.ap-person-status {
  color: #0f172a !important;
  font-size: 13px;
  font-weight: 900;
}
.ap-person-last,
.ap-person-action,
.ap-person-route {
  color: #334155 !important;
  font-size: 13px;
  line-height: 1.35;
}
.ap-urgent {
  border-left-color: #b91c1c !important;
}
.ap-attention {
  border-left-color: #d97706 !important;
}
.ap-check-soon {
  border-left-color: #b45309 !important;
}
.ap-routine {
  border-left-color: #047857 !important;
}
.ap-case-stack {
  display: grid;
  gap: 10px;
}
.ap-case-card {
  background: #ffffff;
  border: 1px solid #94a3b8;
  border-left: 6px solid #b45309;
  border-radius: 8px;
  padding: 12px;
}
.ap-case-head {
  align-items: flex-start;
  display: flex;
  gap: 10px;
  justify-content: space-between;
  margin-bottom: 8px;
}
.ap-case-head code {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  color: #334155;
  font-size: 11px;
  padding: 4px 6px;
}
.ap-case-next {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  color: #0f172a !important;
  font-size: 13px;
  font-weight: 700;
  margin-top: 10px;
  padding: 8px;
}
.ap-case-empty {
  background: #f8fafc;
  border: 1px dashed #94a3b8;
  border-radius: 8px;
  display: block;
  padding: 14px;
}
.ap-case-empty strong {
  color: #0f172a !important;
  display: block;
  font-size: 15px;
  margin-bottom: 4px;
}
.ap-case-empty span {
  color: #475569 !important;
  display: block;
  font-size: 13px;
}
.ap-timeline {
  background: #ffffff;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 14px;
}
.ap-timeline-person {
  border-bottom: 1px solid #cbd5e1;
  margin-bottom: 12px;
  padding-bottom: 10px;
}
.ap-timeline-event {
  display: grid;
  gap: 10px;
  grid-template-columns: 14px minmax(0, 1fr);
  padding: 0 0 14px;
  position: relative;
}
.ap-timeline-event::before {
  background: #cbd5e1;
  bottom: 0;
  content: "";
  left: 6px;
  position: absolute;
  top: 16px;
  width: 1px;
}
.ap-timeline-event:last-child::before {
  display: none;
}
.ap-timeline-dot {
  background: #047857;
  border-radius: 999px;
  height: 13px;
  margin-top: 3px;
  width: 13px;
}
.ap-timeline-title {
  color: #0f172a !important;
  font-size: 14px;
  font-weight: 900;
}
.ap-timeline-detail {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  color: #334155 !important;
  font-size: 12px;
  line-height: 1.45;
  margin-top: 6px;
  max-height: 90px;
  overflow: auto;
  padding: 8px;
}
.ap-autopilot-strip {
  align-items: end;
  display: grid;
  gap: 10px;
  grid-template-columns: 1fr 150px 1fr 160px 150px;
  margin: 10px 0 14px;
}
.ap-composer-shell {
  margin-top: 12px;
}
.ap-composer-shell .wrap,
.ap-composer-shell .block {
  min-width: 0 !important;
}
@media (max-width: 980px) {
  .ap-cockpit,
  .ap-action-row,
  .ap-autopilot-strip {
    grid-template-columns: 1fr;
  }
  .ap-pulse-row {
    grid-template-columns: 1fr;
  }
}
"""


def refresh_dashboard():
    settings = db.autopilot_settings()
    return (
        autopilot_summary_html(),
        gr.Dropdown(value=settings["enabled"]),
        gr.Number(value=settings["scan_interval_minutes"]),
        gr.Dropdown(value=settings["send_whatsapp"]),
        operations_status_html(),
        status_cards_html(),
        active_requests_html(),
        recent_responses_html(),
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
        gr.Dropdown(choices=alert_choices()),
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


def operations_status_html():
    storage = db.storage_status()
    modal_url = modal_client.modal_base_url()
    modal_result = modal_client.modal_health()
    modal_state = "online" if modal_result.ok else f"not ready: {modal_result.error}"
    twilio_state = "configured" if twilio_client.configured() else "missing SID/token/from"
    sender = twilio_client.configured_from() or "not set"
    return f"""
<section class="ap-autopilot">
  <div><strong>Modal:</strong> {esc(modal_state)}</div>
  <div><strong>Modal URL:</strong> {esc(modal_url or 'not configured')}</div>
  <div><strong>Twilio:</strong> {esc(twilio_state)}</div>
  <div><strong>WhatsApp sender:</strong> {esc(sender)}</div>
  <div><strong>Storage:</strong> {esc(storage['db_path'])}</div>
  <div><strong>Outbound attempts:</strong> {storage['outbound_count']}</div>
</section>
"""


def human_status_class(status):
    return {
        "Routine": "routine",
        "Check soon": "check-soon",
        "Needs attention": "attention",
        "Urgent follow-up": "urgent",
    }.get(status or "", "routine")


def human_priority_label(priority):
    return {
        "routine": "Routine",
        "amber": "Needs attention",
        "red": "Urgent follow-up",
    }.get(priority or "", (priority or "Routine").replace("_", " ").title())


def human_alert_label(alert_type):
    alert_type = (alert_type or "").lower()
    if alert_type == "needs_review":
        return "Review needed"
    if "red" in alert_type:
        if "silence" in alert_type:
            return "Urgent follow-up: no recent check-in"
        return "Urgent follow-up: concerning reply"
    if "amber" in alert_type:
        if "silence" in alert_type:
            return "Needs attention: no recent check-in"
        return "Needs attention: concerning reply"
    if "reminder" in alert_type:
        return "Check soon: reminder due"
    return (alert_type or "Open case").replace("_", " ").title()


def humanize_legacy_text(value, member_name=None):
    text = str(value or "").strip()
    if not text:
        return ""
    person = member_name or "this person"

    def missed_urgent(match):
        return f"We have not heard from {person} for {match.group(1).strip()}. This needs urgent follow-up."

    def missed_attention(match):
        return f"We have not heard from {person} for {match.group(1).strip()}. Please check soon."

    def missed_reminder(match):
        return f"{person} is due for a routine check-in."

    text = re.sub(r"No check-in for ([^;.]+)[.;]\s*red threshold is [^.]+\.?", missed_urgent, text, flags=re.I)
    text = re.sub(r"No check-in for ([^;.]+)[.;]\s*amber threshold is [^.]+\.?", missed_attention, text, flags=re.I)
    text = re.sub(r"No check-in for ([^;.]+)[.;]\s*reminder threshold is [^.]+\.?", missed_reminder, text, flags=re.I)
    text = re.sub(
        r"Ask ([^(]+)\(([^)]+)\) to check on (.+?) after red silence\.?",
        lambda m: f"Ask {m.group(1).strip()} to check on {m.group(3).strip()} and send a short update.",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"Ask ([^(]+)\(([^)]+)\) to check on (.+?) after amber silence\.?",
        lambda m: f"Ask {m.group(1).strip()} to check on {m.group(3).strip()} and send an update soon.",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"Concern score (\d+) from latest check-in\.?",
        "Latest reply was flagged for follow-up, but this older case does not include evidence text. Open the person timeline and recent update before acting.",
        text,
        flags=re.I,
    )
    replacements = {
        "red threshold": "urgent follow-up window",
        "amber threshold": "check-soon window",
        "reminder threshold": "routine check-in window",
        "red silence": "urgent missed check-in",
        "amber silence": "missed check-in",
        "emergency_contact": "emergency contact",
        "primary_coordinator": "primary coordinator",
        "backup_coordinator": "backup coordinator",
        "first_party_contact": "first-party contact",
        "nearby_relative": "nearby relative",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def attention_sort_key(row):
    alert_type = (row.get("Type") or "").lower()
    if "red" in alert_type:
        return 0
    if "amber" in alert_type:
        return 1
    if "needs_review" in alert_type:
        return 2
    if "reminder" in alert_type:
        return 3
    return 4


def family_pulse_html(limit=14):
    rows = dashboard_rows()[:limit]
    if not rows:
        return '<div class="ap-empty">No family members yet. Add people and their care links to start the family pulse.</div>'
    items = []
    for row in rows:
        status = row["Status"]
        status_class = human_status_class(status)
        silent = row.get("Minutes silent") or 0
        last = "No check-in yet" if int(silent) >= 9999 else f"Last heard from about {human_duration_for_ui(silent)} ago"
        items.append(
            f"""
            <article class="ap-pulse-row ap-{status_class}">
              <div class="ap-person-main">
                <strong>{esc(row['Name'])}</strong>
                <span>{esc(row['City'] or row['Region'] or 'Location not set')} · {esc(row.get('Role') or 'family')}</span>
              </div>
              <div class="ap-person-status">{esc(status)}</div>
              <div class="ap-person-last">{esc(last)}</div>
              <div class="ap-person-action">{esc(row['Next action'])}</div>
              <div class="ap-person-route">{esc(row.get('Care route') or 'No care contact assigned')}</div>
            </article>
            """
        )
    return '<section class="ap-pulse-list">' + "\n".join(items) + "</section>"


def attention_queue_html(limit=6):
    open_alerts = [row for row in alert_rows() if row["State"].lower() == "open"]
    open_alerts = sorted(open_alerts, key=attention_sort_key)[:limit]
    if not open_alerts:
        return """
        <section class="ap-case-empty">
          <strong>No open family cases</strong>
          <span>Ani Kɛse will surface missed check-ins, concerning replies, and review-needed updates here.</span>
        </section>
        """
    cards = []
    for row in open_alerts:
        label = human_alert_label(row["Type"])
        notes = alert_note_html(humanize_legacy_text(row["Notes"], row["Member"]))
        cards.append(
            f"""
            <article class="ap-case-card ap-{case_class(row['Type'])}">
              <div class="ap-case-head">
                <div>
                  <strong>{esc(row['Member'])}</strong>
                  <span>{esc(label)}</span>
                </div>
                <code>{esc(row['Created'])}</code>
              </div>
              {notes}
              <div class="ap-case-next">{esc(alert_next_action(row['Type']))}</div>
            </article>
            """
        )
    return '<section class="ap-case-stack">' + "\n".join(cards) + "</section>"


def case_class(alert_type):
    alert_type = (alert_type or "").lower()
    if "red" in alert_type:
        return "urgent"
    if "amber" in alert_type:
        return "attention"
    if "needs_review" in alert_type:
        return "review"
    return "check-soon"


def person_timeline_html(member_id=None, limit=12):
    if not member_id:
        latest = db.one("SELECT id FROM members WHERE active = 1 ORDER BY created_at DESC LIMIT 1")
        member_id = latest["id"] if latest else None
    if not member_id:
        return '<div class="ap-empty">Add a family member to see their care timeline.</div>'
    member = db.one("SELECT * FROM members WHERE id = ?", (member_id,))
    if not member:
        return '<div class="ap-empty">Choose a valid family member.</div>'
    checkins = db.rows(
        """
        SELECT submitted_at AS event_at, 'reply' AS kind, source, input_type, analysis_status,
               concern_level, summary, transcript, translation, processing_error, request_id
        FROM checkins
        WHERE member_id = ?
        ORDER BY submitted_at DESC
        LIMIT ?
        """,
        (member_id, limit),
    )
    requests = db.rows(
        """
        SELECT created_at AS event_at, 'request' AS kind, request_type, reason_code, reason_detail,
               status, priority, token
        FROM checkup_requests
        WHERE member_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (member_id, limit),
    )
    alerts = db.rows(
        """
        SELECT created_at AS event_at, 'case' AS kind, alert_type, notes, resolved, resolved_at
        FROM alerts
        WHERE member_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (member_id, limit),
    )
    events = sorted([*checkins, *requests, *alerts], key=lambda row: row["event_at"] or "", reverse=True)[:limit]
    if not events:
        return f'<div class="ap-empty">No timeline yet for {esc(member["name"])}.</div>'
    rendered = []
    for event in events:
        if event["kind"] == "reply":
            title = "Reply received"
            meta = f"{event['source']} · {event['analysis_status']}"
            if event.get("concern_level") is not None:
                meta += f" · concern {event['concern_level']}/10"
            body = event.get("summary") or "Saved for review."
            detail = event.get("translation") or event.get("transcript") or event.get("processing_error") or ""
        elif event["kind"] == "request":
            title = "Check-in requested"
            meta = f"{friendly_reason(event['reason_code'])} · {human_priority_label(event['priority'])} · {event['status']}"
            body = humanize_legacy_text(event.get("reason_detail"), member["name"]) or "Family check-in requested."
            detail = f"/checkin/{event['token']}"
        else:
            title = human_alert_label(event["alert_type"])
            meta = "Closed" if event.get("resolved") else "Open"
            normalized_notes = humanize_legacy_text(event.get("notes"), member["name"])
            body = first_note_line(normalized_notes)
            detail = normalized_notes
        rendered.append(
            f"""
            <article class="ap-timeline-event">
              <div class="ap-timeline-dot"></div>
              <div>
                <div class="ap-timeline-title">{esc(title)}</div>
                <div class="ap-item-meta">{esc(event['event_at'])} · {esc(meta)}</div>
                <div class="ap-item-note">{esc(body)}</div>
                {f'<div class="ap-timeline-detail">{esc(detail)}</div>' if detail else ''}
              </div>
            </article>
            """
        )
    return f"""
    <section class="ap-timeline">
      <div class="ap-timeline-person">
        <strong>{esc(member['name'])}</strong>
        <span>{esc(member.get('location_city') or 'Location not set')} · {esc(member.get('language') or 'language unset')}</span>
      </div>
      {''.join(rendered)}
    </section>
    """


def first_note_line(notes):
    lines = [line.strip() for line in (notes or "").splitlines() if line.strip()]
    return lines[0] if lines else "Family case opened."


def system_runbook_html():
    modal_url = modal_client.modal_base_url() or "<modal-web-base-url>"
    return f"""
<section class="ap-runbook">
  <article>
    <strong>Start inference</strong>
    <code>modal deploy modal_backend/adwuma_modal.py</code>
  </article>
  <article>
    <strong>Connect Space</strong>
    <code>hf spaces variables set build-small-hackathon/family-care-network MODAL_API_BASE_URL={esc(modal_url)}</code>
  </article>
  <article>
    <strong>Start cron</strong>
    <code>modal deploy modal_backend/cron.py</code>
  </article>
  <article>
    <strong>Stop after testing</strong>
    <code>modal app stop ani-kese-inference --yes && modal app stop ani-kese-cron --yes</code>
  </article>
</section>
<div class="ap-note">Autopilot can be paused from this app. Modal start/stop stays outside the public Space so billing controls are not exposed to visitors.</div>
"""


def human_duration_for_ui(minutes):
    try:
        minutes = int(minutes)
    except Exception:
        minutes = 0
    if minutes >= 9999:
        return "a long time"
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def cockpit_refresh():
    return (
        family_pulse_html(),
        attention_queue_html(),
        active_requests_html(),
        recent_responses_html(),
        person_timeline_html(None),
        gr.Dropdown(choices=alert_choices()),
        gr.Dropdown(choices=pending_request_choices()),
        operations_status_html(),
    )


def demo_story_html():
    return """
<section>
  <div class="ap-story">
    <h2>Autopilot keeps the family loop moving.</h2>
    <p>
      A coordinator registers the family, Ani Kɛse detects who is due or silent,
      sends a secure check-in, processes the response, routes a relative when needed,
      and keeps the loop open until someone confirms the next action.
    </p>
    <div class="ap-flow">
      <div class="ap-flow-step"><strong>1. Watch</strong><span>Configurable check-in windows per person.</span></div>
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
               r.created_at, m.name, m.location_city,
               c.name AS contact_name,
               c.location_city AS contact_city
        FROM checkup_requests r
        JOIN members m ON m.id = r.member_id
        LEFT JOIN nudges n ON n.id = r.related_nudge_id
        LEFT JOIN members c ON c.id = n.contact_id
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
        detail = humanize_legacy_text(row["reason_detail"], row["name"]) or friendly_reason(row["reason_code"])
        link = f"/checkin/{row['token']}"
        is_report = row["request_type"] == "field_report"
        label = "Relative update" if is_report else "Family check-in"
        responder = (
            f"{row['contact_name']} ({row['contact_city'] or 'location unset'})"
            if is_report and row.get("contact_name")
            else row["name"]
        )
        cards.append(
            f"""
            <article class="ap-item ap-{priority}">
              <div>
                <div class="ap-item-title">{esc(row['name'])}</div>
                <div class="ap-item-meta">{esc(label)} · {esc(friendly_reason(row['reason_code']))} · {esc(row['status'])}</div>
                <div class="ap-item-note"><strong>Expected responder:</strong> {esc(responder)}</div>
                <div class="ap-item-note">{esc(detail)}</div>
              </div>
              <code>{esc(link)}</code>
            </article>
            """
        )
    return '<section class="ap-list">' + "\n".join(cards) + "</section>"


def family_overview_html(limit=12):
    rows = dashboard_rows()[:limit]
    if not rows:
        return '<div class="ap-empty">No family members yet. Add the first person in Members.</div>'
    cards = []
    for row in rows:
        status = human_status_class(row["Status"])
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
        return '<div class="ap-empty">No family members registered. Add members, then add affiliations to relatives who can respond.</div>'
    items = []
    for row in rows:
        member = db.one("SELECT id FROM members WHERE name = ? AND active = 1 LIMIT 1", (row["Name"],))
        contact = route_contact(member["id"]) if member else None
        if contact:
            role = (contact.get("care_role") or "family").replace("_", " ")
            note = f"Valid route: {esc(row['Name'])} -> {esc(contact['name'])} ({esc(role)}, priority {esc(contact.get('affiliation_priority') or 1)})"
            state = "valid"
        else:
            note = "Missing route: add an affiliation where this person is being checked on and the related family member has a care role such as first-party contact or nearby relative."
            state = "needs route"
        items.append(
            f"""
            <article class="ap-item">
              <div>
                <div class="ap-item-title">{esc(row['Name'])}</div>
                <div class="ap-item-meta">{note}</div>
              </div>
              <span class="ap-state">{state}</span>
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
        notes = alert_note_html(row["Notes"])
        items.append(
            f"""
            <article class="ap-item ap-alert">
              <div>
                <div class="ap-item-title">{esc(row['Member'])}</div>
                <div class="ap-item-meta">{esc(row['Type'])} · {esc(state_label)}</div>
                {notes}
                <div class="ap-item-note"><strong>Next:</strong> {action}</div>
              </div>
              <span class="ap-state">{state_label.lower()}</span>
            </article>
            """
        )
    return '<section class="ap-list">' + "\n".join(items) + "</section>"


def alert_note_html(notes):
    lines = [line.strip() for line in (notes or "").splitlines() if line.strip()]
    if not lines:
        return '<div class="ap-item-note">No details yet.</div>'
    rendered = []
    for line in lines[:6]:
        if ":" in line:
            label, value = line.split(":", 1)
            rendered.append(f"<div class=\"ap-item-note\"><strong>{esc(label)}:</strong>{esc(value)}</div>")
        else:
            rendered.append(f"<div class=\"ap-item-note\">{esc(line)}</div>")
    return "\n".join(rendered)


def alert_next_action(alert_type):
    alert_type = (alert_type or "").lower()
    if alert_type.startswith("red"):
        return "Call this person or ask the assigned relative to confirm they are okay, then close the case with a note."
    if alert_type.startswith("amber"):
        return "Ask the assigned relative to check in, then close the case once the family has a confirmed update."
    if alert_type == "needs_review":
        return "Review the saved response, translation, and model error before deciding the next family action."
    if alert_type.startswith("reminder"):
        return "Send or resend a check-in link, then close after a response arrives."
    return "Close after the coordinator confirms the next family action."


def friendly_reason(reason):
    return {
        "coordinator_request": "Coordinator requested check-in",
        "routine_check": "Routine check-in",
        "reminder_silence": "Time to check in",
        "amber_silence": "Needs attention",
        "red_silence": "Urgent follow-up",
        "first_party_amber_silence": "Relative asked to check in",
        "first_party_red_silence": "Urgent relative update",
    }.get(reason or "", (reason or "Check-in").replace("_", " ").title())


def status_cards_html():
    rows = dashboard_rows()
    counts = {status: 0 for status in ["Routine", "Check soon", "Needs attention", "Urgent follow-up"]}
    for row in rows:
        counts[row["Status"]] = counts.get(row["Status"], 0) + 1
    return f"""
<div class="ap-status-grid">
  <div class="ap-status-card ap-routine"><div class="ap-status-label">Routine</div><div class="ap-status-value">{counts.get("Routine", 0)}</div></div>
  <div class="ap-status-card ap-check-soon"><div class="ap-status-label">Check soon</div><div class="ap-status-value">{counts.get("Check soon", 0)}</div></div>
  <div class="ap-status-card ap-attention"><div class="ap-status-label">Needs attention</div><div class="ap-status-value">{counts.get("Needs attention", 0)}</div></div>
  <div class="ap-status-card ap-urgent"><div class="ap-status-label">Urgent follow-up</div><div class="ap-status-value">{counts.get("Urgent follow-up", 0)}</div></div>
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


def alert_choices():
    rows = db.rows(
        """
        SELECT a.id, m.name, a.alert_type, a.created_at
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
        LIMIT 30
        """
    )
    return [(f"{row['name']} - {human_alert_label(row['alert_type'])} - {row['created_at']}", row["id"]) for row in rows]


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
    <div class="ap-profile-row"><strong>Policy</strong><br>check soon {esc(member.get('reminder_minutes'))} min, needs attention {esc(member.get('escalation_minutes_amber'))} min, urgent follow-up {esc(member.get('escalation_minutes_red'))} min</div>
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


def recent_responses_html(limit=8):
    inbound = db.rows(
        """
        SELECT created_at, sender, body, status
        FROM inbound_messages
        ORDER BY created_at DESC
        LIMIT 20
        """
    )
    rows = db.rows(
        """
        SELECT c.submitted_at, c.source, c.input_type, c.analysis_status, c.concern_level,
               c.summary, c.translation, c.transcript, c.raw_input, c.processing_error,
               m.name,
               r.reason_code,
               r.request_type
        FROM checkins c
        JOIN members m ON m.id = c.member_id
        LEFT JOIN checkup_requests r ON r.id = c.request_id
        ORDER BY c.submitted_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    if not rows and not inbound:
        return '<div class="ap-empty">No responses received yet.</div>'
    cards = []
    grouped_inbound = []
    seen_inbound = {}
    for row in inbound:
        key = (row.get("sender") or "", (row.get("body") or "").strip().lower())
        if key in seen_inbound:
            seen_inbound[key]["count"] += 1
            continue
        item = dict(row)
        item["count"] = 1
        seen_inbound[key] = item
        grouped_inbound.append(item)
    for row in grouped_inbound[:4]:
        match = "Matched to family" if row["status"] == "matched" else "Not matched to a family member"
        count = f" · {row['count']} similar" if row.get("count", 1) > 1 else ""
        cards.append(
            f"""
            <article class="ap-item">
              <div>
                <div class="ap-item-title">WhatsApp update</div>
                <div class="ap-item-meta">{esc(row['created_at'])} · {esc(match)} · {esc(row['sender'])}{esc(count)}</div>
                <div class="ap-item-note"><strong>Message:</strong> {esc(row['body'])}</div>
              </div>
            </article>
            """
        )
    for row in rows:
        concern = "" if row["concern_level"] is None else f" · concern {row['concern_level']}"
        translation = row.get("translation") or ""
        transcript = row.get("transcript") or row.get("raw_input") or ""
        error = row.get("processing_error") or ""
        cards.append(
            f"""
            <article class="ap-item">
              <div>
                <div class="ap-item-title">{esc(row['name'])}</div>
                <div class="ap-item-meta">{esc(row['submitted_at'])} · {esc(row['source'])} · {esc(row['analysis_status'])}{esc(concern)}</div>
                <div class="ap-item-note"><strong>Request:</strong> {esc(friendly_reason(row.get('reason_code')))} · {esc(row.get('request_type') or 'direct')}</div>
                <div class="ap-item-note"><strong>Summary:</strong> {esc(row.get('summary') or 'No summary yet.')}</div>
                <div class="ap-item-note"><strong>Transcript:</strong> {esc(transcript or 'No transcript saved.')}</div>
                <div class="ap-item-note"><strong>Translation:</strong> {esc(translation or 'English input or translation unavailable.')}</div>
                {f'<div class="ap-item-note"><strong>Error:</strong> {esc(error)}</div>' if error else ''}
              </div>
            </article>
            """
        )
    return '<section class="ap-list">' + "\n".join(cards) + "</section>"


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
        family_pulse_html(),
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
        family_pulse_html(),
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
        family_pulse_html(),
        care_routes_html(),
    )


def load_sample_data():
    db.seed_demo_data()
    choices = member_dropdown()
    return (
        "Sample data loaded.",
        status_cards_html(),
        active_requests_html(),
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
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
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
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
    reporter = "nearby relative" if request["request_type"] == "field_report" else "person being checked on"
    return f"""
### Response for {request['member_name']}

Requested by: **{request.get('requester') or 'Ani Kɛse'}**  
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
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
        recent_responses_html(),
        gr.Dropdown(choices=alert_choices()),
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


def resolve_selected_alert(alert_id, resolved_by, notes):
    if not alert_id:
        raise gr.Error("Choose the alert or case to resolve.")
    db.resolve_alert(alert_id, resolved_by or "Coordinator", notes or "Loop closed.")
    return (
        f"Resolved {alert_id}.",
        attention_queue_html(),
        gr.Dropdown(choices=alert_choices(), value=None),
        family_pulse_html(),
        care_routes_html(),
        status_cards_html(),
    )


def nudge(member_id):
    if not member_id:
        raise gr.Error("Choose a family member.")
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
        recent_responses_html(),
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
        gr.Dropdown(choices=alert_choices()),
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


def recent_checkin_choices():
    rows = db.rows(
        """
        SELECT c.id, c.submitted_at, m.name, c.analysis_status
        FROM checkins c
        JOIN members m ON m.id = c.member_id
        ORDER BY c.submitted_at DESC
        LIMIT 30
        """
    )
    return [(f"{row['name']} - {row['submitted_at']} - {row['analysis_status']}", row["id"]) for row in rows]


def load_translation_review(checkin_id):
    if not checkin_id:
        return "", "", "", "Choose a response to review."
    row = db.one(
        """
        SELECT c.raw_input, c.transcript, c.translation, c.summary, c.processing_error, m.name
        FROM checkins c
        JOIN members m ON m.id = c.member_id
        WHERE c.id = ?
        """,
        (checkin_id,),
    )
    if not row:
        return "", "", "", "Response not found."
    original = row.get("transcript") or row.get("raw_input") or ""
    notes = f"Loaded response for {row['name']}."
    if row.get("processing_error"):
        notes += f" Current review note: {row['processing_error']}"
    return original, row.get("translation") or "", row.get("summary") or "", notes


def save_translation_review(checkin_id, corrected_translation):
    if not checkin_id:
        raise gr.Error("Choose a response to review.")
    if not (corrected_translation or "").strip():
        raise gr.Error("Enter the corrected English translation.")
    db.update_checkin_translation(checkin_id, corrected_translation.strip())
    return (
        "Saved corrected translation. Analysis is marked needs_review until rerun.",
        recent_responses_html(),
        gr.Dropdown(choices=recent_checkin_choices(), value=checkin_id),
    )


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
        operations_status_html(),
        status_cards_html(),
        active_requests_html(),
        recent_responses_html(),
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
        gr.Dropdown(choices=alert_choices()),
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
    expected = "relative or caregiver" if request["request_type"] == "field_report" else "person being checked on"
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
  <title>Ani Kɛse Check-in</title>
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
      <h1>Ani Kɛse</h1>
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
        return {"ok": True, "service": "ani-kese-twilio"}

    @server.get("/debug/storage")
    async def debug_storage():
        return db.storage_status()

    @server.post("/api/autopilot/scan")
    async def api_autopilot_scan(request: Request):
        db.init_db()
        expected_secret = os.getenv("ADWUMA_PA_AUTOPILOT_SECRET", "")
        supplied_secret = request.headers.get("X-Adwuma-Secret", "")
        if not expected_secret:
            return Response(
                content=json.dumps({"status": "disabled", "reason": "ADWUMA_PA_AUTOPILOT_SECRET is not configured."}),
                status_code=503,
                media_type="application/json",
            )
        if supplied_secret != expected_secret:
            return Response(
                content=json.dumps({"status": "forbidden", "reason": "Invalid autopilot secret."}),
                status_code=403,
                media_type="application/json",
            )
        result = run_autopilot_scan(force=False, actor="scheduled endpoint")
        return Response(content=json.dumps(result), media_type="application/json")

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
    result = run_autopilot_scan(force=True, actor="Coordinator")
    return (
        autopilot_result_text(result),
        autopilot_summary_html(),
        operations_status_html(),
        status_cards_html(),
        active_requests_html(),
        recent_responses_html(),
        family_pulse_html(),
        care_routes_html(),
        attention_queue_html(),
        gr.Dropdown(choices=alert_choices()),
        gr.Dropdown(choices=pending_request_choices()),
        outbound_table_value(),
    )


def save_autopilot_controls(enabled, scan_interval_minutes, send_whatsapp):
    settings = db.save_autopilot_settings(as_bool(enabled), scan_interval_minutes, as_bool(send_whatsapp))
    mode = "on" if settings["enabled"] else "off"
    delivery = "auto-send WhatsApp" if settings["send_whatsapp"] else "queue links only"
    return (
        f"Autopilot {mode}. Scan interval: {settings['scan_interval_minutes']} minutes. Delivery: {delivery}.",
        autopilot_summary_html(),
        gr.Dropdown(value=settings["enabled"]),
        gr.Number(value=settings["scan_interval_minutes"]),
        gr.Dropdown(value=settings["send_whatsapp"]),
    )


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def autopilot_result_text(result):
    lines = [f"Status: {result.get('status')}"]
    if result.get("reason"):
        lines.append(f"Reason: {result['reason']}")
    actions = result.get("actions") or []
    deliveries = result.get("deliveries") or []
    if actions:
        lines.append("")
        lines.append("Actions:")
        lines.extend(f"- {action}" for action in actions)
    if deliveries:
        lines.append("")
        lines.append("Delivery:")
        lines.extend(f"- {delivery}" for delivery in deliveries)
    return "\n".join(lines)


def update_escalation_settings(member_id, reminder_minutes, amber_minutes, red_minutes):
    if not member_id:
        raise gr.Error("Choose a family member.")
    db.update_escalation(member_id, reminder_minutes, amber_minutes, red_minutes)
    member = db.one("SELECT name, reminder_minutes, escalation_minutes_amber, escalation_minutes_red FROM members WHERE id = ?", (member_id,))
    return (
        f"Updated {member['name']}: reminder {member['reminder_minutes']} min, "
        f"needs attention {member['escalation_minutes_amber']} min, urgent follow-up {member['escalation_minutes_red']} min.",
        gr.Number(value=member["reminder_minutes"]),
        gr.Number(value=member["escalation_minutes_amber"]),
        gr.Number(value=member["escalation_minutes_red"]),
        status_cards_html(),
        family_pulse_html(),
    )


def load_escalation_settings(member_id):
    if not member_id:
        return gr.Number(value=10080), gr.Number(value=14400), gr.Number(value=20160), "Choose a family member."
    member = db.one(
        "SELECT name, reminder_minutes, escalation_minutes_amber, escalation_minutes_red FROM members WHERE id = ?",
        (member_id,),
    )
    if not member:
        return gr.Number(value=10080), gr.Number(value=14400), gr.Number(value=20160), "Member not found."
    return (
        gr.Number(value=member["reminder_minutes"]),
        gr.Number(value=member["escalation_minutes_amber"]),
        gr.Number(value=member["escalation_minutes_red"]),
        f"Loaded {member['name']}: reminder {member['reminder_minutes']} min, needs attention {member['escalation_minutes_amber']} min, urgent follow-up {member['escalation_minutes_red']} min.",
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


def build_notes_html():
    return """
<section class="ap-build-grid">
  <article class="ap-build-panel">
    <h3>Submission positioning</h3>
    <p>This is a Backyard AI project for one real family coordination problem: making sure family members are checked on, understood, routed to the right relative, and followed through.</p>
    <p>The AI is load-bearing in four places: Twi/Fante speech-to-text, Twi/Fante-to-English translation, Qwen structured concern analysis, and choosing the next human action. If Modal is unavailable, the app stores the response as needs_review instead of producing a fake score.</p>
  </article>
  <article class="ap-build-panel">
    <h3>OpenAI track case</h3>
    <p>The project is Codex-built and demonstrates a practical agentic workflow: monitor, interpret, pick the nearest responsible person, escalate, and close the loop.</p>
    <p>Commit history and build notes are kept in the repo so the Codex authorship path is visible.</p>
  </article>
  <article class="ap-build-panel">
    <h3>Implemented by Codex</h3>
    <ul>
      <li>ASR evaluation Space with community voting.</li>
      <li>Main Gradio care dashboard with SQLite persistence.</li>
      <li>Tokenized checkup links, alerts, first-party nudge drafts, and closure flow.</li>
      <li>Configurable check-soon, needs-attention, and urgent-follow-up intervals.</li>
      <li>Modal-safe boundary for ASR, translation, Qwen analysis, and TTS.</li>
    </ul>
  </article>
  <article class="ap-build-panel">
    <h3>Current execution plan</h3>
    <p>Modal is stopped while not testing. Autopilot settings are now in the dashboard; next live session is one controlled end-to-end loop with queue-only delivery first, then optional WhatsApp auto-send.</p>
  </article>
</section>
"""


def build_tts_prompt(member_id, prompt_type, language):
    member = db.one("SELECT * FROM members WHERE id = ?", (member_id,)) if member_id else None
    name = member["name"] if member else "Opanyin"
    if language == "eng":
        templates = {
            "reminder": f"Hello {name}. This is Ani Kɛse checking in. Please send a short update so your family knows how you are doing.",
            "call_greeting": f"Hello {name}. This is Ani Kɛse calling for your family. How are you feeling today?",
            "call_close": "Thank you. Your family will receive this update. We will follow up if anyone needs to check on you.",
        }
    else:
        templates = {
            "reminder": f"{name}, Ani Kɛse re bisa wo ho asɛm. Yɛsrɛ wo, kyerɛw anaa ka sɛnea wo ho te.",
            "call_greeting": f"{name}, Ani Kɛse na ɛrefrɛ wo ama abusua no. Ɛnnɛ wo ho te sɛn?",
            "call_close": "Meda wo ase. Yɛde wo nkra bɛkɔ ama abusua no, na sɛ ɛhia a obi bɛba abɛhwɛ wo.",
        }
    return templates.get(prompt_type, templates["reminder"])


def synthesize_tts_prompt(text, language):
    if not text or not text.strip():
        raise gr.Error("Generate or type TTS text first.")
    result = modal_client.synthesize_speech(text, language)
    if not result.ok:
        raise gr.Error(f"TTS needs review: {result.error}")
    audio = tts_audio_from_modal(result.data)
    status = f"Generated with {result.data.get('model_used', 'Modal TTS')}."
    return audio, status


def tts_audio_from_modal(data):
    encoded = data.get("audio_wav_base64")
    if not encoded:
        raise gr.Error("TTS completed but returned no audio payload.")
    import soundfile as sf

    audio_bytes = base64.b64decode(encoded)
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return int(sample_rate), waveform


def build_app():
    db.init_db()
    with gr.Blocks(title="Ani Kɛse - Family Care Network") as demo:
        gr.HTML(
            """
            <div class="ap-header">
              <div class="ap-title">Ani Kɛse</div>
              <div class="ap-subtitle">
                A small AI care network for Ghanaian families: multilingual check-ins, concern scoring,
                silence detection, nearest-relative nudges, and loop closure for family members who may not ask for help.
              </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                status_cards = gr.HTML(status_cards_html())
                with gr.Row():
                    refresh = gr.Button("Refresh", variant="primary", scale=0)

                with gr.Row(elem_classes=["ap-cockpit"]):
                    with gr.Column(scale=3, elem_classes=["ap-cockpit-main"]):
                        gr.HTML('<div class="ap-cockpit-title">Family Pulse</div>')
                        family_table = gr.HTML(family_pulse_html())
                    with gr.Column(scale=2, elem_classes=["ap-cockpit-side"]):
                        gr.HTML('<div class="ap-cockpit-title">Attention Queue</div>')
                        alerts = gr.HTML(attention_queue_html())
                        alert_picker = gr.Dropdown(choices=alert_choices(), label="Case to close")
                        resolved_by = gr.Textbox(label="Confirmed by", value="")
                        resolution_notes = gr.Textbox(label="What happened", value="", lines=2)
                        resolve_btn = gr.Button("Close selected case", variant="primary")
                        resolve_output = gr.Textbox(label="Closure result", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="ap-section-title">Waiting for replies</div>')
                        requests = gr.HTML(active_requests_html())
                    with gr.Column(scale=1):
                        gr.HTML('<div class="ap-section-title">Recent updates</div>')
                        recent_responses = gr.HTML(recent_responses_html())

                source_state = gr.State("self")
                gr.HTML('<div class="ap-section-title">Care actions</div>')
                with gr.Row(elem_classes=["ap-action-row"]):
                    with gr.Column(scale=1, elem_classes=["ap-action-panel"]):
                        gr.HTML('<div class="ap-panel-title">Send a check-in</div>')
                        request_member_picker = gr.Dropdown(choices=member_choices(), label="Person to check on")
                        with gr.Row():
                            manual_type = gr.Dropdown(
                                choices=[("Ask this person directly", "elder_checkin"), ("Ask a relative for an update", "field_report")],
                                value="elder_checkin",
                                label="Who should answer",
                            )
                            manual_priority = gr.Dropdown(
                                choices=[("Routine", "routine"), ("Please check soon", "amber"), ("Needs urgent follow-up", "red")],
                                value="routine",
                                label="Care level",
                            )
                        manual_reason = gr.Dropdown(
                            choices=[
                                ("Coordinator request", "coordinator_request"),
                                ("Routine check", "routine_check"),
                                ("Time to check in", "reminder_silence"),
                                ("Please check soon", "amber_silence"),
                                ("Needs urgent follow-up", "red_silence"),
                            ],
                            value="coordinator_request",
                            label="Reason",
                        )
                        manual_detail = gr.Textbox(label="Message", lines=3, value="Please send a short update so the family knows how you are doing.")
                        with gr.Row():
                            create_request_btn = gr.Button("Create link", variant="primary")
                            send_whatsapp_btn = gr.Button("Send by WhatsApp", variant="primary")
                        create_request_output = gr.Markdown()
                        send_request_picker = gr.Dropdown(choices=pending_request_choices(), label="Pending request")
                        send_whatsapp_output = gr.Textbox(label="Send result", interactive=False)
                    with gr.Column(scale=1, elem_classes=["ap-action-panel"]):
                        gr.HTML('<div class="ap-panel-title">Save a reply</div>')
                        with gr.Row():
                            request_token = gr.Textbox(label="Check-in link", placeholder="/checkin/...")
                            load_request = gr.Button("Find", variant="primary")
                        request_context = gr.Markdown("Find the request, then type or record the update in one place.")
                        with gr.Row():
                            request_member = gr.Textbox(label="Person checked on", interactive=False)
                            request_reason = gr.Textbox(label="Reason", interactive=False)
                        with gr.Group(elem_classes=["ap-composer-shell"]):
                            text = gr.Textbox(
                                label="Update",
                                lines=4,
                                placeholder="Type the update here, or record voice below.",
                            )
                            voice_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="numpy",
                                label="Record voice",
                            )
                            with gr.Row():
                                language = gr.Dropdown(
                                    choices=[("Twi", "twi"), ("Fante", "fat"), ("English", "eng")],
                                    value="twi",
                                    label="Language",
                                )
                                input_mode = gr.Radio([("Text", "text"), ("Voice", "voice")], value="text", label="Send as")
                            submit = gr.Button("Send update", variant="primary")
                        receipt = gr.Textbox(label="Result", interactive=False, lines=2)
                        ai_json = gr.Code(label="Care processing result", language="json", visible=False)
                        with gr.Accordion("Review translation", open=False):
                            translation_checkin = gr.Dropdown(choices=recent_checkin_choices(), label="Response")
                            load_translation = gr.Button("Load response")
                            translation_original = gr.Textbox(label="Original / transcript", lines=3, interactive=False)
                            translation_edit = gr.Textbox(label="Corrected English translation", lines=3)
                            translation_summary = gr.Textbox(label="Current summary", lines=2, interactive=False)
                            translation_review_output = gr.Textbox(label="Translation review", interactive=False)
                            save_translation = gr.Button("Save corrected translation", variant="primary")

                gr.HTML('<div class="ap-section-title">Person timeline</div>')
                timeline_member = gr.Dropdown(choices=member_choices(), label="Open a person")
                member_timeline = gr.HTML(person_timeline_html(None))

            with gr.Tab("Family"):
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

            with gr.Tab("Settings"):
                settings = db.autopilot_settings()
                gr.HTML('<div class="ap-section-title">Autopilot settings</div>')
                with gr.Row():
                    autopilot_enabled = gr.Dropdown(
                        choices=[("On", True), ("Off", False)],
                        label="Autopilot",
                        value=settings["enabled"],
                    )
                    autopilot_interval = gr.Number(label="Scan every minutes", value=settings["scan_interval_minutes"], precision=0)
                    autopilot_send_whatsapp = gr.Dropdown(
                        choices=[("Queue links only", False), ("Auto-send WhatsApp", True)],
                        label="WhatsApp delivery",
                        value=settings["send_whatsapp"],
                    )
                with gr.Row():
                    save_autopilot_btn = gr.Button("Save autopilot settings")
                    scan_btn = gr.Button("Run scan now", variant="primary")
                autopilot_status = gr.HTML(autopilot_summary_html())
                autopilot_output = gr.Textbox(label="Last scan result", lines=4, interactive=False)

                gr.HTML('<div class="ap-section-title">Care links</div>')
                care_routes = gr.HTML(care_routes_html())

                with gr.Accordion("Care policy and utilities", open=False):
                    with gr.Row():
                        relay_member = gr.Dropdown(choices=member_choices(), label="Person needing follow-up")
                        nudge_btn = gr.Button("Draft relative nudge", variant="primary")
                    nudge_output = gr.Textbox(label="WhatsApp nudge draft", lines=4, interactive=False)
                    gr.HTML('<div class="ap-section-title">Escalation timing</div>')
                    policy_member = gr.Dropdown(choices=member_choices(), label="Family member")
                    with gr.Row():
                        reminder_minutes = gr.Number(label="Reminder after minutes", value=10080, precision=0)
                        amber_minutes = gr.Number(label="Please check soon after minutes", value=14400, precision=0)
                        red_minutes = gr.Number(label="Urgent follow-up after minutes", value=20160, precision=0)
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

                gr.HTML('<div class="ap-section-title">System status</div>')
                operations_status = gr.HTML(operations_status_html())
                budget = gr.HTML(model_budget_markdown())
                modal_status = gr.Markdown(modal_health_markdown())
                gr.HTML('<div class="ap-section-title">Startup and shutdown</div>')
                gr.HTML(system_runbook_html())
                with gr.Accordion("Delivery log and data controls", open=False):
                    outbound_messages = gr.Dataframe(headers=OUTBOUND_HEADERS, value=outbound_table_value(), label="Recent WhatsApp attempts", interactive=False, wrap=True)
                    gr.Markdown("Production data starts empty. This only clears records; it never loads dummy data.")
                    clear_data_btn = gr.Button("Clear all data", variant="stop")
                    admin_output = gr.Textbox(label="Admin action", interactive=False)

        refresh.click(
            refresh_dashboard,
            outputs=[
                autopilot_status,
                autopilot_enabled,
                autopilot_interval,
                autopilot_send_whatsapp,
                operations_status,
                status_cards,
                requests,
                recent_responses,
                family_table,
                care_routes,
                alerts,
                alert_picker,
                modal_status,
                budget,
            ],
        )
        save_autopilot_btn.click(
            save_autopilot_controls,
            inputs=[autopilot_enabled, autopilot_interval, autopilot_send_whatsapp],
            outputs=[autopilot_output, autopilot_status, autopilot_enabled, autopilot_interval, autopilot_send_whatsapp],
        )
        timeline_member.change(person_timeline_html, inputs=[timeline_member], outputs=[member_timeline])
        scan_btn.click(
            run_silence_scan,
            outputs=[
                autopilot_output,
                autopilot_status,
                operations_status,
                status_cards,
                requests,
                recent_responses,
                family_table,
                care_routes,
                alerts,
                alert_picker,
                send_request_picker,
                outbound_messages,
            ],
        )
        load_request.click(
            load_request_context,
            inputs=[request_token],
            outputs=[request_context, language, request_member, request_reason, source_state],
        )
        submit.click(
            submit_checkin_by_token,
            inputs=[request_token, language, text, voice_audio, input_mode, source_state],
            outputs=[receipt, ai_json, status_cards, requests, family_table, care_routes, alerts, recent_responses, alert_picker],
        )
        load_translation.click(
            load_translation_review,
            inputs=[translation_checkin],
            outputs=[translation_original, translation_edit, translation_summary, translation_review_output],
        )
        save_translation.click(
            save_translation_review,
            inputs=[translation_checkin, translation_edit],
            outputs=[translation_review_output, recent_responses, translation_checkin],
        )
        resolve_btn.click(
            resolve_selected_alert,
            inputs=[alert_picker, resolved_by, resolution_notes],
            outputs=[resolve_output, alerts, alert_picker, family_table, care_routes, status_cards],
        )
        create_request_btn.click(
            create_manual_request,
            inputs=[request_member_picker, manual_reason, manual_detail, manual_type, manual_priority],
            outputs=[create_request_output, status_cards, requests, recent_responses, family_table, care_routes, alerts, alert_picker, send_request_picker],
        )
        send_whatsapp_btn.click(
            send_checkin_whatsapp,
            inputs=[send_request_picker],
            outputs=[send_whatsapp_output, operations_status, status_cards, requests, recent_responses, family_table, care_routes, alerts, alert_picker, outbound_messages, send_request_picker],
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
        policy_member.change(load_escalation_settings, inputs=[policy_member], outputs=[reminder_minutes, amber_minutes, red_minutes, policy_output])
        policy_btn.click(
            update_escalation_settings,
            inputs=[policy_member, reminder_minutes, amber_minutes, red_minutes],
            outputs=[policy_output, reminder_minutes, amber_minutes, red_minutes, status_cards, family_table],
        )
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
