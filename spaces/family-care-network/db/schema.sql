CREATE TABLE IF NOT EXISTS members (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  phone TEXT NOT NULL,
  whatsapp TEXT NOT NULL,
  location_city TEXT,
  location_region TEXT,
  language TEXT DEFAULT 'twi',
  family_role TEXT DEFAULT 'relative',
  is_coordinator INTEGER DEFAULT 0,
  checkin_url_token TEXT UNIQUE,
  active INTEGER DEFAULT 1,
  escalation_days_amber INTEGER DEFAULT 7,
  escalation_days_red INTEGER DEFAULT 14,
  reminder_minutes INTEGER DEFAULT 10080,
  escalation_minutes_amber INTEGER DEFAULT 14400,
  escalation_minutes_red INTEGER DEFAULT 20160,
  call_enabled INTEGER DEFAULT 1,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS first_party_contacts (
  id TEXT PRIMARY KEY,
  elder_id TEXT NOT NULL REFERENCES members(id),
  contact_id TEXT NOT NULL REFERENCES members(id),
  priority INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS member_affiliations (
  id TEXT PRIMARY KEY,
  subject_member_id TEXT NOT NULL REFERENCES members(id),
  related_member_id TEXT NOT NULL REFERENCES members(id),
  relationship TEXT NOT NULL,
  care_role TEXT DEFAULT 'family',
  priority INTEGER DEFAULT 5,
  can_coordinate INTEGER DEFAULT 0,
  notes TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(subject_member_id, related_member_id, relationship)
);

CREATE TABLE IF NOT EXISTS checkins (
  id TEXT PRIMARY KEY,
  member_id TEXT NOT NULL REFERENCES members(id),
  request_id TEXT REFERENCES checkup_requests(id),
  submitted_at TEXT NOT NULL,
  input_type TEXT NOT NULL,
  raw_input TEXT,
  transcript TEXT,
  translation TEXT,
  analysis_status TEXT DEFAULT 'needs_review',
  analysis_json TEXT,
  processing_error TEXT,
  asr_model_used TEXT,
  asr_confidence REAL,
  summary TEXT,
  concern_level INTEGER,
  flags TEXT DEFAULT '[]',
  language_detected TEXT,
  source TEXT DEFAULT 'self'
);

CREATE TABLE IF NOT EXISTS checkup_requests (
  id TEXT PRIMARY KEY,
  token TEXT NOT NULL UNIQUE,
  member_id TEXT NOT NULL REFERENCES members(id),
  requester TEXT DEFAULT 'Adwuma Pa autopilot',
  request_type TEXT DEFAULT 'elder_checkin',
  reason_code TEXT NOT NULL,
  reason_detail TEXT,
  channel TEXT DEFAULT 'web',
  status TEXT DEFAULT 'pending',
  priority TEXT DEFAULT 'routine',
  created_at TEXT NOT NULL,
  expires_at TEXT,
  completed_at TEXT,
  related_alert_id TEXT REFERENCES alerts(id),
  related_nudge_id TEXT REFERENCES nudges(id)
);

CREATE TABLE IF NOT EXISTS model_runs (
  id TEXT PRIMARY KEY,
  checkin_id TEXT REFERENCES checkins(id),
  run_type TEXT NOT NULL,
  model_id TEXT,
  status TEXT NOT NULL,
  latency_ms INTEGER,
  input_summary TEXT,
  output_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY,
  member_id TEXT NOT NULL REFERENCES members(id),
  alert_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  resolved INTEGER DEFAULT 0,
  resolved_at TEXT,
  resolved_by TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS calls (
  id TEXT PRIMARY KEY,
  member_id TEXT NOT NULL REFERENCES members(id),
  initiated_at TEXT NOT NULL,
  duration_seconds INTEGER,
  transcript TEXT,
  asr_model_used TEXT,
  summary TEXT,
  concern_level INTEGER,
  twilio_call_sid TEXT,
  status TEXT DEFAULT 'initiated'
);

CREATE TABLE IF NOT EXISTS nudges (
  id TEXT PRIMARY KEY,
  elder_id TEXT NOT NULL REFERENCES members(id),
  contact_id TEXT REFERENCES members(id),
  request_id TEXT REFERENCES checkup_requests(id),
  sent_at TEXT NOT NULL,
  responded_at TEXT,
  checkin_id TEXT REFERENCES checkins(id)
);

CREATE TABLE IF NOT EXISTS inbound_messages (
  id TEXT PRIMARY KEY,
  sender TEXT,
  channel TEXT DEFAULT 'whatsapp',
  body TEXT,
  matched_member_id TEXT REFERENCES members(id),
  matched_contact_id TEXT REFERENCES members(id),
  status TEXT DEFAULT 'unmatched',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outbound_messages (
  id TEXT PRIMARY KEY,
  request_id TEXT REFERENCES checkup_requests(id),
  recipient_member_id TEXT REFERENCES members(id),
  channel TEXT DEFAULT 'whatsapp',
  recipient TEXT,
  body TEXT,
  provider_sid TEXT,
  status TEXT NOT NULL,
  error TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TEXT NOT NULL
);
