# Adwuma Pa Hackathon Todo

Built from `ADWUMA_PA_SPEC_V2.md` and current progress.

## Current Status

- ASR eval Space is live: `build-small-hackathon/family-care-asr-eval`.
- Community voting was added to compare MMS, the Adwuma Pa fine-tune, and GiftMark Akan Whisper.
- Early ASR finding: the three models are comparable; MMS uses `aka` for Akan rather than separate `twi` or `fat` adapters.
- Main app Space is live: `build-small-hackathon/family-care-network`.
- Main app now starts from real data only: coordinator dashboard, tokenized checkup requests, request-backed responses, alert feed, nudge draft, loop resolution, and configurable silence escalation policy.
- Modal inference is wired through cost-safe client boundaries. If Modal is off, responses are saved as `needs_review` rather than fake-scored.

## Active Execution Plan

1. [x] ASR eval Space and community voting.
2. [x] Main app foundation and SQLite care loop.
3. [x] Configurable silence escalation policy.
4. [x] Codex build provenance for the OpenAI/Codex track.
5. [x] UI polish and status cards.
6. [x] Member detail and history view.
7. [x] Tokenized checkup request workflow.
8. [x] Modal client stubs for ASR, translation, Qwen analysis, and TTS.
9. [x] Local autopilot scan creates real queued requests.
10. [ ] Validate Modal translation endpoint with one real Twi phrase.
11. [ ] Validate Modal ASR/Qwen/TTS endpoints only when ready to spend credits.

## Phase 1: ASR Evaluation

- [x] Create `family-care-asr-eval` Space.
- [x] Add microphone/upload input.
- [x] Add language selector for Twi, Fante, Ghanaian English.
- [x] Add model selector for MMS, Adwuma Pa fine-tune, and GiftMark.
- [x] Fix MMS language code to use `aka`.
- [x] Add community voting.
- [ ] Collect at least 20 community votes across Twi and Fante.
- [ ] Summarize findings in Field Notes.
- [ ] Decide production ASR default.

## Phase 2: Main App Foundation

- [x] Create `family-care-network` Space.
- [x] Add SQLite schema.
- [x] Remove automatic dummy data from the public product path.
- [x] Add coordinator dashboard.
- [x] Add request-backed check-in submission.
- [x] Remove fake concern scoring from the product path; unavailable AI becomes `needs_review`.
- [x] Add alert feed.
- [x] Add first-party nudge draft.
- [x] Add loop resolution.
- [x] Show 9.8B / 32B parameter budget including translation.
- [x] Fix `[object Object]` table rendering.
- [x] Improve layout and visual polish for the demo.
- [x] Add member detail/history view.
- [x] Add status cards for green, reminder, amber, red counts.
- [ ] Add clearer demo flow controls for judges.

## Phase 3: Wire Speech And Translation Into Main App

- [x] Add voice input to the request-backed check-in tab.
- [x] Reuse selected ASR model path from eval results.
- [x] Store transcript, translation, model status, and review error when inference is unavailable.
- [x] Keep text check-in as reliable fallback.
- [ ] Run one Modal ASR test with real audio.
- [ ] Run one Modal translation test with real Twi/Fante.
- [ ] Run one Modal Qwen test after translation succeeds.

## Phase 4: Real Relay Workflow

- [ ] Add contact assignment UI.
- [ ] Add Twilio WhatsApp sandbox config.
- [x] Generate real request links and tokens.
- [ ] Send real check-in link draft or sandbox WhatsApp message.
- [ ] Save nudge records in SQLite.
- [ ] Add field report submission and attach it to the elder.

## Phase 5: Silence Detection

- [x] Implement scheduled scan logic.
- [x] Add configurable per-member reminder, amber, and red intervals.
- [x] Create reminder alert after configured reminder interval.
- [x] Create amber alert after configured amber interval.
- [x] Create red alert after configured red interval.
- [x] Add manual "run silence scan now" button that uses the same scan logic as cron.
- [x] Add Modal Cron skeleton, not deployed during development.

## Phase 6: Voice Call Engine

- [ ] Add call records table UI.
- [ ] Add Twilio outbound call stub.
- [x] Add MMS TTS greeting/close prompt generation through Modal boundary.
- [ ] Add call transcript ingestion path.
- [ ] Keep this as a stretch feature if time gets tight.

## Phase 7: Submission Package

- [x] Add Codex build log.
- [x] Add in-app Codex build trace tab.
- [x] Add README provenance section.
- [ ] Publish Field Notes write-up.
- [ ] Record 60-second demo video.
- [ ] Share community ASR eval link.
- [ ] Share main app link.
- [ ] Write social post.
- [ ] Submit to Backyard AI track.
- [ ] Explicitly mention OpenAI/Codex build process and agentic care loop.

## Winning Focus

- The demo must show a full loop: autopilot request -> elder response -> ASR/translation/Qwen analysis -> alert -> nearest relative nudge -> field report -> resolved loop.
- Do not over-index on perfect ASR. The winning story is resilient family care with small models and honest `needs_review` fallback behavior.
- The OpenAI track angle is the Codex-built workflow plus agentic routing: monitor, interpret, assign human follow-up, and close the loop.
