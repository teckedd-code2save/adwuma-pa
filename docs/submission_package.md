# Ani Kese Submission Package

## Hugging Face Article Draft

### Ani Kese: a small-model care loop for Ghanaian families

Ani Kese means "big eye" in Twi. The project is a small AI care network for families who want to look after parents, grandparents, aunties, uncles, siblings, and other relatives without turning care into a dashboard chore.

The problem is simple: families often notice too late. Someone has not replied for a while. A relative says "I'm fine" when they are not. A coordinator sends reminders manually and then has to remember who replied, who was asked to visit, and whether the loop was closed.

Ani Kese turns that into an autonomous care loop:

1. A coordinator registers family members, WhatsApp numbers, locations, care policy, and affiliations.
2. Autopilot scans every 30 minutes and compares each active family member with their own care policy.
3. If someone is due, Ani Kese opens or reuses one care case and chooses the best contact path.
4. It sends a WhatsApp check-in link to the person or asks an assigned relative to check on them.
5. Replies can be text or voice. Voice goes through ASR, Akan/Twi/Fante text is translated to English, and Qwen produces structured concern analysis with evidence.
6. Low-concern replies can close the request. Concerning replies stay open with a clear reason. Urgent silence nudges a coordinator and the closest assigned relative.
7. The coordinator sees request/reply pairs grouped like a conversation and closes the loop with a note.

The small-model stack stays under the 32B parameter budget:

- ASR: MMS-1B-all for multilingual speech recognition experiments.
- Translation: NLLB-200 distilled 600M family for Akan/Twi/Fante to English evaluation.
- Concern analysis: Qwen2.5-7B-Instruct for strict JSON scoring after translation.
- TTS: MMS TTS for generated check-in prompts.
- Delivery and orchestration: Twilio WhatsApp, Hugging Face Space UI, Modal serverless inference and cron.

The most important design choice is honesty. If ASR, translation, or analysis fails, the app does not invent a concern score. It marks the response as `needs_review` and keeps the original response visible.

The current system is intentionally conservative:

- Modal containers scale down quickly.
- Cron wakes every 30 minutes, but the app has its own scan interval and per-person message caps.
- "Do not notify" is a hard skip.
- Existing open requests are reused instead of sending duplicate links.
- Request/reply threads show exactly who was contacted, when, whether they replied, and what action remains.

Limitations remain. Twi/Fante ASR and translation still need better fine-tuning and evaluation. WhatsApp sandbox onboarding is still a constraint for public testing. The next model work is a targeted ASR fine-tune on GhanaNLP and AfriSpeech samples, plus a translation evaluation set with real family-care phrases.

Ani Kese is not trying to replace family judgment. It is trying to become the family's central nervous system: watching for silence, routing attention, preserving evidence, and making sure care loops actually close.

## 60-90 Second Demo Story

1. Open the Space and show the Overview: no open priorities, model budget, and three tabs.
2. Add three family members: one coordinator, one person being watched, one nearby relative.
3. Add the affiliation: the watched person is linked to the nearby relative as a first-party contact.
4. In Settings, show the care policy: routine check-in, check-soon, urgent follow-up, daily message caps, and 30-minute autopilot cadence.
5. Run scan now or wait for cron. Ani Kese opens one care case and sends a WhatsApp check-in or relative update.
6. Open WhatsApp and show the human message. No internal reason code, just a clear family request and link.
7. Submit a short text or voice response through the link.
8. Return to Overview. Show the request and reply nested together, with sent time, reply time, concern score, evidence, and action guidance.
9. Close the case with a note. Show the board returning to all clear.
10. End on the line: "Ani Kese watches for silence, routes help, and closes the loop."

## Social Post Draft

Built Ani Kese for the Build Small Hackathon: a small-model care loop for Ghanaian families.

It watches for silence, sends WhatsApp check-ins, routes urgent follow-up to relatives, accepts text or voice replies, translates Akan/Twi/Fante to English for analysis, and keeps request/reply evidence grouped until the family closes the loop.

Stack: Hugging Face Spaces, Modal, Twilio WhatsApp, MMS ASR/TTS, NLLB translation, Qwen structured analysis.

The goal is not another dashboard. It is a central nervous system for family care.

Space: https://huggingface.co/spaces/build-small-hackathon/family-care-network
GitHub: https://github.com/teckedd-code2save/adwuma-pa
Article: https://huggingface.co/blog/build-small-hackathon/ani-kese-small-model-care-loop
