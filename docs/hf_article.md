# Ani Kɛse: a small-model care loop for Ghanaian families

Ani Kɛse means "big eye" in Twi. The idea is simple: a family should not have to wait until something is obviously wrong before someone checks in.

In many Ghanaian families, care already works through a living network: aunties, uncles, siblings, cousins, children abroad, neighbors, church friends, and the one person who somehow coordinates everyone. The problem is not that nobody cares. The problem is that the signal is scattered.

Someone has not replied for days. Someone says "I'm fine" when they are not. A relative promises to visit, but nobody records whether the visit happened. The coordinator keeps a mental checklist across WhatsApp chats, phone calls, and family memory.

Ani Kɛse turns that into an autonomous care loop.

It watches for silence, sends check-ins, routes follow-up to the right relative, accepts text or voice replies, translates Akan/Twi/Fante responses for structured analysis, and keeps the loop open until a human confirms what happened.

Space: https://huggingface.co/spaces/build-small-hackathon/family-care-network  
GitHub: https://github.com/teckedd-code2save/adwuma-pa

## What the system does

Ani Kɛse is built around one operating question:

> Who needs attention, who should be contacted, what did they say, and has the family closed the loop?

The coordinator registers family members, locations, WhatsApp numbers, preferred language, and care affiliations. A family member can also be a coordinator. Affiliations describe who can help whom: first-party contact, nearby relative, caregiver, emergency contact, primary coordinator, or backup coordinator.

Each family member has their own care policy:

- Routine check-in after a configured number of minutes.
- Ask family to check soon after a longer silence window.
- Urgent follow-up after the urgent window.
- Daily message caps for routine, check-soon, and urgent messages.
- A do-not-notify exclusion list for people who should be skipped by autopilot.

Autopilot runs on a Modal cron every 30 minutes. The app also has its own scan interval setting, with a tolerance so scheduled ticks do not skip just because the previous scan completed a few seconds late.

When a scan finds a due family member, Ani Kɛse opens or reuses one care case. It does not keep creating duplicate requests. It chooses the strongest stage for that scan, so urgent follow-up supersedes check-soon and routine reminders.

Then it chooses the best contact path:

- Send a check-in link directly to the person.
- Ask an assigned relative to check on them.
- Keep the case open if the frequency cap has already been reached.
- Skip completely if the person is excluded from autopilot.

## The user experience

The interface is deliberately small:

- **Overview** shows current urgencies, request/reply threads, and closure actions.
- **Family** manages members, coordinators, and affiliations.
- **Settings** controls autopilot, Twilio, Modal health, care policies, and frequency caps.

The care board is designed like a family conversation, not like a spreadsheet. Each request is grouped with its reply:

- who Ani Kɛse contacted
- why the request was sent
- when it was sent
- delivery state
- whether the reply is still waiting
- reply time
- concern score
- summary and evidence
- what the coordinator should do next

Raw `/checkin/...` paths are hidden behind Open link and Copy link controls. The coordinator sees a clean care thread; WhatsApp keeps the actual link clickable for the person receiving it.

## Why WhatsApp matters

For this use case, the channel matters as much as the model.

Most families will not open a new dashboard every day. They will answer a WhatsApp message. That is why Ani Kɛse uses Twilio WhatsApp for check-in links and relative nudges.

The messages are intentionally human:

> Hi Auntie Afia, Ani Kɛse is checking in because we have not heard from you recently. Please send a short update.

> Hi Edward, Ani Kɛse has not heard from Auntie Afia for a while. Could you check on her and send a short family update?

> Hi Edward, this needs urgent follow-up: Ani Kɛse has not heard from Auntie Afia past her urgent window. Please confirm she is okay.

The app avoids leaking internal reason codes like `first_party_amber_silence` into family messages. The person should understand the request without understanding the implementation.

## Small-model stack

The hackathon constraint made the design better. Instead of using one huge model for everything, Ani Kɛse separates the workflow into smaller pieces.

Current stack:

- ASR: `facebook/mms-1b-all`
- Translation: NLLB-200 distilled 600M family for Akan/Twi/Fante to English evaluation
- Concern analysis: `Qwen2.5-7B-Instruct`
- TTS: MMS TTS
- UI: Gradio on Hugging Face Spaces
- Delivery: Twilio WhatsApp
- Inference and cron: Modal

Approximate model budget:

- ASR: 1.0B
- Translation: 0.6B
- LLM: 7.0B
- TTS: 1.2B
- Total: about 9.8B, under the 32B cap

The important rule is that Qwen is not asked to magically understand an opaque audio file. The intended pipeline is:

1. Receive text or voice.
2. If voice, transcribe with ASR.
3. Translate Akan/Twi/Fante to English when needed.
4. Send the original text, English translation, family context, silence duration, request reason, and recent history to Qwen.
5. Require strict JSON output.
6. Store the result with evidence.

The structured output looks like this:

```json
{
  "summary": "...",
  "concern_level": 0,
  "flags": [],
  "sentiment": "stable",
  "evidence": [],
  "recommended_action": "normal",
  "confidence": "low|medium|high"
}
```

If transcription, translation, or analysis fails, Ani Kɛse does not invent a score. It marks the response as `needs_review` and keeps the original reply visible.

## Autonomy without notification spam

The most useful version of this product is not a manual check-in form. A coordinator can already send a WhatsApp message manually.

The value is the autonomous loop:

1. Scan active family members.
2. Compare last contact with each person's care policy.
3. Open or reuse one care case.
4. Pick the best route.
5. Send a WhatsApp request only if the frequency cap allows it.
6. Receive a response.
7. Analyze the response with evidence.
8. Resolve low-concern requests or keep concerning cases open.
9. Nudge the right relative for urgent cases.
10. Close the loop only when a human confirms what happened.

Frequency caps are critical. A family care system should not become noisy enough that people ignore it.

Ani Kɛse keeps separate daily caps for routine, check-soon, and urgent messages. Frequency caps affect WhatsApp sends, not case visibility. So if a person has reached their urgent-message cap, the case still appears on the board, but the system does not keep sending duplicate WhatsApp messages.

## What happens when a reply comes in

Replies are linked to the exact request they answer. This matters because a family can receive multiple nudges over time.

If a reply comes in early, it is attached to the open request and analysis runs immediately.

If a reply comes in late, it is still attached to its original request when possible. The UI keeps the request/reply pair together so the coordinator can see what the reply was answering.

After analysis:

- Low concern can close the request and optionally resolve the case.
- Needs-attention replies keep the case open with evidence.
- Urgent replies keep the case open and recommend escalation.
- Low-confidence or failed model output becomes human review.

This minimizes false positives in three ways:

- The system preserves evidence instead of showing only a score.
- It uses family policy and silence duration, not only sentiment.
- It lets humans close the loop after real-world confirmation.

## Cost control

Modal is used carefully because inference credits are finite.

The API uses scale-to-zero behavior, lazy model loading, and small endpoint boundaries:

- `/health`
- `/translate`
- `/transcribe`
- `/analyze`
- `/speak`

Cron is deployed for autonomous scans, but the app-level scan interval, do-not-notify list, and frequency caps control actual work. During development, heavy tests were run manually and stopped when done.

## Built with Codex

This project was built with Codex as an active coding partner. The commit history is intentionally public so judges can see the build trail:

- UI simplification from a bulky dashboard into a three-tab care console.
- Persistent SQLite storage mounted on the Space bucket.
- Twilio WhatsApp delivery and sandbox onboarding guidance.
- Modal inference API and 30-minute cron deployment.
- Request-backed check-in links.
- Request/reply grouping.
- Per-person frequency caps.
- Human-readable WhatsApp templates.
- Honest `needs_review` states when models fail.

Codex was not used only for code generation. It was used as a debugging and product-shaping loop: reading the spec, inspecting the running Space, fixing persistence, tracing cron behavior, tightening copy, and keeping the implementation aligned with the care workflow.

## What still needs work

The biggest open area is Akan/Twi/Fante model quality.

Early ASR experiments showed that generic multilingual models are useful but not enough. A Whisper small fine-tune trained on WaxalNLP Akan ASR produced better subjective results on some phrases, but the current system still needs a cleaner evaluation harness and targeted fine-tuning.

Next steps:

- Build a small family-care phrase evaluation set.
- Fine-tune ASR on GhanaNLP Twi speech and relevant AfriSpeech samples.
- Evaluate Twi/Fante translation with real check-in language, not only generic benchmark text.
- Improve TTS naturalness for Akan/Twi prompts.
- Add production WhatsApp sender approval beyond the sandbox.
- Add richer routing rules using proximity, availability, and relationship strength.

## Why this matters

Ani Kɛse is not trying to replace family judgment. It is trying to reduce the chance that care falls through the cracks.

The best version of the system is quiet most of the time. When everything is fine, it stays out of the way. When silence stretches too long or a reply sounds concerning, it brings the right people into the loop, shows the evidence, and asks for closure.

That is the central nervous system I wanted for family care: not just alerts, not just check-ins, but coordinated attention.
