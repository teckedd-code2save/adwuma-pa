# Ani Kɛse Submission Plan

## One-Line Pitch

Ani Kɛse is a small AI family care network that checks on Ghanaian elders in Twi, Fante, or English, detects concern or silence, and routes follow-up to the nearest relative until the loop is closed.

## Track

Backyard AI.

## Awards To Target

- OpenAI Track: Codex-built app plus documented agent trace and agentic care workflow.
- Backyard AI podium: specific real person, real family workflow, and measurable loop closure.
- Best Agent: monitor, interpret, route, escalate, and close-the-loop behavior.
- Off-Brand: custom Gradio UI beyond default styling.
- Field Notes: publish `FIELD_NOTES.md` as a Hugging Face article.
- Well-Tuned: wire the published `teckedd/whisper_small-waxal_akan-asr-v1` model into the ASR eval path.
- Best Demo: lead with the human story, show the red scenario, show the nudge, then close the loop.

## Demo Video Script

1. "In many Ghanaian families, elders do not always say when something is wrong. By the time news travels, it can be too late."
2. Show the dashboard with green family members.
3. Submit a check-in from Uncle Kwame: "My chest hurts and I have been alone in the house. I cannot walk well."
4. Show the red alert and concern JSON.
5. Open Relay and draft the WhatsApp nudge to Ama, the nearest relative.
6. Resolve the loop with a field report.
7. Close: "Ani Kɛse means Big Eye. Small AI, routed to real family action."

## Social Post Draft

I built Ani Kɛse for the Build Small Hackathon: a small-model family care network for Ghanaian elders.

It accepts Twi/Fante/English check-ins, scores health and isolation concern, watches for silence, and nudges the nearest relative until the care loop is closed.

Small models should make real family work easier.

Space: <link>
Demo: <link>

## Judge Checklist

- The app runs immediately on a public Gradio Space.
- Parameter budget is visible in the UI and stays under 32B.
- The dashboard includes seeded demo data for a fast judge walkthrough.
- The red-alert path can be triggered in one click.
- The nudge path produces a concrete human action.
- The field notes explain what worked, what failed, and what remains honest future work.

