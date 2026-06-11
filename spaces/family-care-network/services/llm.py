from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ConcernResult:
    summary: str
    concern_level: int
    flags: list[str]
    language_detected: str
    sentiment: str
    method: str
    matched_signals: dict[str, list[str]]
    scoring_steps: list[str]
    next_action: str


KEYWORDS = {
    "pain": ["pain", "sick", "fever", "hospital", "chest", "breath", "dizzy", "medicine", "yare", "ayaresa"],
    "food": ["food", "eat", "eating", "hunger", "sika", "money", "aduane"],
    "isolation": ["alone", "lonely", "nobody", "visit", "silence", "house", "nkoaa"],
    "mobility": ["walk", "walking", "fall", "fell", "leg", "nantew", "nan"],
    "good": ["fine", "well", "okay", "visited", "came by", "me ho ye"],
}


def score_concern(text: str, language_hint: str = "twi") -> ConcernResult:
    """Deterministic fallback scorer; swap with a 7B instruct model when GPU is ready."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    flags: list[str] = []
    matched_signals: dict[str, list[str]] = {}
    scoring_steps: list[str] = ["Start at baseline concern score 1."]
    score = 1

    for flag, words in KEYWORDS.items():
        if flag == "good":
            continue
        matches = [word for word in words if word in normalized]
        if matches:
            flags.append(flag)
            matched_signals[flag] = matches
            score += 2
            scoring_steps.append(f"Matched {flag} signal ({', '.join(matches)}): +2.")

    urgent_matches = [word for word in ["emergency", "cannot breathe", "chest pain", "collapsed", "stroke"] if word in normalized]
    if urgent_matches:
        score = max(score, 9)
        flags.append("urgent_medical")
        matched_signals["urgent_medical"] = urgent_matches
        scoring_steps.append(f"Matched urgent phrase ({', '.join(urgent_matches)}): raise score to at least 9.")
    good_matches = [word for word in KEYWORDS["good"] if word in normalized]
    if good_matches and score <= 3:
        score = 1
        matched_signals["stable_language"] = good_matches
        scoring_steps.append(f"Matched stable language ({', '.join(good_matches)}) with no major flags: keep score at 1.")
    if not normalized:
        score = 6
        flags.append("empty_response")
        matched_signals["empty_response"] = []
        scoring_steps.append("No usable text: set score to 6 for follow-up.")

    score = max(0, min(score, 10))
    sentiment = "neutral-negative" if score >= 4 else "stable"
    summary = summarize(normalized, score, flags)
    next_action = next_action_for_score(score)
    scoring_steps.append(f"Final score {score}: {next_action}")
    return ConcernResult(
        summary=summary,
        concern_level=score,
        flags=sorted(set(flags)),
        language_detected=language_hint,
        sentiment=sentiment,
        method="deterministic_keyword_fallback_pending_small_llm",
        matched_signals=matched_signals,
        scoring_steps=scoring_steps,
        next_action=next_action,
    )


def summarize(text: str, score: int, flags: list[str]) -> str:
    if not text:
        return "No usable response was received; follow-up is needed."
    if score >= 7:
        return f"High concern response mentioning {', '.join(flags)}. Coordinator should trigger same-day follow-up."
    if score >= 4:
        return f"Moderate concern response mentioning {', '.join(flags)}. A nearby relative should check in."
    return "Low concern response. Log the check-in and keep the normal schedule."


def next_action_for_score(score: int) -> str:
    if score >= 7:
        return "Urgent follow-up: alert the coordinator, nudge the assigned contact, and prepare a call."
    if score >= 4:
        return "Needs attention: notify the coordinator and ask a nearby relative to check in."
    return "Routine: log the check-in and keep the normal schedule."


def as_json(result: ConcernResult) -> str:
    return json.dumps(result.__dict__, indent=2)
