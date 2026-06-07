from __future__ import annotations

import json
from typing import Any

import numpy as np

from config.models import LLM_CONFIG, TRANSLATION_CONFIG
from db import database as db
from services import modal_client


REQUIRED_ANALYSIS_KEYS = {
    "summary",
    "concern_level",
    "flags",
    "sentiment",
    "evidence",
    "recommended_action",
    "confidence",
}


def submit_request_response(
    token: str,
    text: str,
    language: str,
    input_type: str = "text",
    audio: tuple[int, np.ndarray] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    request = db.get_request_by_token(token)
    if not request:
        return {"ok": False, "status": "failed", "message": "No checkup request exists for this token."}
    if request["status"] not in {"pending", "sent", "processing", "needs_review"}:
        return {"ok": False, "status": "failed", "message": f"This request is already {request['status']}."}

    source = source or ("field_report" if request["request_type"] == "field_report" else "self")
    transcript = (text or "").strip()
    raw_input = transcript
    asr_model = None
    asr_confidence = None
    model_errors: list[str] = []

    if input_type == "voice":
        asr_result = modal_client.transcribe_audio(audio, language)
        if asr_result.ok:
            transcript = str(asr_result.data.get("text", "")).strip()
            asr_model = asr_result.data.get("model_used")
            asr_confidence = asr_result.data.get("confidence")
            db.record_model_run(
                None,
                "asr",
                asr_model,
                "complete",
                input_summary=f"request={request['id']} language={language}",
                output_json=asr_result.data,
                latency_ms=asr_result.latency_ms,
            )
        else:
            model_errors.append(asr_result.error or "ASR failed.")
            db.record_model_run(None, "asr", None, "needs_review", error=asr_result.error, latency_ms=asr_result.latency_ms)

    if not transcript:
        return _save_needs_review(
            request,
            source,
            language,
            input_type,
            raw_input,
            transcript,
            "No usable text or transcript was received.",
            asr_model,
            asr_confidence,
        )

    translation_result = modal_client.translate_text(transcript, language)
    translation = ""
    if translation_result.ok:
        translation = str(translation_result.data.get("translated_text") or translation_result.data.get("translation") or "").strip()
        db.record_model_run(
            None,
            "translation",
            translation_result.data.get("model_id", TRANSLATION_CONFIG["model_id"]),
            "complete",
            input_summary=f"request={request['id']} source={language}",
            output_json=translation_result.data,
            latency_ms=translation_result.latency_ms,
        )
    else:
        model_errors.append(translation_result.error or "Translation failed.")
        db.record_model_run(
            None,
            "translation",
            TRANSLATION_CONFIG["model_id"],
            "needs_review",
            error=translation_result.error,
            latency_ms=translation_result.latency_ms,
        )

    if not translation:
        return _save_needs_review(
            request,
            source,
            language,
            input_type,
            raw_input,
            transcript,
            "; ".join(model_errors) or "Translation did not produce usable English text.",
            asr_model,
            asr_confidence,
        )

    analysis_payload = {
        "original_text": transcript,
        "english_translation": translation,
        "language": language,
        "member": {
            "id": request["member_id"],
            "name": request["member_name"],
            "city": request.get("location_city"),
            "region": request.get("location_region"),
        },
        "request": {
            "id": request["id"],
            "reason_code": request["reason_code"],
            "reason_detail": request.get("reason_detail"),
            "type": request["request_type"],
            "created_at": request["created_at"],
        },
        "recent_history": recent_history(request["member_id"]),
        "required_schema": sorted(REQUIRED_ANALYSIS_KEYS),
    }
    analysis_result = modal_client.analyze_concern(analysis_payload)
    if not analysis_result.ok:
        return _save_needs_review(
            request,
            source,
            language,
            input_type,
            raw_input,
            transcript,
            analysis_result.error or "Qwen concern analysis did not run.",
            asr_model,
            asr_confidence,
            translation,
        )

    analysis = analysis_result.data.get("analysis", analysis_result.data)
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            analysis = {}
    validation_error = validate_analysis(analysis)
    if validation_error:
        return _save_needs_review(
            request,
            source,
            language,
            input_type,
            raw_input,
            transcript,
            validation_error,
            asr_model,
            asr_confidence,
            translation,
        )

    concern_level = int(analysis["concern_level"])
    checkin_id = db.add_checkin(
        request["member_id"],
        source,
        transcript,
        analysis["summary"],
        concern_level,
        list(analysis.get("flags") or []),
        language,
        input_type=input_type,
        raw_input=raw_input,
        asr_model_used=asr_model,
        asr_confidence=float(asr_confidence) if asr_confidence is not None else None,
        request_id=request["id"],
        translation=translation,
        analysis_status="complete",
        analysis_json=analysis,
    )
    db.record_model_run(
        checkin_id,
        "analysis",
        analysis_result.data.get("model_id", LLM_CONFIG["model_id"]),
        "complete",
        input_summary=f"request={request['id']} member={request['member_name']}",
        output_json=analysis,
        latency_ms=analysis_result.latency_ms,
    )
    return {
        "ok": True,
        "status": "complete",
        "checkin_id": checkin_id,
        "summary": analysis["summary"],
        "concern_level": concern_level,
        "translation": translation,
        "analysis": analysis,
    }


def validate_analysis(analysis: dict[str, Any]) -> str | None:
    missing = REQUIRED_ANALYSIS_KEYS.difference(analysis)
    if missing:
        return f"Qwen returned invalid JSON; missing keys: {', '.join(sorted(missing))}."
    try:
        concern = int(analysis["concern_level"])
    except Exception:
        return "Qwen returned invalid concern_level; expected an integer from 0 to 10."
    if concern < 0 or concern > 10:
        return "Qwen returned invalid concern_level; expected 0 to 10."
    if not isinstance(analysis.get("flags"), list) or not isinstance(analysis.get("evidence"), list):
        return "Qwen returned invalid JSON; flags and evidence must be arrays."
    return None


def recent_history(member_id: str) -> list[dict[str, Any]]:
    return db.rows(
        """
        SELECT submitted_at, source, summary, concern_level, analysis_status
        FROM checkins
        WHERE member_id = ?
        ORDER BY submitted_at DESC
        LIMIT 5
        """,
        (member_id,),
    )


def _save_needs_review(
    request: dict[str, Any],
    source: str,
    language: str,
    input_type: str,
    raw_input: str,
    transcript: str,
    error: str,
    asr_model: str | None = None,
    asr_confidence: float | None = None,
    translation: str | None = None,
) -> dict[str, Any]:
    checkin_id = db.add_checkin(
        request["member_id"],
        source,
        transcript,
        "Needs human review. No automated concern score was produced.",
        None,
        [],
        language,
        input_type=input_type,
        raw_input=raw_input,
        asr_model_used=asr_model,
        asr_confidence=float(asr_confidence) if asr_confidence is not None else None,
        request_id=request["id"],
        translation=translation,
        analysis_status="needs_review",
        analysis_json={},
        processing_error=error,
    )
    return {
        "ok": False,
        "status": "needs_review",
        "checkin_id": checkin_id,
        "summary": "Needs human review. No automated concern score was produced.",
        "concern_level": None,
        "translation": translation or "",
        "error": error,
    }
