from __future__ import annotations

import argparse
import base64
import json
import sys
from urllib.parse import urljoin

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-cost smoke checks for the Adwuma Pa Modal API.")
    parser.add_argument("--base-url", required=True, help="Modal ASGI app base URL.")
    parser.add_argument("--health", action="store_true", help="Check /health.")
    parser.add_argument("--translate", help="Translate one Twi/Fante/Akan text sample.")
    parser.add_argument("--source-language", default="twi")
    parser.add_argument("--transcribe-audio", help="Path to one short WAV/MP3/M4A/FLAC sample for ASR.")
    parser.add_argument("--language", default="twi", help="Language hint for ASR/TTS.")
    parser.add_argument("--analyze-sample", action="store_true", help="Run one fixed translated concern-analysis sample.")
    parser.add_argument("--speak", help="Synthesize one short prompt.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"
    ran = False

    if args.health:
        ran = True
        response = requests.get(urljoin(base_url, "health"), timeout=30)
        print_response("health", response)
        response.raise_for_status()

    if args.translate is not None:
        ran = True
        response = requests.post(
            urljoin(base_url, "translate"),
            json={"text": args.translate, "source_language": args.source_language},
            timeout=150,
        )
        print_response("translate", response)
        response.raise_for_status()

    if args.transcribe_audio is not None:
        ran = True
        with open(args.transcribe_audio, "rb") as audio_file:
            encoded = base64.b64encode(audio_file.read()).decode("ascii")
        response = requests.post(
            urljoin(base_url, "transcribe"),
            json={"audio_wav_base64": encoded, "language": args.language},
            timeout=180,
        )
        print_response("transcribe", response)
        response.raise_for_status()

    if args.analyze_sample:
        ran = True
        response = requests.post(
            urljoin(base_url, "analyze"),
            json={
                "original_text": "Me ho ye, na me nsa aka aduan. Meda wo ase.",
                "english_translation": "I am well, I have had food. Thank you.",
                "language": "twi",
                "member": {
                    "id": "smoke-member",
                    "name": "Smoke Test Elder",
                    "city": "Accra",
                    "region": "Greater Accra",
                },
                "request": {
                    "id": "smoke-request",
                    "reason_code": "routine_due",
                    "reason_detail": "Routine welfare check.",
                    "type": "self_checkin",
                    "created_at": "2026-06-09T00:00:00Z",
                },
                "recent_history": [],
                "required_schema": [
                    "summary",
                    "concern_level",
                    "flags",
                    "sentiment",
                    "evidence",
                    "recommended_action",
                    "confidence",
                ],
            },
            timeout=240,
        )
        print_response("analyze", response)
        response.raise_for_status()

    if args.speak is not None:
        ran = True
        response = requests.post(
            urljoin(base_url, "speak"),
            json={"text": args.speak, "language": args.language},
            timeout=180,
        )
        print_response("speak", response)
        response.raise_for_status()

    if not ran:
        parser.error("Choose at least one check: --health, --translate TEXT, --transcribe-audio PATH, --analyze-sample, or --speak TEXT")
    return 0


def print_response(label: str, response: requests.Response) -> None:
    print(f"## {label}: HTTP {response.status_code}")
    try:
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("audio_wav_base64"), str):
            data = dict(data)
            audio = data["audio_wav_base64"]
            data["audio_wav_base64"] = f"<{len(audio)} base64 chars>"
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except ValueError:
        print(response.text[:1000])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
