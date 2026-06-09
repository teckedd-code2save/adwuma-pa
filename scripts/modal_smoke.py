from __future__ import annotations

import argparse
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

    if not ran:
        parser.error("Choose at least one check: --health or --translate TEXT")
    return 0


def print_response(label: str, response: requests.Response) -> None:
    print(f"## {label}: HTTP {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(response.text[:1000])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
