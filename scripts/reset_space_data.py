#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import requests
from gradio_client import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear Ani Kese data from a running Hugging Face Space.")
    parser.add_argument(
        "--space-url",
        default="https://build-small-hackathon-family-care-network.hf.space",
        help="Base URL of the running Space.",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm destructive reset without prompting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    space_url = args.space_url.rstrip("/")
    if not args.yes:
        print(f"This will delete all Ani Kese family records from {space_url}.")
        answer = input("Type RESET to continue: ").strip()
        if answer != "RESET":
            print("Aborted.")
            return 1

    client = Client(space_url)
    result = client.predict(api_name="/clear_data")
    message = result[0] if isinstance(result, (list, tuple)) and result else result
    print(message)

    storage = requests.get(f"{space_url}/debug/storage", timeout=30).json()
    print(json.dumps(storage, indent=2))
    failed = {
        key: storage.get(key)
        for key in ("member_count", "request_count", "outbound_count")
        if storage.get(key) != 0
    }
    if failed:
        print(f"Reset did not fully clear tracked counts: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
