from __future__ import annotations

from db import database as db
from services.autopilot import run_autopilot_scan


def scan_for_silence() -> dict:
    db.init_db()
    return run_autopilot_scan(force=False, actor="local scheduler")


if __name__ == "__main__":
    result = scan_for_silence()
    for action in result.get("actions", []):
        print(action)
    for delivery in result.get("deliveries", []):
        print(delivery)
    if result.get("reason"):
        print(result["reason"])
