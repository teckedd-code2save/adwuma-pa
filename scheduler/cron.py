from __future__ import annotations

from db import database as db
from services.relay import scan_silence


def scan_for_silence() -> list[str]:
    db.init_db()
    return scan_silence()


if __name__ == "__main__":
    for action in scan_for_silence():
        print(action)
