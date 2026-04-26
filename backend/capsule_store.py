"""
JSON-backed persistent store for time capsules and anniversaries.
Completely independent of ChromaDB — guarantees data survives server restarts.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CAPSULES_FILE    = DATA_DIR / "capsules.json"
ANNIVERSARIES_FILE = DATA_DIR / "anniversaries.json"


def _read(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write(path: Path, data: list):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Time Capsules ──────────────────────────────────────────────

def add_capsule(topic: str, content: str, open_date: str, session_id: str = "default", capsule_id: str = None) -> dict:
    capsules = _read(CAPSULES_FILE)
    today = datetime.now().date().isoformat()
    item = {
        "id": capsule_id or f"cap_{uuid.uuid4().hex}",
        "topic": topic,
        "content": content,
        "open_date": open_date,
        "created_at": datetime.now().isoformat(),
        "is_opened": open_date <= today if open_date else False,
        "session_id": session_id,
    }
    capsules.append(item)
    _write(CAPSULES_FILE, capsules)
    return {k: v for k, v in item.items() if k != "session_id"}


def get_all_capsules() -> list:
    today = datetime.now().date().isoformat()
    capsules = _read(CAPSULES_FILE)
    result = []
    for c in capsules:
        od = c.get("open_date", "")
        result.append({k: v for k, v in c.items() if k != "session_id"} | {"is_opened": (od <= today) if od else False})
    result.sort(key=lambda x: x.get("open_date", ""))
    return result


def capsule_count() -> int:
    return len(_read(CAPSULES_FILE))


# ── Anniversaries ──────────────────────────────────────────────

def add_anniversary(name: str, date: str, description: str, session_id: str = "default", ann_id: str = None) -> dict:
    anniversaries = _read(ANNIVERSARIES_FILE)
    item = {
        "id": ann_id or f"ann_{uuid.uuid4().hex}",
        "name": name,
        "date": date,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "session_id": session_id,
    }
    anniversaries.append(item)
    _write(ANNIVERSARIES_FILE, anniversaries)
    return {k: v for k, v in item.items() if k != "session_id"}


def get_all_anniversaries() -> list:
    return [{k: v for k, v in a.items() if k != "session_id"} for a in _read(ANNIVERSARIES_FILE)]


def anniversary_count() -> int:
    return len(_read(ANNIVERSARIES_FILE))


def delete_capsule(capsule_id: str):
    items = [c for c in _read(CAPSULES_FILE) if c["id"] != capsule_id]
    _write(CAPSULES_FILE, items)


def delete_anniversary(ann_id: str):
    items = [a for a in _read(ANNIVERSARIES_FILE) if a["id"] != ann_id]
    _write(ANNIVERSARIES_FILE, items)
