import json
import re
import uuid
from datetime import datetime

import capsule_store


# Tags the main AI embeds in its reply — backend strips and persists them.
# [CAPSULE]{"topic":"...","open_date":"YYYY-MM-DD","content":"..."}[/CAPSULE]
# [ANNIVERSARY]{"name":"...","date":"MM-DD","description":"..."}[/ANNIVERSARY]
_CAP_RE = re.compile(r'\[CAPSULE\](.*?)\[/CAPSULE\]', re.DOTALL)
_ANN_RE = re.compile(r'\[ANNIVERSARY\](.*?)\[/ANNIVERSARY\]', re.DOTALL)


class CapsuleManager:
    def __init__(self, rag_system):
        self.rag = rag_system  # kept for RAG semantic search

    def parse_and_save(self, ai_response: str, session_id: str = "default") -> dict:
        """
        Scan the AI's raw reply for embedded capsule/anniversary tags,
        persist them to RAG, and return the stripped response text plus
        lists of what was saved.
        """
        result = {
            "capsules": [],        # topic strings (for legacy compat)
            "capsules_data": [],   # full objects for immediate frontend render
            "anniversaries": [],   # name strings
            "anniversaries_data": [],
            "clean_text": ai_response,
        }

        today = datetime.now().date().isoformat()
        clean = ai_response

        # ── Time capsules ──
        for m in _CAP_RE.finditer(ai_response):
            raw_tag = m.group(0)
            try:
                data = json.loads(m.group(1).strip())
                topic     = data.get("topic", "").strip()
                open_date = data.get("open_date", "").strip()
                content   = data.get("content", "").strip()
                if topic and open_date:
                    cap_id = f"cap_{uuid.uuid4().hex}"
                    saved = capsule_store.add_capsule(topic, content, open_date, session_id, capsule_id=cap_id)
                    try:
                        self.rag.add_capsule(topic, content, open_date, session_id, capsule_id=cap_id)
                    except Exception:
                        pass
                    result["capsules"].append(topic)
                    result["capsules_data"].append({
                        "topic": topic,
                        "open_date": open_date,
                        "is_opened": saved["is_opened"],
                    })
                    print(f"[CapsuleManager] 胶囊已写入JSON: {topic!r} 开封:{open_date}")
                else:
                    print(f"[CapsuleManager] 胶囊字段不完整 topic={topic!r} open_date={open_date!r}")
            except Exception as e:
                print(f"[CapsuleManager] capsule parse error: {e} | raw: {m.group(1)[:300]}")
            clean = clean.replace(raw_tag, "")

        # ── Anniversaries ──
        for m in _ANN_RE.finditer(ai_response):
            raw_tag = m.group(0)
            try:
                data        = json.loads(m.group(1).strip())
                name        = data.get("name", "").strip()
                date        = data.get("date", "").strip()
                description = data.get("description", "").strip()
                if name and date:
                    ann_id = f"ann_{uuid.uuid4().hex}"
                    capsule_store.add_anniversary(name, date, description, session_id, ann_id=ann_id)
                    try:
                        self.rag.add_anniversary(name, date, description, session_id, ann_id=ann_id)
                    except Exception:
                        pass
                    result["anniversaries"].append(name)
                    result["anniversaries_data"].append({"name": name, "date": date})
                    print(f"[CapsuleManager] 纪念日已写入JSON: {name!r} 日期:{date}")
                else:
                    print(f"[CapsuleManager] 纪念日字段不完整 name={name!r} date={date!r}")
            except Exception as e:
                print(f"[CapsuleManager] anniversary parse error: {e} | raw: {m.group(1)[:300]}")
            clean = clean.replace(raw_tag, "")

        result["clean_text"] = clean.strip()
        return result
