import chromadb
from pathlib import Path
from datetime import datetime
import uuid

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class RAGSystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
        self.memories = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )
        self.chat_history = self.client.get_or_create_collection(
            name="chat_history",
            metadata={"hnsw:space": "cosine"},
        )
        self.time_capsules = self.client.get_or_create_collection(
            name="time_capsules",
            metadata={"hnsw:space": "cosine"},
        )
        self.anniversaries = self.client.get_or_create_collection(
            name="anniversaries",
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(self, text: str, session_id: str = "default") -> str:
        mem_id = f"mem_{uuid.uuid4().hex}"
        self.memories.add(
            documents=[text],
            metadatas=[{"session_id": session_id, "timestamp": datetime.now().isoformat()}],
            ids=[mem_id],
        )
        return mem_id

    def add_chat_message(self, role: str, content: str, session_id: str = "default") -> str:
        msg_id = f"msg_{uuid.uuid4().hex}"
        self.chat_history.add(
            documents=[f"[{role.upper()}]: {content}"],
            metadatas=[{
                "session_id": session_id,
                "role": role,
                "timestamp": datetime.now().isoformat(),
            }],
            ids=[msg_id],
        )
        return msg_id

    def get_relevant_memories(self, query: str, n_results: int = 5) -> list[str]:
        count = self.memories.count()
        if count == 0:
            return []
        results = self.memories.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        return results["documents"][0] if results["documents"] else []

    def get_relevant_history(self, query: str, n_results: int = 8) -> list[dict]:
        count = self.chat_history.count()
        if count == 0:
            return []
        results = self.chat_history.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        items = []
        for doc, meta, dist in zip(docs, metas, distances):
            if dist < 0.85:
                items.append({
                    "content": doc,
                    "timestamp": meta.get("timestamp", ""),
                    "role": meta.get("role", ""),
                })
        items.sort(key=lambda x: x["timestamp"])
        return items

    def get_all_memories(self) -> list[dict]:
        count = self.memories.count()
        if count == 0:
            return []
        results = self.memories.get(include=["documents", "metadatas"])
        items = []
        for doc, meta, mem_id in zip(
            results["documents"], results["metadatas"], results["ids"]
        ):
            items.append({
                "id": mem_id,
                "content": doc,
                "timestamp": meta.get("timestamp", ""),
            })
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return items

    def delete_memory(self, memory_id: str):
        self.memories.delete(ids=[memory_id])

    def get_all_history(self) -> list[dict]:
        count = self.chat_history.count()
        if count == 0:
            return []
        results = self.chat_history.get(include=["documents", "metadatas"])
        items = []
        for doc, meta, msg_id in zip(
            results["documents"], results["metadatas"], results["ids"]
        ):
            items.append({
                "id": msg_id,
                "content": doc,
                "role": meta.get("role", ""),
                "timestamp": meta.get("timestamp", ""),
            })
        items.sort(key=lambda x: x["timestamp"])
        return items

    def delete_history(self, msg_id: str):
        self.chat_history.delete(ids=[msg_id])

    def clear_all_history(self):
        """删除并重建 chat_history collection，同时清空整个 embeddings_queue。

        ChromaDB 在 delete_collection 时不清理 embeddings_queue（WAL），
        导致旧记录残留。直接 DELETE FROM embeddings_queue 是彻底清除的唯一方式。
        memories / time_capsules / anniversaries 的队列条目在 HNSW 索引为空时
        属于无效悬空数据，一并清除。
        """
        import sqlite3

        self.client.delete_collection("chat_history")
        self.chat_history = self.client.create_collection(
            name="chat_history",
            metadata={"hnsw:space": "cosine"},
        )

        db_path = DATA_DIR / "chroma_db" / "chroma.sqlite3"
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("DELETE FROM embeddings_queue")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[RAGSystem] embeddings_queue 清理失败: {e}")

    def memory_count(self) -> int:
        return self.memories.count()

    def history_count(self) -> int:
        return self.chat_history.count()

    # ── Time Capsules ──

    def add_capsule(self, topic: str, content: str, open_date: str, session_id: str = "default", capsule_id: str = None) -> str:
        cap_id = capsule_id or f"cap_{uuid.uuid4().hex}"
        self.time_capsules.add(
            documents=[content],
            metadatas=[{
                "topic": topic,
                "open_date": open_date,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
            }],
            ids=[cap_id],
        )
        return cap_id

    def get_all_capsules(self) -> list[dict]:
        count = self.time_capsules.count()
        if count == 0:
            return []
        results = self.time_capsules.get(include=["documents", "metadatas"])
        today = datetime.now().date().isoformat()
        items = []
        for doc, meta, cap_id in zip(results["documents"], results["metadatas"], results["ids"]):
            open_date = meta.get("open_date", "")
            items.append({
                "id": cap_id,
                "topic": meta.get("topic", ""),
                "open_date": open_date,
                "created_at": meta.get("created_at", ""),
                "is_opened": open_date <= today if open_date else False,
            })
        items.sort(key=lambda x: x["open_date"])
        return items

    def capsule_count(self) -> int:
        return self.time_capsules.count()

    def delete_capsule(self, capsule_id: str):
        self.time_capsules.delete(ids=[capsule_id])

    # ── Anniversaries ──

    def add_anniversary(self, name: str, date: str, description: str, session_id: str = "default", ann_id: str = None) -> str:
        ann_id = ann_id or f"ann_{uuid.uuid4().hex}"
        self.anniversaries.add(
            documents=[description],
            metadatas=[{
                "name": name,
                "date": date,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
            }],
            ids=[ann_id],
        )
        return ann_id

    def get_all_anniversaries(self) -> list[dict]:
        count = self.anniversaries.count()
        if count == 0:
            return []
        results = self.anniversaries.get(include=["documents", "metadatas"])
        items = []
        for doc, meta, ann_id in zip(results["documents"], results["metadatas"], results["ids"]):
            items.append({
                "id": ann_id,
                "name": meta.get("name", ""),
                "date": meta.get("date", ""),
                "description": doc,
                "created_at": meta.get("created_at", ""),
            })
        return items

    def anniversary_count(self) -> int:
        return self.anniversaries.count()

    def delete_anniversary(self, ann_id: str):
        self.anniversaries.delete(ids=[ann_id])

    def get_relevant_capsules(self, query: str, n_results: int = 3) -> list[dict]:
        count = self.time_capsules.count()
        if count == 0:
            return []
        results = self.time_capsules.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        today = datetime.now().date().isoformat()
        items = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if dist < 0.7:
                open_date = meta.get("open_date", "")
                items.append({
                    "topic": meta.get("topic", ""),
                    "content": doc,
                    "open_date": open_date,
                    "is_opened": open_date <= today if open_date else False,
                })
        return items

    def get_relevant_anniversaries(self, query: str, n_results: int = 3) -> list[dict]:
        count = self.anniversaries.count()
        if count == 0:
            return []
        results = self.anniversaries.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if dist < 0.7:
                items.append({
                    "name": meta.get("name", ""),
                    "date": meta.get("date", ""),
                    "description": doc,
                })
        return items
