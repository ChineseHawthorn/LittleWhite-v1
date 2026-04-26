import json
import os

from openai import OpenAI

EXTRACTION_PROMPT = """分析以下对话，提取所有应该长期记住的用户重要信息。

必须提取的内容（只要提到就提取）：
- 基本信息：姓名、昵称、年龄、性别、职业、所在地
- 重要日期：生日、结婚纪念日、相识日等（格式：用户生日是X月X日）
- 偏好与习惯：喜好、兴趣、口味、习惯
- 正在进行的项目或目标
- 技术背景（编程语言、工具、技术栈）
- 重要决定或计划
- 用户明确要求"记住"的任何信息

对话内容：
{conversation}

只返回一个JSON数组，每条记忆是独立完整的句子，用用户说话的语言。
每条记忆必须脱离上下文也能理解。
示例：["用户叫小明", "用户生日是5月20日", "用户喜欢喝奶茶"]

没有重要信息则返回：[]"""


class MemoryManager:
    def __init__(self, rag_system):
        self.rag = rag_system
        self._default_client = OpenAI(
            base_url="https://www.gpt4novel.com/api/xiaoshuoai/ext/v1",
            api_key=os.getenv("API_KEY", ""),
        )
        self._default_model = "nalang-turbo-0826"

    def extract_and_save(
        self,
        conversation: list[dict],
        session_id: str = "default",
        client=None,
        fast_model: str | None = None,
    ) -> list[str]:
        if not conversation:
            return []

        use_client = client or self._default_client
        use_model = fast_model or self._default_model
        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)

        try:
            stream = use_client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(conversation=conv_text)}],
                max_tokens=1024,
                temperature=0.3,
                stream=True,
            )
            parts = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    parts.append(content)
            raw = "".join(parts).strip()
            if not raw:
                return []

            # Strip markdown fences if present
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            memories = json.loads(raw)
            if not isinstance(memories, list):
                return []

            saved = []
            for mem in memories:
                if isinstance(mem, str) and mem.strip():
                    self.rag.add_memory(mem.strip(), session_id)
                    saved.append(mem.strip())
            if saved:
                print(f"[MemoryManager] 提取到 {len(saved)} 条记忆: {saved}")
            else:
                print(f"[MemoryManager] 本轮无新记忆（模型返回: {raw[:80]}）")
            return saved

        except Exception as e:
            print(f"[MemoryManager] extraction error: {e}")
            return []
