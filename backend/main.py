import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

import capsule_store
from capsule_manager import CapsuleManager
from memory_manager import MemoryManager
from personality_config import PersonalityConfig, PersonalityManager
from rag_system import RAGSystem

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="小白")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

AVATAR_DIR = Path(__file__).parent / "images" / "avatar"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=Path(__file__).parent / "images"), name="images")

rag = RAGSystem()
memory_mgr = MemoryManager(rag)
capsule_mgr = CapsuleManager(rag)
personality_mgr = PersonalityManager()

# ── 多供应商配置 ──────────────────────────────────────────────────────
PROVIDERS: dict[str, dict] = {
    # 国内供应商
    "gpt4novel": {
        "name": "GPT4Novel",
        "base_url": "https://www.gpt4novel.com/api/xiaoshuoai/ext/v1",
        "api_key_env": "API_KEY",
        "default_model": "nalang-xl-0826-16k",
        "fast_model": "nalang-turbo-0826",
        "extra_body": {"repetition_penalty": 1.05},
        "top_p": 0.35,
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "fast_model": "deepseek-chat",
        "extra_body": None,
        "top_p": 0.95,
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
        "default_model": "qwen-plus",
        "fast_model": "qwen-turbo",
        "extra_body": None,
        "top_p": 0.8,
    },
    "zhipu": {
        "name": "智谱AI",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "default_model": "glm-4-air",
        "fast_model": "glm-4-flash",
        "extra_body": None,
        "top_p": 0.7,
    },
    "moonshot": {
        "name": "月之暗面",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-32k",
        "fast_model": "moonshot-v1-8k",
        "extra_body": None,
        "top_p": 1.0,
    },
    "minimax": {
        "name": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "default_model": "abab6.5s-chat",
        "fast_model": "abab6.5s-chat",
        "extra_body": None,
        "top_p": 0.9,
    },
    "doubao": {
        "name": "豆包",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "DOUBAO_API_KEY",
        "default_model": "doubao-pro-32k",
        "fast_model": "doubao-lite-32k",
        "extra_body": None,
        "top_p": 0.8,
    },
    "baichuan": {
        "name": "百川",
        "base_url": "https://api.baichuan-ai.com/v1",
        "api_key_env": "BAICHUAN_API_KEY",
        "default_model": "Baichuan4",
        "fast_model": "Baichuan3-Turbo",
        "extra_body": None,
        "top_p": 0.85,
    },
    "stepfun": {
        "name": "阶跃星辰",
        "base_url": "https://api.stepfun.com/v1",
        "api_key_env": "STEPFUN_API_KEY",
        "default_model": "step-2-16k",
        "fast_model": "step-1-8k",
        "extra_body": None,
        "top_p": 0.9,
    },
    "lingyi": {
        "name": "零一万物",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "api_key_env": "LINGYI_API_KEY",
        "default_model": "yi-lightning",
        "fast_model": "yi-lightning",
        "extra_body": None,
        "top_p": 0.9,
    },
    "hunyuan": {
        "name": "腾讯混元",
        "base_url": "https://api.hunyuan.tencent.com/v1",
        "api_key_env": "HUNYUAN_API_KEY",
        "default_model": "hunyuan-turbo",
        "fast_model": "hunyuan-lite",
        "extra_body": None,
        "top_p": 0.9,
    },
    "siliconflow": {
        "name": "硅基流动",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "fast_model": "Qwen/Qwen2.5-7B-Instruct",
        "extra_body": None,
        "top_p": 0.9,
    },
    # 国际供应商
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
        "extra_body": None,
        "top_p": 1.0,
    },
    "gemini": {
        "name": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
        "fast_model": "gemini-2.0-flash",
        "extra_body": None,
        "top_p": 0.95,
    },
    "mistral": {
        "name": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
        "fast_model": "mistral-small-latest",
        "extra_body": None,
        "top_p": 1.0,
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "fast_model": "llama-3.1-8b-instant",
        "extra_body": None,
        "top_p": 1.0,
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-3.5-sonnet",
        "fast_model": "anthropic/claude-3-haiku",
        "extra_body": None,
        "top_p": 1.0,
    },
    # 自定义供应商（运行时由请求提供 base_url 和 api_key）
    "custom": {
        "name": "自定义",
        "base_url": "",
        "api_key_env": "",
        "default_model": "",
        "fast_model": "",
        "extra_body": None,
        "top_p": 1.0,
    },
}

DEFAULT_PROVIDER = "gpt4novel"
DEFAULT_MODEL = "nalang-xl-0826-16k"


def get_client(
    provider_id: str,
    custom_base_url: str = "",
    custom_api_key: str = "",
) -> tuple[OpenAI, dict]:
    cfg = PROVIDERS.get(provider_id, PROVIDERS[DEFAULT_PROVIDER])
    if provider_id == "custom":
        base_url = custom_base_url or "https://api.openai.com/v1"
        api_key = custom_api_key or "placeholder"
    else:
        base_url = cfg["base_url"]
        api_key = os.getenv(cfg["api_key_env"], "") if cfg["api_key_env"] else ""
        if not api_key:
            api_key = "placeholder"
    return OpenAI(base_url=base_url, api_key=api_key), cfg

# {session_id: [{"role": ..., "content": ...}]}
sessions: dict[str, list[dict]] = {}

# System prompt 模板由 PersonalityManager 动态生成，不再硬编码

# ── 系统感知辅助工具 ──────────────────────────────────────────────────────

# 触发全量数据检索的关键词（用户在询问自己的存储数据时）
_DATA_QUERY_KW = frozenset([
    "记忆", "记得", "记住", "知道我", "了解我", "关于我", "我的信息",
    "我叫", "我的名字", "叫我", "你知道我", "记住了什么", "有什么记忆",
    "都记得", "还记得", "有没有记", "告诉过你", "说过", "跟你说过",
    "纪念日", "周年", "相识", "特殊日期", "重要日期",
    "胶囊", "时间胶囊", "封存", "开封",
    "聊过什么", "聊过", "历史", "以前说", "之前说",
    "忘了吗", "没忘", "你知道吗", "你了解",
    "昵称", "姓名", "名字叫",
])


def _detect_data_query(msg: str) -> bool:
    """检测用户是否在询问自己的存储数据，触发全量检索模式。"""
    return any(kw in msg for kw in _DATA_QUERY_KW)


def _extract_user_name(memories: list[str]) -> str:
    """从记忆列表中提取用户姓名/昵称，返回空字符串表示未找到。"""
    patterns = [
        r'用户(?:叫|的名字是|昵称是|名叫|叫做)\s*([^\s，,。.！!？?\n]+)',
        r'(?:叫我|你叫我|管我叫|称我为|称呼我为?)\s*([^\s，,。.！!？?\n]+)',
        r'(?:我叫|我的名字是|我的昵称是|我叫做)\s*([^\s，,。.！!？?\n]+)',
        r'(?:名字|昵称|称呼)(?:是|为)\s*([^\s，,。.！!？?\n]+)',
    ]
    for mem in memories:
        for pat in patterns:
            m = re.search(pat, mem)
            if m:
                name = m.group(1).strip("\"'""''【】[]《》（）()")
                if 1 <= len(name) <= 12:
                    return name
    return ""


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    time_location: str = ""
    custom_base_url: str = ""
    custom_api_key: str = ""
    # 调用格式：standard(含扩展参数) | openai(标准OpenAI) | minimal(精简参数)
    custom_format: str = "openai"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    sid = req.session_id
    user_msg = req.message
    provider_id = req.provider if req.provider in PROVIDERS else DEFAULT_PROVIDER
    ai_client, prov_cfg = get_client(provider_id, req.custom_base_url, req.custom_api_key)
    model = req.model or prov_cfg.get("default_model") or DEFAULT_MODEL
    time_location = req.time_location

    if sid not in sessions:
        sessions[sid] = []

    # ── 1. 意图检测 ───────────────────────────────────────────────────────
    is_data_query = _detect_data_query(user_msg)

    # ── 2. 记忆检索：≤40 条时全量注入；更多时按意图切换 ──────────────────
    _MEM_FULL_THRESHOLD = 40
    all_mem_items   = rag.get_all_memories()               # 始终拉全量（用于计数/昵称提取）
    all_mem_texts   = [m["content"] for m in all_mem_items]
    total_mems      = len(all_mem_texts)

    if total_mems == 0:
        inject_memories: list[str] = []
    elif total_mems <= _MEM_FULL_THRESHOLD or is_data_query:
        inject_memories = all_mem_texts                    # 全量注入
    else:
        inject_memories = rag.get_relevant_memories(user_msg, n_results=8)  # 语义相关

    # ── 3. 历史检索：数据查询时扩大窗口 ─────────────────────────────────
    n_hist     = 20 if is_data_query else 8
    rel_history = rag.get_relevant_history(user_msg, n_results=n_hist)

    # ── 4. 胶囊 & 纪念日：始终全量（数量有限、极重要）───────────────────
    all_capsules      = capsule_store.get_all_capsules()
    all_anniversaries = capsule_store.get_all_anniversaries()

    # ── 5. 提取用户昵称 ───────────────────────────────────────────────────
    user_name = _extract_user_name(all_mem_texts)

    print(
        f"[RAG] sid={sid} "
        f"记忆={total_mems}(注入={'全部' if len(inject_memories) == total_mems and total_mems > 0 else len(inject_memories)}) "
        f"历史={len(rel_history)} 胶囊={len(all_capsules)} 纪念日={len(all_anniversaries)} "
        f"数据查询={is_data_query} 昵称={user_name or '未知'} "
        f"最近消息={len(sessions[sid][-20:])}/{len(sessions[sid])}"
    )

    # ── 6. 组装各上下文分区 ───────────────────────────────────────────────
    today_mmdd = datetime.now().strftime("%m-%d")

    # 记忆区
    if inject_memories:
        is_full = len(inject_memories) == total_mems
        mem_header = (
            f"## 你对用户的全部记忆（共 {total_mems} 条）："
            if is_full else
            f"## 与本次话题相关的记忆（{total_mems} 条中的 {len(inject_memories)} 条）："
        )
        mem_section = mem_header + "\n" + "\n".join(f"• {m}" for m in inject_memories)
    else:
        mem_section = ""

    # 历史对话区
    if rel_history:
        hist_header = (
            f"## 以往对话记录（最近相关 {len(rel_history)} 条）："
            if not is_data_query else
            f"## 以往对话记录（扩展检索 {len(rel_history)} 条）："
        )
        hist_section = hist_header + "\n" + "\n".join(h["content"] for h in rel_history)
    else:
        hist_section = ""

    # 时间胶囊区（全量，含完整内容）
    cap_section = ""
    if all_capsules:
        cap_lines = []
        for c in all_capsules:
            status = "【已开封】" if c["is_opened"] else f"【未开封，开封日期：{c['open_date']}】"
            cap_lines.append(f"• [{c['topic']}] {status}\n  封存内容：{c['content']}")
        cap_section = (
            f"## 用户的时间胶囊（共 {len(all_capsules)} 个，以下是全部内容）：\n"
            + "\n".join(cap_lines)
        )

    # 纪念日区（全量，标注今日）
    ann_section = ""
    if all_anniversaries:
        ann_lines = []
        for a in all_anniversaries:
            date_str = a["date"]
            today_flag = " ⭐【今天就是这个日子！】" if date_str.endswith(today_mmdd) else ""
            ann_lines.append(
                f"• {a['name']} | 日期：{date_str}{today_flag} | 说明：{a['description']}"
            )
        ann_section = (
            f"## 用户的重要纪念日（共 {len(all_anniversaries)} 个，以下是全部）：\n"
            + "\n".join(ann_lines)
        )

    # 用户昵称区（提取到时始终注入）
    name_section = f"## 用户昵称：{user_name}（对话中请使用此称呼）" if user_name else ""

    # 时间地点天气
    tl_section = f"## 用户当前的时间、地点与天气：\n{time_location}" if time_location else ""

    # ── 7. 最终拼合 system prompt ─────────────────────────────────────────
    # 用 replace 替换两个占位符，避免 .format() 误解析模板里的 JSON 花括号
    system = (
        personality_mgr.get_prompt_template()
        .replace("{memory_section}",  mem_section)
        .replace("{history_section}", hist_section)
        .strip()
    )
    for extra in [cap_section, ann_section, name_section]:
        if extra:
            system += "\n\n" + extra
    if tl_section:
        system = tl_section + "\n\n" + system

    # System prompt goes as first message (OpenAI-compatible)
    api_messages = (
        [{"role": "system", "content": system}]
        + sessions[sid][-20:]
        + [{"role": "user", "content": user_msg}]
    )

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        full_response = ""

        try:
            # 根据供应商和格式类型构建调用参数
            extra_body = prov_cfg.get("extra_body")
            if provider_id == "custom" and req.custom_format == "standard":
                extra_body = {"repetition_penalty": 1.05}
            elif provider_id == "custom":
                extra_body = None

            call_kwargs: dict = {
                "model": model,
                "messages": api_messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            if req.custom_format != "minimal":
                call_kwargs["top_p"] = prov_cfg.get("top_p", 0.95)
            if extra_body:
                call_kwargs["extra_body"] = extra_body

            stream = ai_client.chat.completions.create(**call_kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield sse({"type": "token", "content": content})

        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
            return

        # 3. Strip leading ]:/]: artifacts (bleed from tag format in system prompt)
        full_response = re.sub(r'^\s*\]:\s*', '', full_response)

        # 4. Parse capsule/anniversary tags from AI response, strip them from clean text
        parsed = capsule_mgr.parse_and_save(full_response, sid)
        clean_response = parsed["clean_text"]

        # If tags were found, push a patch event so frontend strips the raw tags
        if parsed["capsules"] or parsed["anniversaries"]:
            yield sse({"type": "patch_text", "clean_text": clean_response})

        # 5. Persist to session + vector store (use clean text without tags)
        sessions[sid].append({"role": "user", "content": user_msg})
        sessions[sid].append({"role": "assistant", "content": clean_response})
        rag.add_chat_message("user", user_msg, sid)
        rag.add_chat_message("assistant", clean_response, sid)

        # 5. Extract memories from clean response
        conv_pair = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": clean_response},
        ]
        fast_model = prov_cfg.get("fast_model") or model
        new_memories = memory_mgr.extract_and_save(
            conv_pair, sid, client=ai_client, fast_model=fast_model
        )

        # 6. Final metadata event
        yield sse({
            "type": "done",
            "memories_used": inject_memories,
            "history_context": [h["content"] for h in rel_history],
            "new_memories": new_memories,
            "new_capsules": parsed["capsules"],
            "new_capsules_data": parsed["capsules_data"],
            "new_anniversaries": parsed["anniversaries"],
            "new_anniversaries_data": parsed["anniversaries_data"],
            "stats": {
                "total_memories": rag.memory_count(),
                "total_history": rag.history_count(),
                "total_capsules": capsule_store.capsule_count(),
                "total_anniversaries": capsule_store.anniversary_count(),
                "session_messages": len(sessions[sid]),
            },
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/memories")
async def get_memories():
    items = rag.get_all_memories()
    return {"memories": items, "count": len(items)}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    rag.delete_memory(memory_id)
    return {"status": "deleted"}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": list(sessions.keys())}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}


class RestoreRequest(BaseModel):
    messages: list[dict]


@app.post("/api/sessions/{session_id}/restore")
async def restore_session(session_id: str, body: RestoreRequest):
    msgs = []
    for m in body.messages or []:
        role = m.get("role", "")
        if role == "ai":
            role = "assistant"
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        msgs.append({"role": role, "content": content})
    sessions[session_id] = msgs[-20:]
    return {"restored": len(sessions[session_id])}


SUMMARIZE_PROMPT = """以下是一段对话记录。请将这段对话总结为一段简洁的记忆摘要，保留所有重要信息（用户信息、偏好、决定、讨论的重要话题等）。

对话记录：
{conversation}

请用中文返回一段简洁的摘要，不超过300字。只返回摘要内容，不要加任何标题或解释。"""


class CompactRequest(BaseModel):
    messages: list[dict]
    provider: str = DEFAULT_PROVIDER
    custom_base_url: str = ""
    custom_api_key: str = ""


@app.post("/api/compact/{session_id}")
async def compact_session(session_id: str, body: CompactRequest):
    if not body.messages:
        return {"memory": ""}
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in body.messages if m.get("content")
    )
    provider_id = body.provider if body.provider in PROVIDERS else DEFAULT_PROVIDER
    comp_client, prov_cfg = get_client(provider_id, body.custom_base_url, body.custom_api_key)
    fast_model = prov_cfg.get("fast_model") or prov_cfg.get("default_model") or DEFAULT_MODEL
    try:
        stream = comp_client.chat.completions.create(
            model=fast_model,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(conversation=conv_text)}],
            max_tokens=512,
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
        summary = "".join(parts).strip()
        if summary:
            mem_text = f"[对话摘要] {summary}"
            rag.add_memory(mem_text, session_id)
            if session_id in sessions and len(sessions[session_id]) >= 50:
                sessions[session_id] = sessions[session_id][50:]
            return {"memory": mem_text}
    except Exception as e:
        print(f"[compact] error: {e}")
    return {"memory": ""}


@app.get("/api/providers")
async def list_providers():
    """返回供应商列表，并标注哪些已在 .env 中配置了 API Key。"""
    result = []
    for pid, cfg in PROVIDERS.items():
        env_key = cfg.get("api_key_env", "")
        configured = bool(os.getenv(env_key, "")) if env_key else (pid == "custom")
        result.append({
            "id": pid,
            "name": cfg["name"],
            "configured": configured,
            "default_model": cfg.get("default_model", ""),
        })
    return {"providers": result}


ENV_PATH = Path(__file__).parent.parent / ".env"

class ProviderKeyRequest(BaseModel):
    api_key: str

@app.post("/api/providers/{provider_id}/key")
async def save_provider_key(provider_id: str, body: ProviderKeyRequest):
    """将指定供应商的 API Key 写入 .env 并热加载到当前进程。"""
    if provider_id not in PROVIDERS or provider_id == "custom":
        raise HTTPException(status_code=400, detail="provider not found or not configurable")
    cfg = PROVIDERS[provider_id]
    env_var = cfg.get("api_key_env", "")
    if not env_var:
        raise HTTPException(status_code=400, detail="provider has no api_key_env")
    api_key = (body.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key must not be empty")

    # 读取 / 更新 .env
    content = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    lines = content.splitlines()
    new_lines, found = [], False
    for line in lines:
        if line.lstrip().startswith(f"{env_var}="):
            new_lines.append(f"{env_var}={api_key}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{env_var}={api_key}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 热加载到当前进程（无需重启）
    os.environ[env_var] = api_key
    return {"status": "saved", "provider": provider_id, "env_var": env_var}


@app.get("/api/history")
async def get_history():
    items = rag.get_all_history()
    return {"history": items, "count": len(items)}


@app.delete("/api/history/{msg_id}")
async def delete_history_msg(msg_id: str):
    rag.delete_history(msg_id)
    return {"status": "deleted"}


@app.delete("/api/history")
async def clear_all_history():
    """一次性清除 ChromaDB 中的全部对话历史，并清空所有内存会话。"""
    rag.clear_all_history()
    sessions.clear()
    return {"status": "cleared"}


@app.get("/api/capsules")
async def get_capsules():
    items = capsule_store.get_all_capsules()
    return {"capsules": items, "count": len(items)}


@app.get("/api/anniversaries")
async def get_anniversaries():
    items = capsule_store.get_all_anniversaries()
    return {"anniversaries": items, "count": len(items)}


@app.delete("/api/capsules/{capsule_id}")
async def delete_capsule(capsule_id: str):
    capsule_store.delete_capsule(capsule_id)
    try: rag.delete_capsule(capsule_id)
    except Exception: pass
    return {"status": "deleted"}


@app.delete("/api/anniversaries/{ann_id}")
async def delete_anniversary(ann_id: str):
    capsule_store.delete_anniversary(ann_id)
    try: rag.delete_anniversary(ann_id)
    except Exception: pass
    return {"status": "deleted"}


_WEATHER_ICONS = {
    (113,): "☀️", (116,): "⛅", (119, 122): "☁️",
    (143, 248, 260): "🌫️",
    (176, 263, 266, 281, 284, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 353, 356, 359): "🌧️",
    (179, 182, 185, 323, 326, 329, 332, 335, 338, 350, 362, 365, 368, 371, 374, 377): "❄️",
    (200, 386, 389, 392, 395): "⛈️",
}

def _weather_icon(code: int) -> str:
    for codes, icon in _WEATHER_ICONS.items():
        if code in codes:
            return icon
    return "🌡️"


@app.get("/api/weather")
async def get_weather(city: str = ""):
    if not city:
        raise HTTPException(status_code=400, detail="city is required")
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(
                f"https://wttr.in/{city}?format=j1",
                headers={"Accept-Language": "zh-CN,zh;q=0.9"},
            )
            r.raise_for_status()
            d = r.json()
        cur = d["current_condition"][0]
        # wttr.in 返回 lang_zh 中文描述
        lang_zh = cur.get("lang_zh", [])
        desc = lang_zh[0]["value"] if lang_zh else cur["weatherDesc"][0]["value"]
        code = int(cur.get("weatherCode", 113))
        return {
            "icon":        _weather_icon(code),
            "desc":        desc,
            "temp_c":      int(cur["temp_C"]),
            "feels_like":  int(cur["FeelsLikeC"]),
            "humidity":    int(cur["humidity"]),
            "wind_kmph":   int(cur["windspeedKmph"]),
        }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="weather API timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    return {
        "memory_count": rag.memory_count(),
        "history_count": rag.history_count(),
        "capsule_count": capsule_store.capsule_count(),
        "anniversary_count": capsule_store.anniversary_count(),
    }


_ALLOWED_AVATAR_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@app.post("/api/avatar/{role}")
async def upload_avatar(role: str, file: UploadFile = File(...)):
    if role not in ("user", "ai"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'ai'")
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if ext not in _ALLOWED_AVATAR_EXTS:
        ext = ".jpg"
    filename = f"{role}_custom{ext}"
    dest = AVATAR_DIR / filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": f"/images/avatar/{filename}?v={int(time.time())}"}


@app.delete("/api/avatar/{role}")
async def delete_avatar(role: str):
    if role not in ("user", "ai"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'ai'")
    for ext in _ALLOWED_AVATAR_EXTS:
        f = AVATAR_DIR / f"{role}_custom{ext}"
        if f.exists():
            f.unlink()
    return {"status": "deleted"}


# ── 人格控制 API ──────────────────────────────────────────────────────────

@app.get("/api/personality")
async def get_personality():
    """获取当前人格配置（JSON 格式）。"""
    return personality_mgr.get().model_dump()


@app.put("/api/personality")
async def update_personality(cfg: PersonalityConfig):
    """完整替换人格配置，立即生效并持久化到 data/personality.json。"""
    personality_mgr.update(cfg)
    return {"status": "saved"}


@app.post("/api/personality/reset")
async def reset_personality():
    """重置为出厂默认人格。"""
    personality_mgr.reset()
    return {"status": "reset", "config": personality_mgr.get().model_dump()}


@app.get("/api/personality/preview")
async def preview_personality():
    """返回当前配置生成的 system prompt 预览（占位符保持原样）。"""
    return {"prompt": personality_mgr.get_prompt_template()}


@app.get("/manage")
async def serve_manage():
    """记忆管理道具（与主页同源，可共享 localStorage）。"""
    path = Path(__file__).resolve().parent.parent / "frontend" / "memory_manager.html"
    if path.exists():
        return HTMLResponse(
            content=path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return HTMLResponse(f"<h1>未找到: {path}</h1>", status_code=404)


@app.get("/")
async def serve_index():
    path = Path(__file__).parent.parent / "frontend" / "index.html"
    if path.exists():
        return HTMLResponse(
            content=path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
