# 小白

一个支持多 AI 供应商的私人聊天助手，带有 RAG 长期记忆、时间胶囊、纪念日提醒和可定制人格系统。

## 功能特性

- **多供应商切换** — 内置 17+ 国内外 AI 供应商，UI 内即可管理 API Key，随时一键切换模型
- **RAG 长期记忆** — 基于 ChromaDB 向量数据库，自动从对话中提取并检索记忆，跨会话持续生效
- **时间胶囊** — 设定未来某个时间自动开启的内容，到期前倒计时展示
- **纪念日提醒** — 重要日期管理，自动计算距今天数
- **人格系统** — 可视化配置 AI 的身份、性格、语言风格、输出格式、互动方式、记忆优先级、防幻觉等维度
- **流式输出** — 实时 token 流，响应即时显示
- **自定义供应商** — 支持任意 OpenAI 兼容 API 端点
- **自定义头像** — 用户和 AI 头像均可上传替换

## 支持的 AI 供应商

| 分类 | 供应商 |
|------|--------|
| 国内 | GPT4Novel、DeepSeek、通义千问、智谱AI、月之暗面、MiniMax、豆包、文心一言、百川、阶跃星辰、零一万物、腾讯混元、硅基流动 |
| 国际 | OpenAI、Claude (Anthropic)、Gemini、Mistral、Groq、OpenRouter |
| 自定义 | 任意 OpenAI 兼容端点 |

## 快速开始

### 环境要求

- Python 3.10+

### 安装与启动

1. 克隆项目并进入目录

2. 复制环境变量模板并填写至少一个 API Key：

```
# .env 示例（填写你要使用的供应商对应的 Key）
API_KEY=          # GPT4Novel
DEEPSEEK_API_KEY=
QWEN_API_KEY=
ZHIPU_API_KEY=
MOONSHOT_API_KEY=
MINIMAX_API_KEY=
DOUBAO_API_KEY=
BAIDU_API_KEY=
BAICHUAN_API_KEY=
STEPFUN_API_KEY=
LINGYI_API_KEY=
HUNYUAN_API_KEY=
SILICONFLOW_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
MISTRAL_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
```

3. 双击运行启动脚本：

```
start.bat
```

脚本会自动安装依赖并启动服务，随后在浏览器打开：

```
http://localhost:8000
```

## 目录结构

```
小白/
├── backend/
│   ├── main.py               # FastAPI 服务，API 路由，流式推理
│   ├── rag_system.py         # ChromaDB 向量数据库
│   ├── memory_manager.py     # 记忆提取与检索
│   ├── capsule_manager.py    # 时间胶囊管理
│   ├── capsule_store.py      # 胶囊 JSON 持久化
│   ├── personality_config.py # 人格配置读写
│   └── requirements.txt
├── frontend/
│   ├── index.html            # 主聊天界面（单文件 SPA）
│   └── memory_manager.html   # 记忆管理页面
├── data/
│   ├── chroma_db/            # 向量数据库文件
│   ├── capsules.json         # 时间胶囊数据
│   ├── anniversaries.json    # 纪念日数据
│   └── personality.json      # 人格配置
├── .env                      # API Key 配置（不纳入版本控制）
└── start.bat                 # 一键启动脚本
```

## 使用说明

### 切换模型

点击顶栏的模型按钮，在弹出的选择器中：
- 左侧选择供应商
- 右侧选择具体模型
- 在 API 密钥卡片中粘贴对应 Key 并保存（Key 写入本地 `.env` 文件）

### 人格配置

点击左侧边栏底部的设置按钮，进入人格控制面板，可配置：
- **身份** — 角色名称、身份背景、关系定位
- **性格** — 核心特质、禁止倾向
- **语言** — 表达风格、口头语
- **输出** — 格式偏好、回复长度
- **互动** — 主动行为设置
- **记忆** — 记忆提取优先级
- **防幻觉** — 不确定性处理策略

### 自定义供应商

在模型选择器中选择"自定义"，填写：
- API 地址（OpenAI 兼容端点）
- API 密钥
- 模型名称
- 调用格式（OpenAI 兼容 / 扩展参数 / 精简）

## API 文档

服务启动后访问：

```
http://localhost:8000/docs
```

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python · FastAPI · uvicorn |
| AI 调用 | openai-python SDK（兼容多供应商） |
| 向量数据库 | ChromaDB |
| 前端 | 原生 HTML / CSS / JS（单文件） |
| Markdown 渲染 | marked.js |
