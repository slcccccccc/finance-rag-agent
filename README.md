# Finance RAG Agent

财务场景的 AI Agent：既能查制度文档，也能查业务单据。

> 基于 FastAPI + DeepSeek + 向量检索 + Tool Calling 构建。
> 这个项目解决的问题是：财务人员有两个高频问题，一类要翻制度（"差旅住宿标准是多少"），一类要查数据（"单号 BX2026091 审到哪一步了"）。传统做法要打开两个系统分别找，现在用一句话就能问。

---

## 项目状态

| 模块 | 状态 |
|---|---|
| FastAPI 服务骨架 | 已完成 |
| DeepSeek 对话接口 | 已完成 |
| 流式响应（SSE） | 已完成 |
| 接口测试脚本 | 已完成 |
| PDF 解析与分块 | 已完成 |
| 向量检索 | 已完成 |
| 引用溯源与拒答 | 已完成 |
| 文档上传入库 | 已完成 |
| Tool Calling（查单据 / 算金额） | 已完成 |
| 评估集（20 条） | 已完成 |
| Docker 化 | 待开始 |

---

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **大模型**：DeepSeek（`deepseek-chat`，OpenAI SDK 兼容协议）
- **Embedding**：`BAAI/bge-small-zh-v1.5`（本地运行，CPU 可跑）
- **向量库**：Chroma + **PostgreSQL/pgvector**（单据数据，Docker 容器）
- **文档解析**：PyMuPDF
- **容器**：Docker + Docker Compose

---

## 快速开始

### 1. 准备环境

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

> 注意：`.env` 已加入 `.gitignore`，不会被提交。**不要**把密钥写进代码里。

### 3. 启动服务

```bash
uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/docs> 即可看到接口文档并直接调试。

---

## 接口说明

### `GET /health`

健康检查，确认服务与密钥状态。

```json
{
  "status": "ok",
  "api_key_loaded": true,
  "model": "deepseek-chat"
}
```

### `POST /chat`

普通对话接口。

```json
// 请求
{ "message": "差旅住宿标准是多少" }

// 响应
{
  "answer": "...",
  "model": "deepseek-chat",
  "usage": { "prompt_tokens": 25, "completion_tokens": 80, "total_tokens": 105 }
}
```

### `POST /chat/stream`

流式对话接口，基于 SSE（Server-Sent Events）。模型边生成边推送，适合需要「打字机效果」的场景。

```json
// 请求
{ "message": "介绍一下差旅报销流程" }
```

响应是 `text/event-stream`，每段格式为 `data: {json}` + 空行，以 `data: [DONE]` 结束：

```
data: {"delta": "差旅"}

data: {"delta": "报销"}

data: [DONE]
```

> 注意：这个接口在 Swagger 的 `/docs` 页面看不出流式效果（会等全部返回才显示），需用 `test_api.py` 或 curl 验证。

---

## RAG 接口

这是这个项目的核心。与 `/chat` 的本质区别：

| | `/chat` | `/rag/ask` |
|---|---|---|
| 信息来源 | 模型自己的知识 | **只有你上传的文档** |
| 会不会编 | 会 | 查不到就拒答，不会编 |
| 能不能核对 | 不能 | 每个结论带 `[第X页]`，可回原文核对 |

### `POST /rag/ask`

```json
// 请求
{ "question": "出差住酒店一晚最多能报多少钱" }

// 响应
{
  "answer": "住宿费上限取决于出差目的地城市类别：一类城市（北京、上海、广州、深圳）500 元/人/晚，二类城市 400 元/人/晚，三类城市 300 元/人/晚 [第1页]。",
  "refused": false,
  "top_score": 0.659,
  "sources": [
    { "page": 1, "section": "第二章　差旅费用标准", "score": 0.659,
      "source": "财务报销管理制度.pdf", "preview": "第四条　住宿费标准按出差目的地城市类别执行…" }
  ],
  "citations": 2,
  "model": "deepseek-chat",
  "usage": { "prompt_tokens": 640, "completion_tokens": 63, "total_tokens": 703 }
}
```

**查不到时的响应**（`refused: true`，此时**不会调用大模型**）：

```json
{
  "answer": "根据现有资料无法回答这个问题。检索到的最相关内容相似度为 0.44，低于阈值 0.51，为避免给出错误信息，我不做推测。",
  "refused": true,
  "top_score": 0.439,
  "sources": [],
  "citations": 0,
  "model": null,
  "usage": null
}
```

### `POST /rag/ask/stream`

流式版本。事件顺序 `meta` → `delta` → `done` → `[DONE]`：

```
data: {"type": "meta", "sources": [...], "top_score": 0.7298, "refused": false}

data: {"type": "delta", "text": "出差期间伙食补助"}

data: {"type": "done", "citations": 1}

data: [DONE]
```

先发 `meta` 是有意的：前端可以在答案还没吐出来时，就把「参考了哪几页」先渲染出来。

### `POST /rag/upload`

上传文档并自动走完「解析 → 分块 → 向量化 → 入库」。支持 `.pdf` / `.txt` / `.md`，上限 20 MB。

三重校验：后缀白名单、文件名净化（挡路径穿越）、大小上限。

```json
{ "uploaded": "制度.pdf", "bytes": 364129, "pages": 3, "chars": 1874,
  "chunks": 20, "total_chunks_in_db": 20 }
```

### `GET /rag/stats`

查看当前配置与库内规模，排查问题时用。

---

## Agent 接口（Function Calling）

`/rag/ask` 只会检索文档。但财务人员有一类问题它答不了——「单号 BX2026091 审到哪了」「三万块要谁批」这类需要**查单据、算金额**的问题。

`/agent/ask` 让模型在四个工具之间**自己判断该用哪个**：

| 工具 | 干什么 | 数据源 | 典型问题 |
|---|---|---|---|
| `search_docs` | 查制度文档 | Chroma 向量库 | 住宿标准是多少 |
| `query_reimbursement` | 查报销单状态 | PostgreSQL（Docker 容器） | BX2026091 审到哪了 |
| `query_approval` | 查金额审批权限 | 审批区间表 | 三万块要谁批 |
| `calc_amount` | 安全计算金额 | AST 白名单求值 | 超标 20% 是多少 |

### 运行数据库

单据数据存在 Docker 里的 PostgreSQL（pgvector 镜像）中，连接串读自 `.env` 的 `DATABASE_URL`。

```bash
# 启动数据库容器（首次）
docker run -d --name ragdb -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb -p 5432:5432 pgvector/pgvector:pg16

# 建表并灌入示例单据（跑一次即可，幂等）
./.venv/Scripts/python.exe init_db.py
```

> 国内拉不动 Docker Hub 的话，走镜像源：
> `docker pull docker.m.daocloud.io/pgvector/pgvector:pg16` 再 `docker tag` 改名。

### `POST /agent/ask`

```json
// 请求
{ "question": "三万块钱要经过哪些人审批？" }

// 响应
{
  "answer": "30,000 元属于 20,000 至 50,000 元区间，需要经过：部门负责人、财务负责人、分管副总。",
  "tool_calls": [ { "tool": "query_approval", "args": { "amount": 30000 } } ]
}
```

`tool_calls` 记录了模型调用过哪些工具、用了什么参数——这对调试和评估都有用。

### 为什么要有工具，而不是全靠检索

这是整个项目最重要的一个设计判断，直接来自一个真实的失败案例：

- 问「花三万块钱要经过谁审批」，纯检索会失败——因为 embedding 只比语义相似，**不会做数值推理**，它没法把「三万」映射到「20,000 至 50,000」这个区间。
- 靠 `top_k` 多取几块让模型碰运气，是「大概率对」，不是「确定对」。

所以金额审批这类**强结构化数据**，抽成 `query_approval` 工具精确查表：**工具是「确定对」，检索是「大概率对」。财务场景，「大概率」不够。**

`calc_amount` 另一个细节：用 AST 白名单解析，而不是 `eval()`，避免用户注入恶意代码。

---

## 拒答阈值是算出来的，不是拍脑袋定的

阈值定太高，资料里明明有答案也会被拒；定太低，没有依据的也硬答，那就是幻觉。

`calibrate_threshold.py` 的做法：把 8 个「该答的」和 6 个「该拒的」问题各跑一遍，只看最高相似度，找两组分数之间有没有一条干净的缝。

| 分组 | 极值 | 问题 |
|---|---|---|
| 该答的 · 最低分 | **0.5830** | 花三万块钱要经过哪些人审批 |
| 该拒的 · 最高分 | **0.4330** | 员工食堂几点开饭 |

缝隙是 `0.43 ~ 0.58`，阈值取中间 **0.51**。

> 这个数**不能跨数据集复用**。换一批文档要重新跑校准。
> 原来的 0.35 是随手写的，实测会让「员工食堂几点开饭」通过，接着模型就会拿无关条款硬凑答案。

---

## 目录结构

```
finance-rag-agent/
├── app/
│   ├── config.py        # 配置读取
│   ├── main.py          # FastAPI 入口与接口
│   ├── loader.py        # PDF / 文本解析，保留页码
│   ├── chunking.py      # 按章条分块，供向量化
│   ├── embedding.py     # bge 中文向量模型封装
│   ├── vectorstore.py   # Chroma 封装：入库 / 检索
│   ├── ingest.py        # 入库流水线：读取→分块→向量化→存储
│   ├── rag.py           # RAG 链路：组装上下文 / 提示词 / 拒答 / 溯源
│   ├── tools.py         # 工具集：查单据 / 查审批 / 安全计算
│   └── agent.py         # Agent：Function Calling 主循环
├── data/                # 待处理的原始文档（真实业务 PDF 不提交）
├── eval/                # 评估集与报告（含失败案例分析）
├── calibrate_threshold.py  # 拒答阈值校准
├── test_api.py          # 基础接口测试
├── test_embedding.py    # 向量模型自检
├── test_chunking.py     # 分块自检（打印每一块供人工判断）
├── test_vectorstore.py  # 检索自检（Top-1 / Top-3 准确率）
├── test_rag.py          # 问答验收（答得出 / 标出处 / 答不了会拒）
├── test_agent.py        # Agent 验收（四类问题，看模型是否选对工具）
├── .env.example         # 环境变量模板
├── .gitignore
└── requirements.txt
```

---

## 自检与验收

脚本按 RAG 的三道工序排列，每一步都要肉眼确认结果合理再往下走：

```bash
./.venv/Scripts/python.exe test_embedding.py        # 1. 模型能跑，语义相似度是否合理
./.venv/Scripts/python.exe test_chunking.py         # 2. 每一块单独看，能不能看懂
./.venv/Scripts/python.exe test_vectorstore.py      # 3. 一句话能不能搜到那一块
./.venv/Scripts/python.exe calibrate_threshold.py   # 4. 用数据定拒答阈值（不花钱）
./.venv/Scripts/python.exe test_rag.py              # 5. 完整问答链路（会调用大模型）
./.venv/Scripts/python.exe test_agent.py            # 6. Agent 工具调用（四类问题选工具）
```

**检索基线**（`财务报销管理制度.pdf`，3 页 / 1874 字 / 20 块）：

| 指标 | 结果 |
|---|---|
| Top-1 是正确块 | 4 / 5（80%） |
| Top-3 含正确块 | 5 / 5（100%） |
| 该答的最低分 | 0.5830 |
| 该拒的最高分 | 0.4330 |
| 阈值 | 0.51（取缝隙中间） |

**问答验收**（4 题）：

| 问题 | 预期 | 结果 |
|---|---|---|
| 出差住酒店一晚最多能报多少钱 | 答 + 标页码 | ✅ 引用 2 处 |
| 花三万块钱要经过哪些人审批 | 答 + 标页码 | ✅ 引用 2 处 |
| 员工食堂几点开饭 | 拒答 | ✅ 拒答 |
| 帮我写一首关于春天的诗 | 拒答 | ✅ 拒答 |

**一个值得注意的发现**：第 2 题在纯检索阶段 Top-1 是错的（搜到第十三条而非审批表），但接上 `top_k=5` 之后答对了——因为第十二条的表格也在检索结果里，模型自己做了取舍。

> 这说明 **Top-1 检索错 ≠ 回答错**。多取几块给了模型纠偏的余地。
> 但这不能作为"检索可以不精确"的借口：取多了会挤占上下文、增加成本和干扰项。
> 金额审批这类**结构化数据**仍然应该走工具精确查表（Day 4），而不是指望模型每次都救回来。

详见 [`eval/report.md`](eval/report.md)。

**Agent 验收**（4 题）：

| 问题 | 预期工具 | 结果 |
|---|---|---|
| 出差住酒店一晚最多能报多少钱 | search_docs | ✅ |
| 报销单 BX2026091 审到哪一步了 | query_reimbursement | ✅ |
| 三万块钱要经过哪些人审批 | query_approval | ✅ |
| 报销 15800 元超标准 20% 是多少 | calc_amount | ✅ |

四题全部选对工具，其中「三万块」一题从 Day 2 的检索失败，变成了工具精确查表命中。

---

## 后续计划

- [x] 补完流式接口
- [x] 文档解析 + 分块 + 向量入库
- [x] RAG 问答链路，带引用溯源与拒答
- [x] 文档上传入库
- [x] 多工具 Agent：`search_docs` / `query_reimbursement` / `query_approval` / `calc_amount`
- [x] 20 条测试问题 + 失败案例分析
- [ ] Docker Compose 一键启动

---

## 备注

本项目为个人学习与求职作品，技术选型参考了公开的 AI 应用开发学习路径。
