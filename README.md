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
| 引用溯源与拒答 | 待开始 |
| Tool Calling（查单据 / 算金额） | 待开始 |
| 评估集（20 条） | 待开始 |
| Docker 化 | 待开始 |

---

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **大模型**：DeepSeek（`deepseek-chat`，OpenAI SDK 兼容协议）
- **Embedding**：`BAAI/bge-small-zh-v1.5`（本地运行，CPU 可跑）
- **向量库**：Chroma（开发阶段）→ PostgreSQL + pgvector（生产形态）
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
│   └── ingest.py        # 入库流水线：读取→分块→向量化→存储
├── data/                # 待处理的原始文档（PDF 不提交）
├── eval/                # 评估集与报告（含失败案例分析）
├── test_api.py          # 接口测试
├── test_embedding.py    # 向量模型自检
├── test_chunking.py     # 分块自检（打印每一块供人工判断）
├── test_vectorstore.py  # 检索自检（Top-1 / Top-3 准确率）
├── .env.example         # 环境变量模板
├── .gitignore
└── requirements.txt
```

---

## 检索自检

三个脚本按顺序跑，对应 RAG 的三道工序，每一步都要肉眼确认结果合理再往下走：

```bash
./.venv/Scripts/python.exe test_embedding.py     # 1. 模型能跑，语义相似度是否合理
./.venv/Scripts/python.exe test_chunking.py      # 2. 每一块单独看，能不能看懂
./.venv/Scripts/python.exe test_vectorstore.py   # 3. 一句话能不能搜到那一块
```

**当前基线**（`财务报销管理制度.pdf`，3 页 / 1874 字 / 20 块）：

| 指标 | 结果 |
|---|---|
| Top-1 是正确块 | 4 / 5（80%） |
| Top-3 含正确块 | 5 / 5（100%） |
| 无关问题最高分 | 0.3362（阈值 0.35，可据此拒答） |

唯一失败项是**数字类问题**：「花三万块钱要经过哪些人审批」搜不到审批权限表——embedding 不会把「三万」映射到「20,000 至 50,000」这个区间。这类结构化查询将在 Day 4 交给 `query_reimbursement` 工具精确查表，而不是靠向量猜。详见 [`eval/report.md`](eval/report.md)。

---

## 后续计划

- [x] 补完流式接口
- [x] 文档解析 + 分块 + 向量入库
- [ ] RAG 问答链路，带引用溯源与拒答
- [ ] 三个工具的 Agent：`search_docs` / `query_reimbursement` / `calc_amount`
- [ ] 20 条测试问题 + 失败案例分析
- [ ] Docker Compose 一键启动

---

## 备注

本项目为个人学习与求职作品，技术选型参考了公开的 AI 应用开发学习路径。
