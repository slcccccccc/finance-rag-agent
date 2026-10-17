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
| 流式响应 | 进行中 |
| PDF 解析与分块 | 待开始 |
| 向量检索 | 待开始 |
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

---

## 目录结构

```
finance-rag-agent/
├── app/
│   ├── config.py        # 配置读取
│   └── main.py          # FastAPI 入口与接口
├── data/                # 待处理的原始文档（PDF 不提交）
├── eval/                # 评估集与报告
├── .env.example         # 环境变量模板
├── .gitignore
└── requirements.txt
```

---

## 后续计划

- [ ] 补完流式接口
- [ ] 文档解析 + 分块 + 向量入库
- [ ] RAG 问答链路，带引用溯源与拒答
- [ ] 三个工具的 Agent：`search_docs` / `query_reimbursement` / `calc_amount`
- [ ] 20 条测试问题 + 失败案例分析
- [ ] Docker Compose 一键启动

---

## 备注

本项目为个人学习与求职作品，技术选型参考了公开的 AI 应用开发学习路径。
