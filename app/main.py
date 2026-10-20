"""
财务 Agent：RAG + Tool Calling

进度：
    Day 1 —— /health、/chat、/chat/stream 三个基础接口
    Day 2 —— 文档解析、分块、向量入库、检索
    Day 3 —— /rag/ask（引用溯源 + 拒答）、/rag/ask/stream、/rag/upload
"""
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI

from app import config, ingest, rag, vectorstore

app = FastAPI(
    title="Finance RAG Agent",
    description="财务场景的 RAG + 工具调用 Agent",
    version="0.3.0",
)

# DeepSeek 使用 OpenAI 兼容协议，所以直接复用 openai 的客户端
client = OpenAI(
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_BASE_URL,
)


# ---------- 请求 / 响应模型 ----------
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户问题", min_length=1)
    history: list[dict] = Field(default_factory=list, description="历史消息，可选")


class ChatResponse(BaseModel):
    answer: str
    model: str
    usage: dict


# ---------- 接口 ----------
@app.get("/health")
def health():
    """健康检查：确认服务和密钥都正常。"""
    return {
        "status": "ok",
        "api_key_loaded": bool(config.DEEPSEEK_API_KEY),
        "model": config.DEEPSEEK_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """普通对话：发一句，返回完整回答。"""
    if not config.DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY，请检查 .env 文件")

    messages = [{"role": "system", "content": "你是一个专业的财务助手。"}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    try:
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用大模型失败：{e}")

    return ChatResponse(
        answer=resp.choices[0].message.content,
        model=resp.model,
        usage={
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    流式响应：模型每生成一小段，就立刻推给前端，而不是等全部生成完。

    为什么需要它：
        普通接口要等模型把整段话写完才返回，用户会盯着空白等好几秒。
        流式是边生成边吐字，用户立刻看到第一个字，体感快很多。
        这也是 ChatGPT 那种"一个字一个字冒出来"效果的实现方式。

    传输格式用的是 SSE（Server-Sent Events）：
        每条消息格式固定为  data: {json}\n\n
        最后发一条 data: [DONE] 表示结束。
    """
    if not config.DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY，请检查 .env 文件")

    messages = [{"role": "system", "content": "你是一个专业的财务助手。"}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    # 生成器函数：用 yield 分段产出，而不是一次 return 全部
    def event_generator():
        try:
            # stream=True 是关键，告诉模型"我要一段一段地收"
            stream = client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.3,
                stream=True,
            )

            for chunk in stream:
                # 有些 chunk 不带 choices（比如最后的用量统计），跳过
                if not chunk.choices:
                    continue

                # delta 就是"这一小段新增的文字"
                content = chunk.choices[0].delta.content

                # 第一段和最后一段的 content 常常是 None，要判空
                if content:
                    payload = json.dumps({"delta": content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关掉代理缓冲，否则流式会被攒成一次性返回
        },
    )


# ============================================================
# Day 3：RAG 问答（引用溯源 + 拒答）与文档上传
# ============================================================


class RagRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: int | None = Field(default=None, description="检索几块，默认取 .env 的 TOP_K")
    threshold: float | None = Field(default=None, description="拒答阈值，默认取 .env 的 SCORE_THRESHOLD")


class SourceItem(BaseModel):
    page: int
    section: str
    score: float
    source: str
    preview: str


class RagResponse(BaseModel):
    answer: str
    refused: bool = Field(description="是否走了拒答路径（查不到就不硬答）")
    top_score: float = Field(description="最高检索相似度")
    sources: list[SourceItem] = Field(description="引用来源，按相似度降序、同页去重")
    citations: int = Field(description="答案里出现 [第X页] 的次数")
    model: str | None
    usage: dict | None


@app.post("/rag/ask", response_model=RagResponse)
def rag_ask(req: RagRequest):
    """
    带出处的问答。

    和 /chat 的本质区别：
        /chat 是「模型自由发挥」，答得再流畅也可能是在编。
        /rag/ask 是「模型只能照着资料说」，而且必须在答案里标出依据在第几页，
        查不到就直接拒答。这就是 RAG 存在的意义。
    """
    if not config.DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY，请检查 .env 文件")

    if vectorstore.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="向量库是空的。请先调用 POST /rag/upload 上传文档，或运行 test_vectorstore.py 灌入示例数据。",
        )

    try:
        result = rag.answer(req.question, top_k=req.top_k, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"问答失败：{e}")

    return RagResponse(**result)


@app.post("/rag/ask/stream")
def rag_ask_stream(req: RagRequest):
    """
    /rag/ask 的流式版本。

    事件顺序是：meta（来源先到）→ delta（一个字一个字吐）→ done → [DONE]。
    先发 meta 是有意的：前端可以在答案还没出来时，就把「参考了哪几页」渲染出来。
    """
    if not config.DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY，请检查 .env 文件")

    if vectorstore.count() == 0:
        raise HTTPException(status_code=409, detail="向量库是空的，请先上传文档")

    return StreamingResponse(
        rag.answer_stream(req.question, top_k=req.top_k, threshold=req.threshold),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# 上传相关常量
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB


@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    """
    上传文档并自动走完「解析 → 分块 → 向量化 → 入库」。

    三个校验少一个都是隐患：
      1. 后缀白名单 —— 不是 PDF / TXT / MD 直接拒，免得解析器崩在奇怪格式上
      2. 文件名净化 —— 只用 Path(...).name，挡掉 ../../ 这类路径穿越
      3. 大小上限 —— 挡掉一个几百 MB 的文件把内存和解析时间拖死
    """
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型「{suffix}」，仅支持 {sorted(ALLOWED_SUFFIXES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限",
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.DATA_DIR / filename
    dest.write_bytes(data)

    try:
        info = ingest.ingest_file(dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析入库失败：{e}")

    return {
        "uploaded": filename,
        "bytes": len(data),
        **info,
        "total_chunks_in_db": vectorstore.count(),
    }


@app.get("/rag/stats")
def rag_stats():
    """看一眼当前配置和库里的规模，演示和排查时都用得上。"""
    return {
        "total_chunks": vectorstore.count(),
        "top_k": config.TOP_K,
        "score_threshold": config.SCORE_THRESHOLD,
        "chunk_size": config.CHUNK_SIZE,
        "data_dir": str(config.DATA_DIR),
    }


# ============================================================
# Day 4：带工具调用的 Agent
# ============================================================


class AgentRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)


class AgentResponse(BaseModel):
    answer: str
    tool_calls: list[dict] = Field(description="本次调用过哪些工具，含工具名和参数")


@app.post("/agent/ask", response_model=AgentResponse)
def agent_ask(req: AgentRequest):
    """
    带工具调用的问答。

    和 /rag/ask 的区别：
        /rag/ask 只会检索文档。这个接口让模型在「查文档 / 查单据 / 算金额」之间
        自己选工具，能回答的问题范围更广。
    """
    if not config.DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY，请检查 .env 文件")

    try:
        from app import agent

        result = agent.chat_with_tools(req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent 调用失败：{e}")

    return AgentResponse(**result)
