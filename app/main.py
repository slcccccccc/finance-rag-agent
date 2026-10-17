"""
财务 Agent：RAG + Tool Calling

Day 1 目标（当前已完成）：/health 和 /chat 两个接口能跑通
Day 1 待办：把 /chat/stream 的流式接口补完（找到下方的 TODO）
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

from app import config

app = FastAPI(
    title="Finance RAG Agent",
    description="财务场景的 RAG + 工具调用 Agent",
    version="0.1.0",
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
    Day 1 待办：实现流式响应。

    参考思路：
        1. 把 create(...) 的参数加上 stream=True
        2. 用 FastAPI 的 StreamingResponse 包一个生成器
        3. 生成器里遍历 chunk，取 chunk.choices[0].delta.content
        4. 注意 delta.content 可能是 None，要判空

    验收：用 curl 或 /docs 调用后，文字是一段一段出现的，不是等全部生成完才显示。
    """
    # TODO: 用 StreamingResponse 实现
    raise HTTPException(status_code=501, detail="流式接口尚未实现，见代码注释")
