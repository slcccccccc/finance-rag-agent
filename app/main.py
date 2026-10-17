"""
财务 Agent：RAG + Tool Calling

Day 1 目标（当前已完成）：/health 和 /chat 两个接口能跑通
Day 1 待办：把 /chat/stream 的流式接口补完（找到下方的 TODO）
"""
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
