"""
RAG 问答链路：检索 → 组装上下文 → 让模型带着证据回答 → 引用溯源 + 拒答。

为什么要把「拒答」写在这一层，而不是丢给模型判断：
    模型天生倾向于"给出一个答案"。你问它资料里没有的东西，它也会想办法答，
    因为它被训练成"有用"。把判断权交给它，等于把幻觉的风险交给它。
    所以我们在**调模型之前**就用检索分数拦一道：
    最高相似度低于阈值，连模型都不调，直接返回固定话术。
    这一步不花钱、不会错、而且可以解释给用户听。

为什么引用溯源必须靠"把页码塞进上下文"：
    模型只能看见你给它的东西。上下文里没有页码，它要求"标出处"就只能编一个。
    所以 build_context 里必须带页码，这是溯源的全部实现。
"""
import json

from openai import OpenAI

from app import config, vectorstore

# 拒答话术。写成常量，方便评估脚本判断"是不是拒答了"
REFUSAL_PREFIX = "根据现有资料无法回答"

SYSTEM_PROMPT = """你是企业财务制度问答助手。你的唯一信息来源是下面提供的【参考资料】。

必须遵守的规则：
1. 只依据【参考资料】回答，不使用你自己的知识，不进行推测和补充。
2. 如果【参考资料】里没有能回答问题的依据，直接说"根据现有资料无法回答"，
   并说明缺少哪方面信息。绝对不要编造。
3. 每个结论后面必须标注依据来源，格式为 [第X页]，X 是资料中标注的页码。
4. 涉及金额、天数、审批层级时，必须准确引用原文数字，不要换算、不要近似。
5. 回答要简洁，直接给结论和数字，不要复述制度全文。"""


def get_client() -> OpenAI:
    """DeepSeek 用 OpenAI 兼容协议。"""
    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )


def build_context(hits: list[dict]) -> str:
    """
    把检索结果拼成带页码的上下文。

    页码放在每一段的头部，模型才能知道"这段话出自第几页"，
    规则 3 要求的 [第X页] 才有东西可标。
    """
    parts = []
    for i, h in enumerate(hits, 1):
        header = f"[片段{i} | 第{h['page']}页 | {h['section']}]"
        parts.append(f"{header}\n{h['text']}")
    return "\n\n".join(parts)


def build_messages(question: str, context: str) -> list[dict]:
    """组装发给模型的两条消息。"""
    user = f"""【参考资料】
{context}

【问题】
{question}

请依据上面【参考资料】回答这个问题，并标注依据页码。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """检索这一步单独抽出来，方便评估脚本只测检索、不花钱调模型。"""
    return vectorstore.search(question, top_k=top_k or config.TOP_K)


def should_refuse(hits: list[dict], threshold: float | None = None) -> bool:
    """
    是否拒答。

    判断依据是**最高分**，不是平均分：
    只要有一块足够相关，就值得让模型试着回答；
    如果最高的都不相关，剩下的更不可能相关。
    """
    if not hits:
        return True
    th = config.SCORE_THRESHOLD if threshold is None else threshold
    return hits[0]["score"] < th


def _make_sources(hits: list[dict]) -> list[dict]:
    """去重后的来源列表，给前端展示用。同一页只保留分数最高的一条。"""
    seen: dict[int, dict] = {}
    for h in hits:
        page = h["page"]
        if page not in seen:
            seen[page] = {
                "page": page,
                "section": h["section"],
                "score": h["score"],
                "source": h["source"],
                "preview": h["text"].replace("\n", " ")[:60],
            }
    return sorted(seen.values(), key=lambda x: -x["score"])


def count_citations(answer_text: str) -> int:
    """
    统计答案里出现了几处 [第X页] 标注。

    这是用来**验证提示词有没有生效**的：如果 citation 一直是 0，
    说明规则 3 没起作用，或者上下文里没带页码。
    不测这个，你永远不知道溯源是真做了还是摆设。
    """
    import re

    return len(re.findall(r"第\s*\d+\s*页", answer_text))


def answer(
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> dict:
    """
    完整问答。返回结构直接可转成 JSON 给接口用。

    返回字段：
        answer      答案文本
        refused     是否走了拒答路径
        top_score   最高检索相似度
        sources     来源列表（页码 / 章节 / 相似度）
        citations   答案里出现 [第X页] 的次数
        model / usage
    """
    hits = retrieve(question, top_k)
    top_score = hits[0]["score"] if hits else 0.0

    if should_refuse(hits, threshold):
        th = config.SCORE_THRESHOLD if threshold is None else threshold
        return {
            "answer": (
                f"{REFUSAL_PREFIX}这个问题。"
                f"检索到的最相关内容相似度为 {top_score:.2f}，低于阈值 {th:.2f}，"
                f"为避免给出错误信息，我不做推测。"
                f"建议确认该问题是否属于制度覆盖范围，或换一种说法提问。"
            ),
            "refused": True,
            "top_score": round(top_score, 4),
            "sources": [],
            "citations": 0,
            "model": None,
            "usage": None,
        }

    context = build_context(hits)
    messages = build_messages(question, context)

    client = get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.1,      # 制度问答要稳定，不要发挥
        stream=False,
    )

    text = resp.choices[0].message.content or ""
    usage = resp.usage

    return {
        "answer": text,
        "refused": False,
        "top_score": round(top_score, 4),
        "sources": _make_sources(hits),
        "citations": count_citations(text),
        "model": config.DEEPSEEK_MODEL,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        } if usage else None,
    }


def answer_stream(question: str, top_k: int | None = None, threshold: float | None = None):
    """
    流式版本。SSE 三种事件：

        data: {"type": "meta", "sources": [...], "top_score": 0.66}
        data: {"type": "delta", "text": "根据"}
        data: {"type": "done", "citations": 2}
        data: [DONE]

    先发 meta 再发 delta 是有意的：前端可以先渲染"参考了哪几页"，
    不用等答案吐完才知道引用了什么。
    """
    hits = retrieve(question, top_k)
    top_score = hits[0]["score"] if hits else 0.0

    if should_refuse(hits, threshold):
        th = config.SCORE_THRESHOLD if threshold is None else threshold
        msg = (
            f"{REFUSAL_PREFIX}这个问题。"
            f"检索到的最相关内容相似度为 {top_score:.2f}，低于阈值 {th:.2f}，"
            f"为避免给出错误信息，我不做推测。"
        )
        yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'top_score': round(top_score, 4), 'refused': True}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'delta', 'text': msg}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'citations': 0}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    sources = _make_sources(hits)
    meta = {"type": "meta", "sources": sources, "top_score": round(top_score, 4), "refused": False}
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    context = build_context(hits)
    messages = build_messages(question, context)

    client = get_client()
    stream = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    full = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            full.append(delta)
            yield f"data: {json.dumps({'type': 'delta', 'text': delta}, ensure_ascii=False)}\n\n"

    citations = count_citations("".join(full))
    yield f"data: {json.dumps({'type': 'done', 'citations': citations}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
