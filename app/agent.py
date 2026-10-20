"""
Agent：让模型在"检索文档 / 查单据 / 算金额"之间自己选工具。

Function Calling 的本质（先读懂再往下）：
    模型不会真的执行 Python 代码。它只会"点菜"——
    第一回合，它不回答，而是说："我要调用 xxx 工具，参数是 yyy"。
    你的程序收到这个请求，自己去调用对应函数，把结果塞回消息里。
    第二回合，模型拿到结果，才生成最终答案。

    所以一个 function calling 循环就是：
        问模型 → 看它要不要调工具 → 要就执行、把结果塞回去 → 再问 → 直到它不再调工具，给出答案。

为什么不能只靠写 if-else 判断"该查文档还是该查单据"：
    用户的问题千变万化，"报销单审到哪了""这笔钱谁批""超标 20% 是多少"
    你不可能穷举所有说法。让模型自己判断意图，比人工写规则更抗变化。
"""

import json

from app import config, tools
from app.rag import build_context
from app.vectorstore import search

SYSTEM_PROMPT = """你是企业财务助手，能同时查制度、查单据、算金额。

你可以使用这些工具：
- search_docs(question)：查制度文档，返回相关条款。适合"住宿标准""报销要交什么材料"这类问题。
- query_reimbursement(order_id)：按单号查报销单状态。适合"单号 XXX 审到哪了"。
- query_approval(amount)：查某个报销金额需要哪些人审批。适合"X 块钱要谁批"。
- calc_amount(expression)：计算金额。适合"超额 20% 是多少""3200+500 等于多少"。

规则：
1. 先判断问题该用哪个工具，调用它拿结果，再基于结果回答。
2. 如果问题不需要工具（比如闲聊），直接回答。
3. 如果查不到或算不出，如实说明，不要编造。
4. 涉及制度条款的，回答时标注出处页码（如果工具返回里有）。
5. 你只处理财务报销相关的问题。写诗、写文章、讲天气、查股票等与财务无关的请求，
   一律礼貌拒绝，不要真的去执行，也不要用工具硬凑。"""

# 工具的"说明书"。这四段 JSON 就是告诉模型"你有什么工具、怎么用"。
# description 写得清不清楚，直接决定模型会不会选对工具。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在财务制度文档中检索相关条款。用于回答政策、标准、流程类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要检索的问题，尽量用完整句子"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_reimbursement",
            "description": "按报销单号查询该单据的当前审批状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "报销单号，如 BX2026091"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_approval",
            "description": "查询某个报销金额需要经过哪些人审批。",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "报销金额（元）"},
                },
                "required": ["amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_amount",
            "description": "计算一个算术表达式的结果，如超额比例、金额求和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算术表达式，如 15800 * 0.2"},
                },
                "required": ["expression"],
            },
        },
    },
]

# 工具名 → 实际执行的函数。模型点菜，这里端菜。
TOOL_DISPATCH = {
    "search_docs": lambda **kw: _run_search(**kw),
    "query_reimbursement": lambda **kw: tools.query_reimbursement(**kw),
    "query_approval": lambda **kw: tools.query_approval(**kw),
    "calc_amount": lambda **kw: tools.calc_amount(**kw),
}


def _run_search(question: str) -> dict:
    """search_docs 工具：把检索结果转成模型好读的形式。"""
    hits = search(question, top_k=3)
    if not hits:
        return {"found": False, "message": "没有检索到相关内容"}
    # 复用 rag.build_context，带页码，模型能据此标出处
    return {"found": True, "context": build_context(hits)}


def get_client():
    from openai import OpenAI

    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def _execute_tool(name: str, args: dict) -> str:
    """执行一个工具，把结果序列化成字符串塞回给模型。"""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
    try:
        result = fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def chat_with_tools(question: str, max_rounds: int = 5) -> dict:
    """
    带工具调用的对话主循环。

    参数 max_rounds：最多让模型调几次工具。防止模型陷入"一直调工具不回答"的死循环。

    返回：
        answer    最终答案
        tool_calls 模型调用过哪些工具（记录 reasoning 用，面试可讲）
    """
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []  # 记录每一步，方便调试和讲解

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )

        msg = resp.choices[0].message

        # 关键判断：模型这次有没有要求调工具？
        if not msg.tool_calls:
            # 没有 → 它给出了最终答案，循环结束
            return {
                "answer": msg.content or "",
                "tool_calls": trace,
            }

        # 有 → 逐个执行工具，把结果作为 assistant 消息塞回去
        messages.append(msg)  # 先把"模型要调工具"这条消息记下来

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            trace.append({"tool": name, "args": args})

            result_str = _execute_tool(name, args)

            # 把工具结果作为一条 tool 消息追加，模型下一回合会看到它
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_str,
                }
            )

    # 循环用完还没给出答案（极少数情况），兜底返回
    return {
        "answer": "抱歉，我尝试了多次仍未得到结果，请换个问法。",
        "tool_calls": trace,
    }
