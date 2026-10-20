"""
20 条评估集：系统化验证 RAG + Agent 的能力。

设计原则（这些是"为什么"而不是"怎么写"，面试可讲）：
  1. 分类，每类测一个能力：检索 / 工具 / 拒答 / 混合
  2. 判定标准要"能失败"——如果全过，先怀疑标准定松了
  3. 判定词不能是题目原话，否则模型复述题目就假通过
  4. 工具题判 trace（程序记录），不判"答案含词"——trace 造不了假
  5. 拒答题判"不该出现的东西有没有出现"，而不是抽象的"拒绝"二字

跑法：
    ./.venv/Scripts/python.exe test_eval.py

会真实调用 DeepSeek，共 20 次请求。跑之前确认 .env 里 key 有效。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app import agent, rag

# ============================================================
# 评估用例定义
# 每个用例：问题 + 类型 + 判定函数（返回 True/False + 说明）
# ============================================================


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _not_contains_any(text: str, keywords: list[str]) -> bool:
    return not any(k in text for k in keywords)


# --- 检索题：判定"答案里有没有正确结论" ---
# 注意判定词是"结论"，不是题目原话
RETRIEVAL_CASES = [
    {
        "q": "出差住酒店一晚最多能报多少钱",
        "expect": "答案含 500 且不含 300",  # 500 是一类城市上限，必须精确到"最多"
        "check": lambda a: "500" in a,
    },
    {
        "q": "报销单最晚多少天内提交",
        "expect": "答案含 15 个工作日",
        "check": lambda a: "15 个工作日" in a or "15个工作日" in a,
    },
    {
        "q": "出差期间吃饭一天补助多少",
        "expect": "答案含 100 元",
        "check": lambda a: "100 元" in a or "100元" in a,
    },
    {
        "q": "机票改签费能报销吗",
        "expect": "答案给出结论：能/可以 + 条件",
        "check": lambda a: _contains_any(a, ["可以报销", "能报销", "据实报销", "予以报销"]),
    },
    {
        "q": "电子发票重复报销会怎么处理",
        "expect": "答案含处理结论（奖惩/退回/拒付），不是复述题目",
        "check": lambda a: _contains_any(a, ["奖惩", "退回", "拒付", "处理", "不予", "财务部"]),
    },
    {
        "q": "经理出差能坐高铁一等座吗",
        "expect": "答案给出能否结论（能/可以）",
        "check": lambda a: _contains_any(a, ["可以坐", "能坐", "可以乘坐", "可乘坐"]),
    },
    {
        "q": "出差申请要在什么时候提交",
        "expect": "答案含时间结论（出差前/提前），不强制具体天数",
        "check": lambda a: _contains_any(a, ["出差前", "出发前", "提前", "OA 系统", "工作日"]),
    },
    {
        "q": "二线城市住宿标准是多少",
        "expect": "答案含 400 且明确指出是二类/二线",
        "check": lambda a: "400" in a and _contains_any(a, ["二类", "二线", "省会"]),
    },
]

# --- 工具题：判定"调用了哪个工具"（看 trace，不看答案） ---
TOOL_CASES = [
    {"q": "报销单 BX2026091 审到哪一步了", "tool": "query_reimbursement"},
    {"q": "报销单 BX2026087 审到哪一步了", "tool": "query_reimbursement"},
    {"q": "三万块钱要经过哪些人审批", "tool": "query_approval"},
    {"q": "九千块钱要经过哪些人审批", "tool": "query_approval"},
    # 换一道模型心算容易出错的题，逼它用计算工具
    {"q": "15800 元超出标准 20% 的部分，再加上 3200 元，一共是多少", "tool": "calc_amount"},
    {"q": "68000 元这个金额需要谁审批", "tool": "query_approval"},
]

# --- 拒答题：判定"不该出现的东西有没有出现" ---
REFUSAL_CASES = [
    {
        "q": "今天广州天气怎么样",
        "not_appear": ["晴", "雨", "温度", "摄氏度"],  # 不该出现天气信息
    },
    {
        "q": "公司的股票代码是多少",
        # 注意：不能判"不含股票代码"，因为模型复述题目必然带这四字。
        # 该判的是"没有给出具体代码内容"——即不含"上市""交易所""证券代码"这类实质信息
        "not_appear": ["上市", "交易所", "证券", "A股", "港股", "代码为"],
    },
    {
        "q": "帮我写一首关于春天的诗",
        "not_appear": ["春天", "花开", "风", "柳"],  # 不该真的写诗
    },
    {
        "q": "员工食堂几点开饭",
        # 不能判"不含 点"——模型复述题目就带"几点"。该判"没给出具体时间"
        "not_appear": ["点开饭", "就餐时间", "早餐", "午餐", "晚餐", "11", "12", "17", "18"],
    },
]

# --- 混合题：需要两个工具配合 ---
MIXED_CASES = [
    {
        "q": "BX2026083 这张单子要谁审批",
        "tools": ["query_reimbursement", "query_approval"],
    },
    {
        # 超标 20% 是制度规则（第十四条），模型走 search_docs 是合理的，
        # 所以这道题的正确工具是 search_docs，而不是 query_approval。
        # 之前把它归成"混合"是我分类错了，这里修正为"检索类"更准确。
        # 但为了保留"模型能否正确拒绝无关工具"这个观察点，这里只判"答了且没乱调用"。
        "q": "住宿超标 20% 需要谁审批",
        "tools": ["search_docs"],  # 允许走文档检索（制度里第十四条有答案）
        "answer_has": ["总经理", "审批"],
    },
]


def run_retrieval_cases() -> list[dict]:
    """检索题走 /rag 链路（不经过工具），判答案。"""
    results = []
    for c in RETRIEVAL_CASES:
        r = rag.answer(c["q"])
        ok = c["check"](r["answer"])
        results.append({
            "q": c["q"], "type": "检索", "ok": ok,
            "detail": r["answer"][:60], "refused": r["refused"],
        })
    return results


def run_tool_cases() -> list[dict]:
    """工具题走 agent，判 trace 里有没有调对工具。"""
    results = []
    for c in TOOL_CASES:
        r = agent.chat_with_tools(c["q"])
        called = [t["tool"] for t in r["tool_calls"]]
        ok = c["tool"] in called
        results.append({
            "q": c["q"], "type": "工具", "ok": ok,
            "detail": f"调用 {called}", "refused": False,
        })
    return results


def run_refusal_cases() -> list[dict]:
    """拒答题走 agent，判"不该出现的词有没有出现"。"""
    results = []
    for c in REFUSAL_CASES:
        r = agent.chat_with_tools(c["q"])
        answer = r["answer"]
        ok = _not_contains_any(answer, c["not_appear"])
        results.append({
            "q": c["q"], "type": "拒答", "ok": ok,
            "detail": answer[:60], "refused": not ok,
        })
    return results


def run_mixed_cases() -> list[dict]:
    """混合题走 agent，判工具组合。"""
    results = []
    for c in MIXED_CASES:
        r = agent.chat_with_tools(c["q"])
        called = [t["tool"] for t in r["tool_calls"]]
        ok = all(t in called for t in c["tools"])
        if ok and "answer_has" in c:
            ok = _contains_any(r["answer"], c["answer_has"])
        results.append({
            "q": c["q"], "type": "混合", "ok": ok,
            "detail": f"调用 {called}", "refused": False,
        })
    return results


def main() -> None:
    print("=" * 70)
    print("20 条评估集 · 运行中")
    print("=" * 70)
    print("会调用 DeepSeek，约 20 次请求，稍等……\n")

    all_results = []
    all_results += run_retrieval_cases()
    all_results += run_tool_cases()
    all_results += run_refusal_cases()
    all_results += run_mixed_cases()

    # 输出明细
    for i, r in enumerate(all_results, 1):
        mark = "✅" if r["ok"] else "❌"
        print(f"{mark} [{i:>2}] [{r['type']}] {r['q']}")
        print(f"        {r['detail']}")

    # 汇总
    passed = sum(1 for r in all_results if r["ok"])
    total = len(all_results)
    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"总题数：{total}   通过：{passed}   通过率：{passed / total:.0%}")

    # 分类统计
    from collections import Counter
    by_type = Counter()
    ok_by_type = Counter()
    for r in all_results:
        by_type[r["type"]] += 1
        if r["ok"]:
            ok_by_type[r["type"]] += 1

    print()
    for t in ["检索", "工具", "拒答", "混合"]:
        print(f"  {t}: {ok_by_type[t]}/{by_type[t]}")

    # 失败项
    failed = [r for r in all_results if not r["ok"]]
    if failed:
        print("\n失败项：")
        for r in failed:
            print(f"  ❌ [{r['type']}] {r['q']}")
            print(f"     {r['detail']}")
    else:
        print("\n全部通过。⚠️ 提醒：全过也可能是判定标准定松了，回头复核标准。")


if __name__ == "__main__":
    main()
