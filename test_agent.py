"""
Day 4 验收：Agent 能不能自己选对工具。

四个场景，对应四种工具选择，验证模型不是"碰运气"，是真的会判断：
    1. 制度问题        → 应该调 search_docs
    2. 单据状态问题    → 应该调 query_reimbursement
    3. 金额审批问题    → 应该调 query_approval
    4. 计算问题        → 应该调 calc_amount

跑法：
    ./.venv/Scripts/python.exe test_agent.py

注意：会真实调用 DeepSeek，4 次请求，花费可忽略。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app import agent

CASES = [
    ("出差住酒店一晚最多能报多少钱？", "search_docs", "查制度"),
    ("报销单 BX2026091 审到哪一步了？", "query_reimbursement", "查单据"),
    ("三万块钱要经过哪些人审批？", "query_approval", "查审批权限"),
    ("如果报销 15800 元，超过标准 20% 的部分是多少？", "calc_amount", "算金额"),
]


def run_case(question: str, expect_tool: str, note: str) -> bool:
    print("\n" + "-" * 66)
    print(f"❓ {question}")
    print(f"   预期调用工具：{expect_tool}  （{note}）")

    result = agent.chat_with_tools(question)
    called = [c["tool"] for c in result["tool_calls"]]

    print(f"   实际调用工具：{called if called else '（没调工具）'}")

    print()
    for line in result["answer"].splitlines():
        if line.strip():
            print("   " + line)

    ok = expect_tool in called
    if ok:
        print(f"\n   ✅ 选对了工具（{expect_tool}）")
    else:
        print(f"\n   ❌ 期望 {expect_tool}，实际 {called}")
    return ok


def main() -> None:
    print("=" * 66)
    print("Day 4 验收 · Function Calling")
    print("=" * 66)

    results = [run_case(q, t, n) for q, t, n in CASES]

    print()
    print("=" * 66)
    print("汇总")
    print("=" * 66)
    print(f"通过 {sum(results)} / {len(CASES)}")

    if all(results):
        print("\nDay 4 目标达成：模型能自己判断该用哪个工具。")
    else:
        print("\n有未通过的，看上面的工具调用记录。")
        print("如果是模型选错工具，多半是工具的 description 写得不够清楚，去调 TOOLS。")


if __name__ == "__main__":
    main()
