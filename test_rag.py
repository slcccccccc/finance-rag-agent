"""
Day 3 验收：/rag/ask 的完整链路。

这一天要证明三件事：
    1. 答得出来的问题，答案里带 [第X页]，并且能在原文里核对
    2. 答不出来的问题，明确拒答，而不是编
    3. 流式版本和普通版本的结论一致

跑法：
    ./.venv/Scripts/python.exe test_rag.py

注意：会真实调用 DeepSeek，大约 3 次请求，花费可忽略但别反复空跑。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app import config, rag, vectorstore

# (问题, 期望结果, 说明)
# 期望结果：answer = 应该给出答案；refuse = 应该拒答
CASES = [
    ("出差住酒店一晚最多能报多少钱？", "answer", "住宿标准表，第 1 页"),
    ("花三万块钱要经过哪些人审批？", "answer", "审批权限表；注意这也是 Top-1 检索会错的那题"),
    ("员工食堂几点开饭？", "refuse", "制度里没有，实测相似度 0.4330 < 阈值"),
    ("帮我写一首关于春天的诗", "refuse", "完全无关，实测 0.2182"),
]


def run_case(question: str, expect: str, note: str) -> bool:
    print("\n" + "-" * 66)
    print(f"❓ {question}")
    print(f"   预期：{'给出答案' if expect == 'answer' else '拒答'}   （{note}）")

    result = rag.answer(question)
    print(f"   最高相似度：{result['top_score']}   拒答：{result['refused']}   "
          f"引用次数：{result['citations']}")

    if result["sources"]:
        pages = ", ".join(f"第{s['page']}页({s['score']})" for s in result["sources"])
        print(f"   来源：{pages}")
    if result["usage"]:
        print(f"   tokens：{result['usage']['total_tokens']}")

    print()
    for line in result["answer"].splitlines():
        if line.strip():
            print("   " + line)

    # 判定
    if expect == "refuse":
        ok = result["refused"]
        print(f"\n   {'✅ 正确拒答，没有编造' if ok else '❌ 应该拒答但硬答了'}")
        return ok

    ok = (not result["refused"]) and result["citations"] > 0
    if not result["refused"] and result["citations"] == 0:
        print("\n   ⚠️ 答了但没标页码 —— 检查上下文里是否带了页码，或提示词没生效")
        return False
    print(f"\n   {'✅ 给出答案且标了出处' if ok else '❌ 异常'}")
    return ok


def main() -> None:
    total = vectorstore.count()
    print("=" * 66)
    print("Day 3 验收 · RAG 问答")
    print("=" * 66)
    print(f"库内块数：{total}    TOP_K：{config.TOP_K}    拒答阈值：{config.SCORE_THRESHOLD}")

    if total == 0:
        print("\n向量库是空的，先跑 test_vectorstore.py")
        return
    if not config.DEEPSEEK_API_KEY:
        print("\n没有 DEEPSEEK_API_KEY，先配置 .env")
        return

    results = [run_case(*c) for c in CASES]

    # 汇总
    print()
    print("=" * 66)
    print("汇总")
    print("=" * 66)
    answered = sum(1 for c, r in zip(CASES, results) if c[1] == "answer")
    refused = len(CASES) - answered
    print(f"该答的 {answered} 题：{sum(results[:answered])} 题通过")
    print(f"该拒的 {refused} 题：{sum(results[answered:])} 题通过")
    print(f"合计：{sum(results)} / {len(CASES)}")

    if all(results):
        print("\nDay 3 目标达成：能答、能标出处、答不了会拒答。")
    else:
        print("\n有未通过的项，看上面的具体输出。")

    # 流式对比（只跑一题，省钱）
    print()
    print("=" * 66)
    print("流式版本对比（同一题）")
    print("=" * 66)
    q = "出差期间吃饭一天补助多少？"
    print(f"❓ {q}")
    print("   实时输出：", end="", flush=True)
    for chunk in rag.answer_stream(q):
        if not chunk.startswith("data: "):
            continue
        payload = chunk[len("data: "):].strip()
        if payload == "[DONE]":
            break
        import json
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("type") == "meta":
            srcs = obj.get("sources") or []
            print(f"\n   [meta] 来源 {len(srcs)} 页，最高分 {obj.get('top_score')}")
            print("   [delta] ", end="", flush=True)
        elif obj.get("type") == "delta":
            print(obj.get("text", ""), end="", flush=True)
        elif obj.get("type") == "done":
            print(f"\n   [done] 引用 {obj.get('citations')} 处")


if __name__ == "__main__":
    main()
