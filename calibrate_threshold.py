"""
阈值校准：用数据决定「相似度低于多少就拒答」。

为什么不能拍脑袋定阈值：
    定太高 → 明明资料里有答案，系统却说"不知道"，白丢可用性
    定太低 → 资料里没有的东西也硬答，那就是幻觉
    唯一的办法是把「该答的」和「该拒的」两类问题都跑一遍，
    看两组分数之间有没有一条干净的缝。有缝，阈值就放在缝里。

这个脚本只测检索、不调大模型，不花钱。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app import config, vectorstore

# 该答的（资料里明确有依据）
RELEVANT = [
    "出差住酒店一晚最多能报多少钱",
    "报销单最晚多少天内必须提交",
    "花三万块钱要经过哪些人审批",
    "出差期间吃饭一天补助多少",
    "机票改签的费用能报销吗",
    "电子发票重复报销会怎么处理",
    "出差申请要在什么时候提交",
    "经理出差能坐高铁一等座吗",
]

# 该拒的（制度里根本没有）
IRRELEVANT = [
    "今天广州天气怎么样",
    "公司的股票代码是多少",
    "帮我写一首关于春天的诗",
    "年会什么时候举办",
    "员工食堂几点开饭",
    "如何申请年假",
]


def top_scores(questions: list[str]) -> list[tuple[str, float]]:
    out = []
    for q in questions:
        hits = vectorstore.search(q, top_k=1)
        score = hits[0]["score"] if hits else 0.0
        out.append((q, score))
    return out


def main() -> None:
    if vectorstore.count() == 0:
        print("向量库是空的，先跑 test_vectorstore.py 灌数据")
        return

    print("=" * 66)
    print(f"库内块数：{vectorstore.count()}   当前阈值：{config.SCORE_THRESHOLD}")
    print("=" * 66)

    print("\n【该答的】问题及其最高相似度")
    rel = sorted(top_scores(RELEVANT), key=lambda x: x[1])
    for q, s in rel:
        print(f"  {s:.4f}  {q}")

    print("\n【该拒的】问题及其最高相似度")
    irr = sorted(top_scores(IRRELEVANT), key=lambda x: x[1], reverse=True)
    for q, s in irr:
        print(f"  {s:.4f}  {q}")

    lowest_rel = rel[0][1] if rel else 1.0
    highest_irr = irr[0][1] if irr else 0.0

    print()
    print("=" * 66)
    print("结论")
    print("=" * 66)
    print(f"  该答的里面最低分 : {lowest_rel:.4f}")
    print(f"  该拒的里面最高分 : {highest_irr:.4f}")
    print(f"  两者之间的缝隙   : {highest_irr:.4f} ~ {lowest_rel:.4f}")

    if lowest_rel > highest_irr:
        mid = (lowest_rel + highest_irr) / 2
        print(f"\n  ✅ 两类问题能分开，阈值建议放在 {mid:.2f} 左右")
        print(f"     规律：高于 {lowest_rel:.2f} 一定该答，低于 {highest_irr:.2f} 一定该拒")
        print(f"     当前 .env 里的 {config.SCORE_THRESHOLD} "
              f"{'落在缝隙内，可用' if highest_irr <= config.SCORE_THRESHOLD <= lowest_rel else '不在缝隙内，需要调整'}")
    else:
        print("\n  ⚠️ 两类问题的分数有重叠，单靠阈值分不干净。")
        print("     这时候要么改进检索质量，要么再加一层判断（例如让模型先判断问题是否与资料相关）。")

    print()
    print("面试可以直接讲这个：")
    print("  「我没有拍脑袋设阈值，而是把 8 个该答的问题和 6 个该拒的问题各跑一遍，")
    print("    看两组最高分之间有没有缝，阈值就取缝的中间。」")


if __name__ == "__main__":
    main()
