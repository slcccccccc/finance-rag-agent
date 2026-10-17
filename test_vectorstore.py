"""
检索自检脚本：验证「用一句话能不能搜到那一块」。

这是 Day 2 的唯一验收标准：
    上传一份 PDF → 用一句自然语言提问 → 最相关的片段排在第一位。

跑法：
    ./.venv/Scripts/python.exe test_vectorstore.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app import vectorstore
from app.ingest import ingest_file

PDF = "data/财务报销管理制度.pdf"

# (问题, 正确块里独有、别处不会出现的字样)
# 判据必须「只出现在正确那一块」，否则别的块碰巧也有同样的词，就会造成假通过。
# 最后一条是故意问无关问题，用来看分数会不会掉下去。
CASES = [
    ("出差住酒店一晚最多能报多少钱？", "500"),
    ("报销单最晚多少天内必须提交？", "15 个工作日"),
    ("花三万块钱要经过哪些人审批？", "20,000 至 50,000"),
    ("出差期间吃饭一天补助多少？", "100 元"),
    ("机票改签的费用能报销吗？", "改签费"),
    ("今天广州天气怎么样？", None),          # None 表示：这个问题答不出来才对
]


def main() -> None:
    print("=" * 70)
    print("步骤 1｜清空旧库并重新导入 PDF")
    print("=" * 70)

    vectorstore.reset()
    info = ingest_file(PDF)
    print(f"文件      : {info['source']}")
    print(f"页数      : {info['pages']} 页，共 {info['chars']} 字")
    print(f"入库块数  : {info['chunks']} 块")
    print(f"库内总数  : {vectorstore.count()} 块")
    print()

    print("=" * 70)
    print("步骤 2｜逐条检索（每题看 Top-3）")
    print("=" * 70)

    passed_top1 = 0
    passed_top3 = 0
    tested = 0

    for question, expect in CASES:
        hits = vectorstore.search(question, top_k=3)
        print(f"\n❓ {question}")

        if not hits:
            print("   ⚠️ 一条都没检索到")
            continue

        for i, h in enumerate(hits, 1):
            mark = "★" if i == 1 else " "
            preview = h["text"].replace("\n", " ")[:58]
            print(f"  {mark} #{i} 相似度 {h['score']:.4f} | 第 {h['page']} 页 | {preview}…")

        if expect is None:
            top_score = hits[0]["score"]
            if top_score < 0.45:
                print(f"  ✅ 无关问题得分低（{top_score:.4f}），后面可以据此拒答")
            else:
                print(f"  ⚠️ 无关问题得分偏高（{top_score:.4f}），拒答阈值要往上调")
            continue

        tested += 1
        top1_ok = expect in hits[0]["text"]
        top3_ok = any(expect in h["text"] for h in hits[:3])
        if top3_ok:
            passed_top3 += 1
        if top1_ok:
            passed_top1 += 1

        if top1_ok:
            print(f"  ✅ 第 1 位就是正确块（含「{expect}」）")
        elif top3_ok:
            print(f"  ⚠️ 正确块只排进前三，Top-1 是错的（「{expect}」在后面的块里）")
        else:
            print(f"  ❌ Top-3 都没找到「{expect}」")

    print()
    print("=" * 70)
    print("步骤 3｜结果")
    print("=" * 70)
    print(f"有标准答案的题目 : {tested} 题")
    if tested:
        print(f"Top-1 是正确块   : {passed_top1} 题（{passed_top1 / tested:.0%}）")
        print(f"Top-3 含正确块   : {passed_top3} 题（{passed_top3 / tested:.0%}）")
    print()
    print(
        """
两个指标的意义不一样，别混着看：
  · Top-1 准确  → 直接决定回答质量。Top-1 错了，模型就会拿着无关条款去编。
  · Top-3 命中  → 只是「还有救」。后面接上重排（rerank）才能把正确的顶上来。

如果 Top-1 不准，先别改代码，先看搜出来的是什么：
  · 相邻条款（差一条）→ 语义本来就接近，属正常损耗，可考虑块间带上下文
  · 完全不搭边         → 检查这一块的正文是否被 PDF 解析搞乱了
  · 分数普遍很低（<0.4）→ embedding 模型没加载对，或查询前缀没生效
"""
    )


if __name__ == "__main__":
    main()
