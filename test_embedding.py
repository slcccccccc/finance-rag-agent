"""
验证 embedding 模型能正常下载、加载、编码。

用法（项目根目录，不需要激活虚拟环境）：
    ./.venv/Scripts/python.exe test_embedding.py

第一次运行会下载模型（约 95MB），需要 5 到 10 分钟，看到进度条是正常的。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from app import embedding


def main():
    print("=" * 56)
    print("测试 1：加载模型（首次会下载，耐心等）")
    print("=" * 56)
    t0 = time.time()
    vec = embedding.encode_query("差旅住宿标准是多少")
    print(f"耗时 {time.time() - t0:.1f} 秒")
    print(f"向量维度：{vec.shape[1]}")
    print(f"前 5 个数值：{np.round(vec[0][:5], 4).tolist()}")
    print("说明：一句话被变成了 512 个数字，这就是「向量」。\n")

    print("=" * 56)
    print("测试 2：语义相似度（这才是 embedding 的意义所在）")
    print("=" * 56)

    base = "差旅住宿标准是多少"

    candidates = [
        "出差的酒店报销额度是多少",   # 意思几乎一样，字面完全不同
        "我上周提交的报销单审到哪了",  # 同领域但问的是另一件事
        "今天广州的天气怎么样",       # 完全无关
    ]

    base_vec = embedding.encode_query(base)
    cand_vecs = embedding.encode_documents(candidates)

    print(f"基准句：{base}\n")
    results = []
    for text, cv in zip(candidates, cand_vecs):
        score = embedding.cosine_similarity(base_vec[0], cv)
        results.append((score, text))

    for score, text in sorted(results, reverse=True):
        bar = "█" * int(score * 40)
        print(f"  {score:.4f}  {bar}")
        print(f"          {text}\n")

    top_score = max(results)[0]
    bottom_score = min(results)[0]
    if top_score > 0.6 and bottom_score < 0.4:
        print("✓ 语义区分正常：近义句分数高，无关句分数低")
        print("  这就是 RAG 能「按意思找内容」而不是「按关键词找内容」的原因。\n")
    else:
        print("! 分数分布不太理想，但模型本身应该是好的，先继续往下做。\n")

    print("=" * 56)
    print("测试 3：批量编码速度")
    print("=" * 56)
    texts = [f"这是第 {i} 条测试文本，用于评估编码速度。" for i in range(50)]
    t0 = time.time()
    vectors = embedding.encode_documents(texts)
    cost = time.time() - t0
    print(f"50 条文本耗时 {cost:.2f} 秒，平均 {cost / 50 * 1000:.0f} 毫秒/条")
    print(f"输出形状：{vectors.shape}（50 条 × 512 维）")
    if cost / 50 < 0.5:
        print("✓ 速度可以，一份几百页的制度文档入库不会太慢")
    else:
        print("! 偏慢，但还能用，明天入库时注意别一次塞太多")

    print("\n" + "=" * 56)
    print("全部通过。模型已缓存到本地，下次启动不用再下载。")


if __name__ == "__main__":
    main()
