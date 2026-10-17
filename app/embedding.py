"""
Embedding 封装：把文字变成向量。

为什么需要这一步：
    计算机不认识字，只认识数字。Embedding 就是把一句话压成一串数字（向量），
    而且意思相近的句子，算出来的向量在空间里也靠得更近。
    RAG 的全部基础就建立在这件事上——用"距离"来找语义相关的内容，
    而不是靠关键词匹配。

模型选择：BAAI/bge-small-zh-v1.5
    - 中文效果在 small 这一档里名列前茅
    - 只有约 95MB，纯 CPU 就能跑
    - 输出 512 维向量
    - 用 bge 系列时，查询要加一句固定前缀，检索效果会更好（官方建议）
"""
import os

# 必须在 import sentence_transformers 之前设置镜像，否则下载会走 huggingface 官方站
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# bge 系列的官方建议：检索时给 query 加上这句前缀
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_model = None


def get_model() -> SentenceTransformer:
    """懒加载：第一次调用时才下载并加载模型，避免启动服务时就卡住。"""
    global _model
    if _model is None:
        print(f"[embedding] 加载模型 {MODEL_NAME}（首次会下载，约 95MB）...")
        _model = SentenceTransformer(MODEL_NAME)
        # sentence-transformers 5.x 把方法改名了，这里兼容新旧两种写法
        if hasattr(_model, "get_embedding_dimension"):
            dim = _model.get_embedding_dimension()
        else:
            dim = _model.get_sentence_embedding_dimension()
        print(f"[embedding] 加载完成，向量维度 = {dim}")
    return _model


def encode(texts, normalize: bool = True) -> np.ndarray:
    """
    把一批文本编码成向量矩阵。

    参数：
        texts     —— 单个字符串，或字符串列表
        normalize —— 是否归一化。归一化后向量长度为 1，
                     此时余弦相似度直接等于点积，计算更快也更稳定。

    返回：
        shape 为 (n, dim) 的 numpy 数组
    """
    if isinstance(texts, str):
        texts = [texts]

    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    )
    return vectors


def encode_documents(texts) -> np.ndarray:
    """编码文档片段（入库时用，不加前缀）。"""
    return encode(texts)


def encode_query(text: str) -> np.ndarray:
    """编码用户问题（检索时用，加 bge 建议的前缀）。"""
    return encode([QUERY_INSTRUCTION + text])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度：值域 -1 到 1，越接近 1 越相似。"""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
