"""
向量库：把分好块的文本存起来，之后能用「一句话」按意思搜出来。

为什么不用数据库的 LIKE 模糊查询：
    LIKE 只能匹配字面。用户问「住宿能报多少钱」，
    制度里写的是「住宿费标准按出差目的地城市类别执行」——
    一个字都对不上，LIKE 查不出来。向量检索比的是「意思离多远」，
    所以能捞到。

为什么用 Chroma：
    - 自带本地持久化（一个文件夹就是整个库），不用装数据库服务
    - 支持 metadata 过滤、按 id 删除
    - Day 5 会把它换成 PostgreSQL + pgvector，
      换的时候只改这一个文件，上层代码不用动 —— 这就是分层的好处

一个必须理解的概念：距离 vs 相似度
    Chroma 返回的是「距离」，越小越像；我们对外想给的是「相似度」，越大越像。
    建库时声明了 cosine 距离，且入库向量已归一化，所以：
        相似度 = 1 - 距离
"""
from pathlib import Path

import chromadb

from app import config
from app.embedding import encode_documents, encode_query

COLLECTION_NAME = "finance_docs"

_client = None


def get_client() -> chromadb.PersistentClient:
    """懒加载持久化客户端：数据落在 chroma_db/ 目录里，重启不丢。"""
    global _client
    if _client is None:
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection(create: bool = True):
    """
    拿到（或创建）集合。

    metadata 里声明 hnsw:space=cosine 很关键：
    文本向量的语义比的是「方向」而不是「长度」，用余弦距离才正确。
    默认的 l2（欧氏距离）在高维文本向量上效果会差一些。
    """
    client = get_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)


def count() -> int:
    """库里现有多少块。"""
    try:
        return get_collection(create=False).count()
    except Exception:
        return 0


def add_chunks(chunks: list) -> int:
    """
    把 Chunk 列表写进向量库。

    这里的三个关键动作：
      1. 文本 → 向量（用我们自己的 embedding，不用 Chroma 默认的英文模型）
      2. 原文和页码存进 metadata，检索出来才能做引用溯源
      3. 用 chunk.id 当主键，重复导入同一份文件不会产生重复块
    """
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    vectors = encode_documents(texts)

    collection = get_collection()
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[
            {
                "page": c.page,
                "section": c.section or "",
                "source": c.source,
                "order": c.order,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def delete_source(source: str) -> None:
    """删掉某一份文件的所有块（重新上传前先清旧数据，避免新旧混在一起）。"""
    try:
        get_collection(create=False).delete(where={"source": source})
    except Exception:
        pass


def search(query: str, top_k: int | None = None, source: str | None = None) -> list[dict]:
    """
    按意思检索，返回最相关的若干块。

    参数：
        query   —— 用户的问题
        top_k   —— 取前几条，默认用 .env 里的 TOP_K
        source  —— 只在某个文件内检索（可选）

    返回：
        [{"text", "page", "section", "source", "score"}, ...]，按相似度降序
    """
    k = top_k or config.TOP_K
    collection = get_collection(create=False)

    if collection.count() == 0:
        return []

    query_vector = encode_query(query)

    result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=min(k, collection.count()),
        where={"source": source} if source else None,
    )

    # Chroma 返回的是「一批查询」的结果，我们只传了一个问题，所以取 [0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    hits = []
    for text, meta, distance in zip(docs, metas, distances):
        hits.append(
            {
                "text": text,
                "page": meta.get("page"),
                "section": meta.get("section", ""),
                "source": meta.get("source", ""),
                # 余弦距离转相似度
                "score": round(1.0 - float(distance), 4),
            }
        )
    return hits


def reset() -> None:
    """清空整个集合（测试用；重新灌数据时也方便）。"""
    try:
        get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
