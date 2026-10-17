"""
入库流水线：一个文件从「磁盘上的 PDF」到「能检索的向量」的全过程。

    读取 → 分块 → 向量化 → 存进向量库

单独抽成一个模块，是因为后面 /rag/upload 接口要复用它，
测试脚本也要复用它 —— 同一套逻辑只写一遍。
"""
from pathlib import Path

from app import config, vectorstore
from app.chunking import build_chunks
from app.loader import list_documents, load_document


def ingest_file(path: str | Path) -> dict:
    """
    导入一个文件，返回统计信息。

    重复导入同一份文件是安全的：会先按 source 删掉旧块再写入，
    不会出现「同一段话在库里存了两遍」的情况。
    """
    path = Path(path)
    source = path.name

    pages = load_document(path)
    chunks = build_chunks(pages, source=source, chunk_size=config.CHUNK_SIZE)

    vectorstore.delete_source(source)
    added = vectorstore.add_chunks(chunks)

    return {
        "source": source,
        "pages": len(pages),
        "chars": sum(len(p["text"]) for p in pages),
        "chunks": added,
    }


def ingest_dir(data_dir: str | Path | None = None) -> list[dict]:
    """把 data/ 目录下的文件全部导入。"""
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    results = []
    for path in list_documents(data_dir):
        results.append(ingest_file(path))
    return results
