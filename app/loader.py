"""
文档加载：把文件读成「按页的纯文本」。

为什么保留页码：
    后面做引用溯源要用。用户问「住宿标准多少」，答案如果能带上「第二页第四条」，
    可信度完全不一样。所以从第一步就要把页码记住，不能只留一大坨文本。
"""
from pathlib import Path

import pymupdf


def load_pdf_pages(path) -> list[dict]:
    """
    解析 PDF，逐页提取文本。

    返回：[{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]
    """
    path = Path(path)
    pages = []

    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text()
            # 去掉首尾空白，但保留内部换行（分块时要用段落结构）
            text = text.strip()
            if text:
                pages.append({"page": index, "text": text})

    return pages


def load_text_file(path) -> list[dict]:
    """读取 Markdown / 纯文本，整体当作一页。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    return [{"page": 1, "text": text}] if text else []


def load_document(path) -> list[dict]:
    """按扩展名选择解析方式。"""
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return load_pdf_pages(path)
    if suffix in (".md", ".markdown", ".txt"):
        return load_text_file(path)

    raise ValueError(f"暂不支持的文件类型：{suffix}（目前支持 .pdf / .md / .txt）")


def list_documents(data_dir) -> list[Path]:
    """列出待处理目录下的所有支持文件。"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    exts = {".pdf", ".md", ".markdown", ".txt"}
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in exts)
