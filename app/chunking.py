"""
分块：把整篇文档切成一段一段，供向量化与检索。

为什么分块质量决定 RAG 的上限：
    检索时是拿「块」去匹配问题的。切得太碎，一句话被腰斩，模型看不懂；
    切得太大，一块里塞了三四个主题，检索出来一半内容与问题无关，还会挤占上下文窗口。
    所以分块不是技术细节，是 RAG 效果的第一道门槛。

本项目的策略（针对中文制度类文档优化）：
    1. 优先按「第X章 / 第X条」切 —— 一条制度本身就是最小的完整语义单元
    2. 某个条款太长（超过阈值 1.6 倍）时，再按句子切开
    3. 太短的碎片与相邻块合并，避免出现只有十几个字的无效块
    4. 每个块记住它属于第几页、属于哪一章，供引用溯源使用
"""
import re
from dataclasses import dataclass, asdict

# 「第X章」「第X条」，中文数字
_CHAPTER_RE = re.compile(r"第[一二三四五六七八九十百]+章[^\n]{0,30}")
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百]+条")

# 中文句子边界
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])")


@dataclass
class Chunk:
    """一个可检索的文本块。"""
    id: str
    text: str
    page: int
    section: str
    source: str
    order: int

    def to_dict(self) -> dict:
        return asdict(self)


def _build_full_text(pages: list[dict]):
    """
    把多页文本拼成一整篇，同时记录每一页在全文中的字符区间。

    返回 (full_text, page_ranges)
    page_ranges: [(start, end, page_no), ...]
    """
    full_text = ""
    page_ranges = []

    for item in pages:
        start = len(full_text)
        full_text += item["text"] + "\n"
        page_ranges.append((start, len(full_text), item["page"]))

    return full_text, page_ranges


def _page_at(pos: int, page_ranges: list) -> int:
    """反查某个字符位置属于第几页。"""
    for start, end, page_no in page_ranges:
        if start <= pos < end:
            return page_no
    return page_ranges[-1][2] if page_ranges else 1


def _slice_by_marks(full_text: str):
    """
    按章节 / 条款的标题位置切段。

    返回 [(start, end, section_title), ...]
    """
    marks = []
    for m in _CHAPTER_RE.finditer(full_text):
        marks.append((m.start(), "chapter", m.group().strip()))
    for m in _ARTICLE_RE.finditer(full_text):
        marks.append((m.start(), "article", m.group().strip()))

    if not marks:
        # 没有章条结构，整篇当一段
        return [(0, len(full_text), "")]

    marks.sort(key=lambda x: x[0])

    segments = []
    current_chapter = ""

    # 第一个标记之前的内容（标题、文件编号等）
    if marks[0][0] > 0:
        head = full_text[: marks[0][0]].strip()
        if len(head) > 20:
            segments.append((0, marks[0][0], "文档标题"))

    for i, (pos, kind, title) in enumerate(marks):
        if kind == "chapter":
            current_chapter = title

        end = marks[i + 1][0] if i + 1 < len(marks) else len(full_text)
        segment = full_text[pos:end].strip()
        if not segment:
            continue

        segments.append((pos, end, current_chapter or title))

    return segments


def _split_long_text(text: str, chunk_size: int, overlap_sentences: int = 1) -> list[str]:
    """
    把过长的段落按句子切开，每块不超过 chunk_size。

    overlap_sentences：相邻块之间重复上一块末尾的句子数，
    让边界处的内容不至于被完全切断。
    """
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    pieces = []
    current = []

    for sent in sentences:
        candidate = "".join(current + [sent])
        if current and len(candidate) > chunk_size:
            pieces.append("".join(current))
            # 保留末尾几句作为下一块的开头
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)

    if current:
        pieces.append("".join(current))

    return [p.strip() for p in pieces if p.strip()]


def build_chunks(
    pages: list[dict],
    source: str,
    chunk_size: int = 500,
    overlap_sentences: int = 1,
    min_size: int = 60,
) -> list[Chunk]:
    """
    把「按页的文本」切成 Chunk 列表。

    参数：
        pages        —— loader 返回的 [{"page": n, "text": "..."}]
        source       —— 来源文件名，用于溯源
        chunk_size   —— 单块目标字数
        min_size     —— 小于这个长度的碎片会被并入下一块
    """
    if not pages:
        return []

    full_text, page_ranges = _build_full_text(pages)
    segments = _slice_by_marks(full_text)

    chunks: list[Chunk] = []

    def emit(text: str, start: int, section: str) -> None:
        """生成一个块，自动编号、自动反查页码。"""
        chunks.append(
            Chunk(
                id=f"{source}-{len(chunks):03d}",
                text=text.strip(),
                page=_page_at(start, page_ranges),
                section=section,
                source=source,
                order=len(chunks),
            )
        )

    # 碎片缓冲区：攒得太短的条款，等下一段来了再决定合并还是独立成块
    buffer = ""
    buffer_start = 0
    buffer_section = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            emit(buffer, buffer_start, buffer_section)
        buffer = ""

    pending_heading = ""   # 纯章节标题，不单独成块，挂到下一段前面

    for start, end, section in segments:
        raw = full_text[start:end].strip()
        if not raw:
            continue

        # ① 只由「第X章 标题」构成的片段：记下来，挂到下一段，不占一个块
        if _CHAPTER_RE.fullmatch(raw):
            pending_heading = raw
            continue

        # ② 跨章了：上一章残留的碎片先结掉，不能和新的章混在一起
        if buffer and buffer_section and section != buffer_section:
            flush()

        # ③ 把待挂的章节标题拼到正文前面，让这一块自带上下文
        text = f"{pending_heading}\n{raw}".strip() if pending_heading else raw
        pending_heading = ""

        # ④ 还能塞进现有碎片：合并，避免产生十几字的废块
        if (
            len(text) < min_size
            and buffer
            and buffer_section == section
            and len(buffer) + len(text) <= chunk_size
        ):
            buffer = f"{buffer}\n{text}"
            continue

        # ⑤ 太长：按句子拆成多块，块间留一句重叠
        if len(text) > chunk_size * 1.6:
            flush()
            for piece in _split_long_text(text, chunk_size, overlap_sentences):
                emit(piece, start, section)
            continue

        # ⑥ 太短又没有可合并的对象：先攒着，看下一段能不能拼
        if len(text) < min_size:
            flush()
            buffer = text
            buffer_start = start
            buffer_section = section
            continue

        flush()
        emit(text, start, section)

    flush()
    return chunks
