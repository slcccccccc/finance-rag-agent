"""
分块自检脚本。

分块是 RAG 的第一道门槛：切坏了，后面向量化、检索、生成全都白搭。
这个脚本不测"代码有没有报错"，而是把切出来的每一块**打印出来给你看**，
由你判断：这一块单独拿出来，人能不能看懂、够不够回答问题？

跑法：
    ./.venv/Scripts/python.exe test_chunking.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.config import CHUNK_SIZE
from app.loader import load_document
from app.chunking import build_chunks

PDF = Path("data/财务报销管理制度.pdf")


def main() -> None:
    print("=" * 70)
    print("步骤 1｜读 PDF")
    print("=" * 70)

    pages = load_document(PDF)
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"文件      : {PDF.name}")
    print(f"页数      : {len(pages)} 页")
    print(f"总字数    : {total_chars} 字")
    print()

    print("=" * 70)
    print(f"步骤 2｜分块（目标块大小 {CHUNK_SIZE} 字）")
    print("=" * 70)

    chunks = build_chunks(pages, source=PDF.name, chunk_size=CHUNK_SIZE)

    if not chunks:
        print("❌ 一块都没切出来，检查 loader 是否读到了文本")
        return

    for c in chunks:
        head = c.text.replace("\n", " ")[:70]
        tail = "" if len(c.text) <= 70 else " …"
        print(f"\n[{c.id}]  第 {c.page} 页  |  {len(c.text):>3} 字  |  {c.section}")
        print(f"      {head}{tail}")

    # ---------- 体检 ----------
    print()
    print("=" * 70)
    print("步骤 3｜体检报告")
    print("=" * 70)

    lengths = [len(c.text) for c in chunks]
    avg = sum(lengths) / len(lengths)

    print(f"块总数        : {len(chunks)}")
    print(f"平均块长      : {avg:.0f} 字")
    print(f"最短 / 最长   : {min(lengths)} / {max(lengths)} 字")
    print()

    # 三个常见病
    problems = []

    too_short = [c for c in chunks if len(c.text) < 60]
    if too_short:
        problems.append(
            f"⚠️  {len(too_short)} 块不足 60 字，可能是标题残留或断句碎片："
            + ", ".join(c.id for c in too_short[:5])
        )

    too_long = [c for c in chunks if len(c.text) > CHUNK_SIZE * 1.6]
    if too_long:
        problems.append(
            f"⚠️  {len(too_long)} 块超过 {int(CHUNK_SIZE * 1.6)} 字，说明拆分没生效："
            + ", ".join(c.id for c in too_long[:5])
        )

    no_section = [c for c in chunks if not c.section]
    if no_section:
        problems.append(f"⚠️  {len(no_section)} 块没有归属章节，溯源时会显示空白")

    if problems:
        for p in problems:
            print(p)
    else:
        print("✅ 没有发现明显问题：无碎片块、无超长块、每块都有章节归属")

    print()
    print("=" * 70)
    print("步骤 4｜人工抽检题")
    print("=" * 70)
    print(
        """
请你凭直觉回答下面三个问题，答案只能来自上面打印的某一块：

  1. 一线城市住宿一晚能报多少？
  2. 三万块的报销要走谁审批？
  3. 报销单最晚多少天内必须提交？

如果你能在某一整块里直接读到答案 → 分块合格。
如果答案被切断在两块之间、或者那块里还混着三个无关话题 → 分块不合格，
告诉我具体是哪一块，我调参数重切。
"""
    )


if __name__ == "__main__":
    main()
