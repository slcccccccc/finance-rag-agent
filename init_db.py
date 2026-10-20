"""
初始化数据库：建表 + 灌入单据数据。

跑一次即可（幂等：重复跑会先清再灌）。
    ./.venv/Scripts/python.exe init_db.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2

from app.config import DATABASE_URL

SEED = [
    {"id": "BX2026091", "applicant": "张三", "amount": 3200.0, "status": "部门负责人审批中", "updated": "2026-10-18"},
    {"id": "BX2026087", "applicant": "李四", "amount": 15800.0, "status": "财务复核中", "updated": "2026-10-17"},
    {"id": "BX2026083", "applicant": "王五", "amount": 68000.0, "status": "总经理审批中", "updated": "2026-10-16"},
    {"id": "BX2026079", "applicant": "赵六", "amount": 900.0, "status": "已完成付款", "updated": "2026-10-15"},
]


def main() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reimbursements (
            id        TEXT PRIMARY KEY,
            applicant TEXT NOT NULL,
            amount    NUMERIC(12, 2) NOT NULL,
            status    TEXT NOT NULL,
            updated   DATE NOT NULL
        )
        """
    )
    # 幂等：清掉旧数据再灌
    cur.execute("DELETE FROM reimbursements")
    for r in SEED:
        cur.execute(
            "INSERT INTO reimbursements (id, applicant, amount, status, updated) VALUES (%s, %s, %s, %s, %s)",
            (r["id"], r["applicant"], r["amount"], r["status"], r["updated"]),
        )
    conn.commit()

    cur.execute("SELECT count(*) FROM reimbursements")
    print(f"建表完成，灌入 {cur.fetchone()[0]} 条单据数据")
    conn.close()


if __name__ == "__main__":
    main()
