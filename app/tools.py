"""
工具集：给 Agent 用的"手"，让它能做检索之外的事。

为什么需要工具（而不是全部靠检索）：
    检索擅长"按意思找一段话"，但有两类问题它做不好——
    1. 结构化查询：金额落在哪个审批区间，检索会因为"三万"≠"20000~50000"而算错
    2. 业务数据查询：某张单据审到哪一步，制度文档里根本没有，得查数据库
    这类问题需要"精确地算/查"，而不是"模糊地匹配"。

本模块提供两个工具，加上 vectorstore.search 一共三个，平级关系：
    search_docs          —— 查制度文档（复用 vectorstore.search）
    query_reimbursement  —— 查报销单状态（本文件）
    query_approval       —— 查金额审批权限（本文件）
"""

# ---------- 报销单数据（PostgreSQL） ----------
# 真实数据存在 Docker 里的 PostgreSQL 中（见 init_db.py）。
# 这里只负责"查"，不负责"存"——数据的生命周期归数据库管。

import psycopg2

from app.config import DATABASE_URL


def _query_db(sql: str, params: tuple) -> list[dict]:
    """执行一条查询，返回字典列表。每次开短连接（简单场景够用，高频再上连接池）。"""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


# 审批权限表（来自财务报销管理制度 第十二条）
# 每档：金额下限(含) / 金额上限(不含) / 审批人
_APPROVAL_LEVELS = [
    (0, 5000, "部门负责人"),
    (5000, 20000, "部门负责人 + 财务负责人"),
    (20000, 50000, "部门负责人 + 财务负责人 + 分管副总"),
    (50000, float("inf"), "总经理审批"),
]


def query_reimbursement(order_id: str) -> dict:
    """
    按单号查报销单状态（数据源：PostgreSQL）。

    参数 order_id：单据号，如 BX2026091。
    返回：单据信息，或"未找到"的提示。
    """
    rows = _query_db(
        "SELECT id, applicant, amount, status, updated FROM reimbursements WHERE UPPER(id) = UPPER(%s)",
        (order_id,),
    )
    if rows:
        return {"found": True, **rows[0]}
    return {"found": False, "order_id": order_id, "message": "没有找到该单号"}


def query_approval(amount: float) -> dict:
    """
    查某个金额需要哪些人审批。

    参数 amount：报销金额（元）。
    返回：命中的审批档位和审批人。
    """
    for low, high, approver in _APPROVAL_LEVELS:
        if low <= amount < high:
            return {
                "amount": amount,
                "range": f"{low:,.0f} 至 {high:,.0f}" if high != float("inf") else f"{low:,.0f} 以上",
                "approver": approver,
            }
    # 理论上到不了这里（最后一档上限是无穷），防御性返回
    return {"amount": amount, "range": "未知", "approver": "无法确定"}


def calc_amount(expression: str) -> dict:
    """
    安全的金额计算。

    参数 expression：一个简单的算术表达式，如 "3200 + 500" 或 "15800 * 0.2"。
    用 AST 白名单解析，而不是 eval()，避免执行恶意代码。

    返回：计算结果，或错误说明。
    """
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            left = _eval(node.left)
            right = _eval(node.right)
            if type(node.op) is ast.Div and right == 0:
                raise ValueError("除数不能为 0")
            return allowed_ops[type(node.op)](left, right)
        raise ValueError("表达式里含有不支持的语法")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        # 金额保留两位小数
        if isinstance(result, float):
            result = round(result, 2)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}
