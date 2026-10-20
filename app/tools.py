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

# ---------- 模拟单据数据 ----------
# 真实项目里这应该查数据库（Day 5 接 PostgreSQL）。
# 现在先用内存里的字典模拟，把"工具"的形态先立起来，数据源之后再换。
_REIMBURSEMENTS = [
    {"id": "BX2026091", "applicant": "张三", "amount": 3200.0, "status": "部门负责人审批中", "updated": "2026-10-18"},
    {"id": "BX2026087", "applicant": "李四", "amount": 15800.0, "status": "财务复核中", "updated": "2026-10-17"},
    {"id": "BX2026083", "applicant": "王五", "amount": 68000.0, "status": "总经理审批中", "updated": "2026-10-16"},
    {"id": "BX2026079", "applicant": "赵六", "amount": 900.0, "status": "已完成付款", "updated": "2026-10-15"},
]

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
    按单号查报销单状态。

    参数 order_id：单据号，如 BX2026091。
    返回：单据信息，或"未找到"的提示。
    """
    for r in _REIMBURSEMENTS:
        if r["id"].upper() == order_id.upper():
            return {"found": True, **r}
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
