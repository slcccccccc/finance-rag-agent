"""
手动测试接口的小工具。

用法（在项目根目录，终端里先激活虚拟环境）：
    python test_api.py

为什么需要它：
    Git Bash 里用 curl 传中文时，参数会被转成本地编码（GBK），
    服务器收到后 JSON 解析失败。用 Python 发请求就没有这个问题。
"""
import sys

import requests

# Windows 终端默认不是 UTF-8，先改掉，否则中文输出会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """检查服务是否活着、密钥有没有加载。"""
    print("=" * 50)
    print("[1] 测试 /health")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
    except requests.exceptions.ConnectionError:
        print("    ✗ 连不上服务，先确认 uvicorn 那个窗口还开着")
        return False

    data = r.json()
    print(f"    状态码: {r.status_code}")
    print(f"    返回: {data}")

    if not data.get("api_key_loaded"):
        print("    ✗ 密钥没加载，检查 .env 里的 DEEPSEEK_API_KEY")
        return False

    print("    ✓ 服务正常")
    return True


def test_chat(message: str):
    """测试普通对话接口。"""
    print("=" * 50)
    print("[2] 测试 /chat（普通返回）")
    r = requests.post(f"{BASE_URL}/chat", json={"message": message}, timeout=60)
    if r.status_code != 200:
        print(f"    ✗ 失败: {r.status_code} {r.text}")
        return False
    data = r.json()
    print(f"    回答: {data.get('answer')}")
    print(f"    用量: {data.get('usage')}")
    print("    ✓ 普通对话正常")
    return True


def test_stream(message: str):
    """测试流式接口，观察文字是不是一段一段回来的。"""
    print("=" * 50)
    print("[3] 测试 /chat/stream（流式返回）")
    print("    下面是模型逐段吐出的内容，注意看是不是一点点出现的：\n")

    with requests.post(
        f"{BASE_URL}/chat/stream",
        json={"message": message},
        stream=True,
        timeout=120,
    ) as r:
        if r.status_code != 200:
            print(f"    ✗ 失败: {r.status_code} {r.text[:300]}")
            return False

        chunk_count = 0
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    print("\n    --- 收到结束标记 [DONE] ---")
                    break
                try:
                    import json

                    obj = json.loads(payload)
                    if "delta" in obj:
                        chunk_count += 1
                        print(obj["delta"], end="", flush=True)
                    elif "error" in obj:
                        print(f"\n    ✗ 服务端报错: {obj['error']}")
                        return False
                except json.JSONDecodeError:
                    print(f"\n    ! 无法解析的一行: {payload}")

    print(f"\n\n    共收到 {chunk_count} 段增量内容")
    if chunk_count > 1:
        print("    ✓ 流式正常（分多段返回）")
    elif chunk_count == 1:
        print("    ! 只有 1 段，可能是模型回答太短，换长一点的问题再试")
    else:
        print("    ✗ 没有收到内容")
    return chunk_count > 0


def main():
    print("接口连通性测试，共三项\n")

    if not test_health():
        print("\n服务不可用，后面两项跳过。")
        return

    question = "用三句话介绍一下差旅报销的常见流程"
    test_chat(question)
    test_stream(question)

    print("\n" + "=" * 50)
    print("测试结束。")


if __name__ == "__main__":
    main()
