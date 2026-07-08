"""
6-B：API 冒烟测试 —— 服务起来后跑一遍验证接口可用

用法（另开终端，确保 uvicorn 已启动）：
  python serve/test_api.py
"""

import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    # 1. 健康检查
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        print("健康检查:", r.read().decode("utf-8"))

    # 2. 生成一篇（短一点，快速验证）
    print("\n生成中...")
    resp = post("/generate", {
        "topic": "AI 编程助手会让初级程序员消失吗",
        "requirements": "1500字左右，观点鲜明，先输出<thinking>再输出<article>",
        "max_new_tokens": 2048,
    })
    print("=" * 50)
    print(resp["article"][:1000])
    print("=" * 50)
    print(f"正文字数(含标点): {len(resp['article'])}")


if __name__ == "__main__":
    main()
