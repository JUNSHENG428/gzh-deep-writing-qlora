"""
命令行写作客户端 —— 调用 Docker 服务生成公众号文章

用法示例：
  # 最简：只给主题
  python serve/write.py "为什么英伟达的护城河是CUDA"

  # 完整参数
  python serve/write.py "AI Agent会取代App吗" ^
      --audience "科技从业者" ^
      --requirements "2500字，观点鲜明，至少1个反直觉判断" ^
      --max 3072 --temp 0.7 ^
      --save output/my_article.md

  # 流式输出（打字机效果）
  python serve/write.py "县城青年的出路" --stream
"""

import argparse
import json
import urllib.request

BASE = "http://localhost:8010"   # 对应 docker-compose 映射的宿主机端口


def generate(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def generate_stream(payload: dict):
    req = urllib.request.Request(
        BASE + "/generate/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                print(chunk, end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser(description="公众号写作模型命令行客户端")
    ap.add_argument("topic", help="文章主题")
    ap.add_argument("--audience", default="关注科技的互联网从业者")
    ap.add_argument("--requirements", default="2000字左右，观点鲜明，有论据，避免营销话术")
    ap.add_argument("--system", default=None, help="自定义系统提示（换写作角色）")
    ap.add_argument("--max", type=int, default=3072, dest="max_new_tokens")
    ap.add_argument("--temp", type=float, default=0.7, dest="temperature")
    ap.add_argument("--rep", type=float, default=1.15, dest="repetition_penalty",
                    help="重复惩罚，越大越不易复读（1.1-1.3）")
    ap.add_argument("--stream", action="store_true", help="流式输出")
    ap.add_argument("--save", default=None, help="保存正文到文件")
    args = ap.parse_args()

    payload = {
        "topic": args.topic,
        "audience": args.audience,
        "requirements": args.requirements,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.system:
        payload["system"] = args.system

    if args.stream:
        print("=" * 50)
        generate_stream(payload)
        return

    print("生成中（长文约 1-3 分钟）...\n")
    resp = generate(payload)
    article = resp["article"]
    print("=" * 50)
    print(article)
    print("=" * 50)
    print(f"正文长度: {len(article)} 字符")

    if args.save:
        from pathlib import Path
        Path(args.save).write_text(article, encoding="utf-8")
        print(f"已保存: {args.save}")


if __name__ == "__main__":
    main()
