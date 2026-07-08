"""
Step 2-E-2：文章质量打分（DeepSeek flash 批量质检）

原理：
  微调学的是风格上限，「干净但平庸」的文章（新闻通稿/软文/拼盘文）
  清洗脚本救不了，需要模型级判断。用最便宜的 deepseek-v4-flash
  给每篇打分 + 贴标签，后续按分数线筛出精选集。

  只发文章前 3000 字：判断深度不需要读完全文，省一半 token。

评分维度（教师模型输出 JSON）：
  depth        1-5  思想深度：有无独立观点、反直觉判断、逻辑论证
  evidence     1-5  论据质量：具体数据/案例密度
  genre        深度评论 / 新闻报道 / 访谈 / 软文广告 / 拼盘杂烩
  topic        科技 / 商业 / 社会 / 消费 / 财经 / 其他

断点续跑：结果逐条追加到 data/generated/article_scores.jsonl，
重跑自动跳过已打分的文章。

用法：
  python scripts/score_articles.py            # 全部
  python scripts/score_articles.py --limit 10 # 试跑
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import chat

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
SCORES_FILE = ROOT / "data" / "generated" / "article_scores.jsonl"

# 打分模型：优先读 .env 的 SCORE_MODEL；未配置则用 DEEPSEEK_MODEL 默认值
# （阿里云网关用 deepseek-v4-flash；DeepSeek 官方 API 用 deepseek-chat）
import os
SCORE_MODEL = os.getenv("SCORE_MODEL") or None
MAX_CHARS = 3000                    # 只发前 3000 字
WORKERS = 6                         # 并发数（网关限流内）

SYSTEM_PROMPT = """你是一位严格的内容质量评审。用户给你一篇公众号文章（可能被截断），请评估它作为「深度写作训练数据」的价值。

评分标准：
- depth（思想深度 1-5）：5=有独到观点和反直觉判断，论证严密；3=有观点但平淡；1=纯信息罗列/通稿
- evidence（论据质量 1-5）：5=具体数据、案例、引用密集且可信；3=有一些；1=空泛
- genre（文体）：深度评论/新闻报道/访谈/软文广告/拼盘杂烩 五选一
- topic（题材）：科技/商业/社会/消费/财经/其他 六选一

只输出一行 JSON，不要其他任何文字：
{"depth": 4, "evidence": 3, "genre": "深度评论", "topic": "科技"}"""

_write_lock = threading.Lock()


def load_done() -> set[str]:
    """已打分的文件名集合"""
    if not SCORES_FILE.exists():
        return set()
    done = set()
    for line in SCORES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["file"])
    return done


def parse_score(raw: str) -> dict:
    """从响应中提取 JSON（容忍代码块包裹等杂质）"""
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"无 JSON: {raw[:80]}")
    data = json.loads(m.group(0))
    for key in ("depth", "evidence", "genre", "topic"):
        if key not in data:
            raise ValueError(f"缺字段 {key}")
    data["depth"] = int(data["depth"])
    data["evidence"] = int(data["evidence"])
    return data


def score_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")[:MAX_CHARS]
    raw = chat(
        system=SYSTEM_PROMPT,
        user=f"请评估以下文章：\n\n{text}",
        provider="deepseek",
        model=SCORE_MODEL,
        temperature=0.1,   # 打分要稳定，温度压到最低
        max_tokens=1024,   # 思考型模型会先消耗内部 token，给足余量防空响应
    )
    data = parse_score(raw)
    data["file"] = path.name
    n_cn = sum(1 for c in path.read_text(encoding="utf-8") if "\u4e00" <= c <= "\u9fff")
    data["chars"] = n_cn
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="打分篇数，0=全部")
    args = parser.parse_args()

    files = sorted(CLEANED_DIR.glob("*.md"))
    done = load_done()
    todo = [f for f in files if f.name not in done]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"打分模型: {SCORE_MODEL}（并发 {WORKERS}）")
    print(f"待打分: {len(todo)} 篇（总 {len(files)}，已完成 {len(done)}）\n")
    if not todo:
        print("全部已完成")
        return

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ok, fail = 0, 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(score_one, f): f for f in todo}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                record = future.result()
                with _write_lock, SCORES_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                ok += 1
                if i % 25 == 0 or i == len(todo):
                    elapsed = time.time() - start
                    print(f"  进度 {i}/{len(todo)}  成功 {ok} 失败 {fail}  "
                          f"用时 {elapsed:.0f}s  均速 {elapsed / i:.1f}s/篇")
            except Exception as e:
                fail += 1
                print(f"  [FAIL] {path.name[:30]}: {str(e)[:80]}")

    print("\n" + "=" * 50)
    print(f"完成: 成功 {ok}, 失败 {fail} → {SCORES_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
