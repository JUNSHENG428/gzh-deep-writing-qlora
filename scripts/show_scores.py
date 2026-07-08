"""查看打分结果分布（清单 + 统计）"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES_FILE = ROOT / "data" / "generated" / "article_scores.jsonl"


def main():
    records = [
        json.loads(line)
        for line in SCORES_FILE.read_text(encoding="utf-8").strip().splitlines()
    ]
    print(f"已打分: {len(records)} 篇\n")

    if len(records) <= 20:
        for d in records:
            print(f"depth={d['depth']} evid={d['evidence']} "
                  f"{d['genre']:<5} {d['topic']:<3} {d['chars']:>6}字  {d['file'][:30]}")
        print()

    print("depth 分布:", dict(sorted(Counter(r["depth"] for r in records).items())))
    print("evidence 分布:", dict(sorted(Counter(r["evidence"] for r in records).items())))
    print("文体分布:", dict(Counter(r["genre"] for r in records).most_common()))
    print("题材分布:", dict(Counter(r["topic"] for r in records).most_common()))

    good = [r for r in records if r["depth"] >= 4 and r["genre"] == "深度评论"]
    print(f"\n精选候选（depth>=4 且 深度评论）: {len(good)} 篇 "
          f"({len(good) / len(records) * 100:.0f}%)")


if __name__ == "__main__":
    main()
