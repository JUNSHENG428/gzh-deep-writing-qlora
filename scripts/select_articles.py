"""
Step 2-E-3：按打分结果筛选精选集（用于反向蒸馏）

筛选策略（分两档）：
  核心档：depth >= 4 且 genre == 深度评论            ← 直接入选
  补充档：depth == 3 且 evidence >= 4 且 genre == 深度评论
          ← 按 evidence、字数排序补足到目标数量

  访谈类如果 depth >= 4 也纳入核心档（深度对话也是好的写作素材）。
  软文广告 / 拼盘杂烩 / 新闻报道一律不要。

输出：
  data/selected/           精选文章副本（供反向蒸馏读取）
  data/generated/selection_report.txt  筛选报告

用法：
  python scripts/select_articles.py --target 700
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
SELECTED_DIR = ROOT / "data" / "selected"
SCORES_FILE = ROOT / "data" / "generated" / "article_scores.jsonl"
REPORT_FILE = ROOT / "data" / "generated" / "selection_report.txt"

GOOD_GENRES = ("深度评论", "访谈")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=700, help="目标精选篇数")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in SCORES_FILE.read_text(encoding="utf-8").strip().splitlines()
    ]
    print(f"已打分: {len(records)} 篇")

    core = [
        r for r in records
        if r["depth"] >= 4 and r["genre"] in GOOD_GENRES
    ]
    supplement_pool = [
        r for r in records
        if r["depth"] == 3 and r["evidence"] >= 4 and r["genre"] == "深度评论"
    ]
    # 补充档按 evidence 降序、字数适中优先（2000-6000 字最理想）
    supplement_pool.sort(
        key=lambda r: (-r["evidence"], abs(r["chars"] - 3500))
    )

    selected = list(core)
    need = args.target - len(selected)
    if need > 0:
        selected += supplement_pool[:need]

    # 复制精选文章
    if SELECTED_DIR.exists():
        shutil.rmtree(SELECTED_DIR)
    SELECTED_DIR.mkdir(parents=True)
    missing = 0
    for r in selected:
        src = CLEANED_DIR / r["file"]
        if src.exists():
            shutil.copy2(src, SELECTED_DIR / r["file"])
        else:
            missing += 1

    # 报告
    lines = [
        "精选报告",
        "=" * 50,
        f"总打分:   {len(records)}",
        f"核心档（depth>=4 深度评论/访谈）: {len(core)}",
        f"补充档（depth=3 evidence>=4）:   {min(max(need, 0), len(supplement_pool))}",
        f"最终精选: {len(selected)} 篇 → {SELECTED_DIR}",
        "",
        f"题材分布: {dict(Counter(r['topic'] for r in selected).most_common())}",
        f"depth 分布: {dict(sorted(Counter(r['depth'] for r in selected).items()))}",
        f"字数: 中位 {sorted(r['chars'] for r in selected)[len(selected) // 2]}",
    ]
    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    if missing:
        print(f"[WARN] {missing} 篇打分记录对应的文件不存在（可能已被去重移走）")


if __name__ == "__main__":
    main()
