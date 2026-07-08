"""
Step 5-A（诊断）：扫描 data/final 中残留的公众号版式噪声

原理：
  微调后模型输出了「封面来源｜pexels」「点个小爱心」等版式垃圾，
  说明 clean_raw_articles.py 有漏网之鱼进入了训练集。
  SFT 会忠实模仿 output 中的一切，必须先量化污染面，再决定清洗策略。

  检查两个位置：
  1. 文章尾部（最后 15 行）——转载声明、作者栏、求赞
  2. 全文——封面来源、图片来源等行内噪声

用法：
  python -u scripts/scan_boilerplate.py
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final"

# 每个模式配一个可读名称，便于统计
PATTERNS = {
    "封面/题图来源": re.compile(r"封面来源|题图来源|头图来源|图片来源|封面图"),
    "作者/编辑署名栏": re.compile(r"^\s*\*{0,2}\s*(作者|编辑|排版|校对|设计)\s*[:：|｜]", re.M),
    "转载/授权声明": re.compile(r"未经授权|不得转载|授权发布|转载请|白名单"),
    "本文来自X": re.compile(r"本文来[自源]"),
    "求赞求关注": re.compile(r"点个|在看|小爱心|求关注|分享给|点赞|star标"),
    "联系邮箱": re.compile(r"[\w.-]+@[\w.-]+\.\w+"),
    "参考文献链接堆": re.compile(r"^\[\d+\]\s*.{0,80}https?://", re.M),
    "微信号推广": re.compile(r"微信号|公众号ID|添加微信|扫码"),
}


def main():
    hits = Counter()
    polluted = set()
    examples = {}

    for split in ("train.jsonl", "val.jsonl"):
        for i, line in enumerate((FINAL / split).read_text(encoding="utf-8").strip().splitlines()):
            d = json.loads(line)
            output = d["output"]
            # 只看 article 部分（thinking 是教师生成的，无版式噪声）
            m = re.search(r"<article>(.*?)</article>", output, re.S)
            article = m.group(1) if m else output

            tail = "\n".join(article.strip().splitlines()[-15:])
            for name, pat in PATTERNS.items():
                # 版式噪声主要在尾部；行内噪声查全文
                scope = article if name in ("封面/题图来源", "作者/编辑署名栏") else tail
                found = pat.search(scope)
                if found:
                    hits[name] += 1
                    polluted.add(f"{split}#{i}")
                    if name not in examples:
                        start = max(0, found.start() - 30)
                        examples[name] = scope[start:found.end() + 50].replace("\n", " | ")

    total = 376
    print(f"总样本: {total}, 受污染样本: {len(polluted)} ({len(polluted)/total*100:.1f}%)\n")
    print("各模式命中数:")
    for name, n in hits.most_common():
        print(f"  {n:>4}  {name}")
        print(f"        示例: ...{examples[name][:100]}...")


if __name__ == "__main__":
    main()
