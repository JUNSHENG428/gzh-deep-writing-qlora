"""
Step 5-A（清洗）：清除 data/final 中 article 部分的版式噪声，产出 v2 数据集

策略：
  1. 行级删除：署名栏、封面来源、转载声明、邮箱、扫码推广等整行删除
     （用「行长度上限 + 模式匹配」双条件，避免误伤含相关字样的正文长句）
  2. 尾部剥离：从文章末尾向上，连续剥掉噪声行/链接行/参考文献列表，
     直到遇到正文行为止（顺带解决模型学会编造假参考文献的问题）
  3. 质量闸门：清洗后不足 800 字的样本整条丢弃
  4. 原文件备份到 data/final/backup_v1/

用法：
  python -u scripts/clean_final_boilerplate.py
"""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final"
BACKUP = FINAL / "backup_v1"

# ---- 行级噪声：匹配 且 行长不超限 才删（防误伤正文）----
LINE_NOISE = [
    # (正则, 行长上限)
    (re.compile(r"^\s*[*_\s]*(作者|编辑|排版|校对|设计|策划|视觉|运营|监制|出品)\s*[:：|｜]"), 60),
    (re.compile(r"封面来源|题图来源|头图来源|封面图来源|图片来源|封面图|配图来源"), 60),
    (re.compile(r"本文来[自源]|未经授权|不得转载|转载请|授权发布|白名单|投稿邮箱"), 90),
    (re.compile(r"[\w.-]+@[\w.-]+\.\w{2,}"), 90),
    (re.compile(r"扫码|添加微信|微信号\s*[:：]|长按识别|关注公众号|后台回复"), 60),
    (re.compile(r"点个|求点赞|求在看|戳这里|阅读原文"), 40),
    # 纯符号/纯「点赞 在看 分享」的互动行
    (re.compile(r"^[\s*_\-—|｜·点赞在看分享收藏转发，、]+$"), 30),
    (re.compile(r"^\s*[*_\s]*文\s*[*_\s]*[｜|]"), 60),          # 「文｜某某」署名
    (re.compile(r"来源\s*[｜|].{0,30}$"), 50),                   # 「来源｜某某」
]

# ---- 区块切除标记：尾部出现任一标记，则从该行整段切到文末 ----
# （解决多行参考文献 / 致谢栏 / 固定推广块，尾部剥离逐行删不干净的问题）
TAIL_CUT_MARKERS = re.compile(
    r"^\s*[*_\s]*("
    r"参考文献|参考资料|参考链接|资料来源|图片.{0,6}来源|数据来源|"
    r"策划制作|科学审核|内容审核|本文.{0,4}审核|责编|主编|监制|出品|"
    r"欢迎转发|家有女儿|我们为您推荐|推荐阅读|往期回顾|延伸阅读|点击.{0,6}阅读|"
    r"References|参考$"
    r")"
)

# ---- 尾部逐行剥离：区块切除后仍可能残留的零散行 ----
TAIL_JUNK = [
    re.compile(r"^\s*$"),
    re.compile(r"^\[?\d+\]?[.、]?\s*.{0,120}https?://"),         # [1] xxx http://...
    re.compile(r"^https?://"),
    re.compile(r"^\[\d+\]"),                                     # 无链接的引文条目
    re.compile(r"^[\s*_\-—|｜.·]+$"),
    re.compile(r"^修改于\s*$"),
]


def is_line_noise(line: str) -> bool:
    for pat, max_len in LINE_NOISE:
        if len(line) <= max_len and pat.search(line):
            return True
    return False


def is_tail_junk(line: str) -> bool:
    return is_line_noise(line) or any(p.match(line) for p in TAIL_JUNK)


def clean_article(article: str) -> str:
    lines = [ln for ln in article.strip().splitlines() if not is_line_noise(ln)]

    # 区块切除：仅在尾部 1/3 区域（且至少保留前面正文）找最早的切除标记，
    # 从该行起切到文末。限定尾部避免误伤正文中偶然出现的「参考」等字样。
    tail_zone_start = max(len(lines) * 2 // 3, len(lines) - 25)
    cut_at = None
    for idx in range(tail_zone_start, len(lines)):
        if TAIL_CUT_MARKERS.match(lines[idx]):
            cut_at = idx
            break
    if cut_at is not None:
        lines = lines[:cut_at]

    # 尾部连续剥离残留零散行
    while lines and is_tail_junk(lines[-1]):
        lines.pop()
    # 头部剥掉空行
    while lines and not lines[0].strip():
        lines.pop(0)
    # 压缩连续空行
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip()


def han_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def main():
    BACKUP.mkdir(exist_ok=True)
    stats = {"cleaned": 0, "untouched": 0, "dropped": 0}

    for split in ("train.jsonl", "val.jsonl"):
        src = FINAL / split
        shutil.copy2(src, BACKUP / split)

        kept = []
        for line in src.read_text(encoding="utf-8").strip().splitlines():
            d = json.loads(line)
            m = re.search(r"<article>\n?(.*?)\n?</article>", d["output"], re.S)
            if not m:
                kept.append(d)
                continue
            article = m.group(1)
            cleaned = clean_article(article)

            if han_count(cleaned) < 800:
                stats["dropped"] += 1
                continue
            if cleaned != article.strip():
                stats["cleaned"] += 1
                d["output"] = (
                    d["output"][: m.start()]
                    + f"<article>\n{cleaned}\n</article>"
                    + d["output"][m.end():]
                )
            else:
                stats["untouched"] += 1
            kept.append(d)

        with src.open("w", encoding="utf-8") as f:
            for d in kept:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{split}: 保留 {len(kept)} 条")

    print(f"\n修改 {stats['cleaned']} 条, 原样保留 {stats['untouched']} 条, "
          f"丢弃(清后过短) {stats['dropped']} 条")
    print(f"v1 备份在: {BACKUP}")


if __name__ == "__main__":
    main()
