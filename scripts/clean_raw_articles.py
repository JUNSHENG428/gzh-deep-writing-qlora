"""
Step 2-D-1：清洗爬取的公众号文章（data/raw/*.md → data/cleaned/*.md）

原理：
  爬取的 md 混有封面图、导航链接、日期行、评论区等噪声。
  若不清洗直接训练，模型会学到「在文末生成精选留言」等垃圾模式。

清洗规则（针对本批虎嗅系爬取格式）：
  1. 删除图片行 ![...](...)
  2. 删除 [ 虎嗅APP ](javascript...) 等无效链接
  3. 删除日期行（_2026年xx月xx日..._）
  4. 删除「在小说阅读器读本章」「去阅读」等按钮文本
  5. 删除「以下文章来源于 / 本文来自微信公众号」等转载引言块
  6. 从「分享 留言 收藏」或「精选留言」起截断（评论区）
  7. 跳过「早报」类新闻拼盘（非深度文章）
  8. 清洗后不足 1200 汉字的丢弃（太短撑不起深度写作）

用法：
  python scripts/clean_raw_articles.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"

# 整行匹配即删除的模式
LINE_DROP_PATTERNS = [
    re.compile(r"^!\[.*?\]\(.*?\)\s*$"),            # 图片行
    re.compile(r"^\[.*?\]\(javascript.*?\)\s*$"),    # javascript 伪链接
    re.compile(r"^_\s*\d{4}年\d{1,2}月\d{1,2}日.*_"),  # 日期行
    re.compile(r"^在小说阅读器读本章\s*$"),
    re.compile(r"^去阅读\s*$"),
    re.compile(r"^以下文章来源于.*$"),
    re.compile(r"^预览时标签不可点\s*$"),
    re.compile(r"^原创\s+.*\]\(javascript.*$"),      # 原创 作者 [ 公众号 ](javascript...) 署名行
    re.compile(r".*👉.*"),                           # 文末「👉加入社群」类推广
]

# 行内噪声：本文来自微信公众号引言（保留后续正文）
INLINE_SOURCE_PATTERN = re.compile(r"^本文来自微信公众号.*?$")

# 评论区/文末噪声起始标记：出现即截断全文
COMMENT_MARKERS = [
    "分享  留言  收藏",
    "精选留言",
    "微信扫一扫",
    "继续滑动看下一个",
    "本内容由作者授权发布",
    "如对本稿件有异议或投诉",
]

# 文末孤立噪声行（截断后再逐行清理）
TAIL_DROP_EXACT = {"End", "阅读", "去阅读", "["}
TAIL_DROP_PATTERNS = [
    re.compile(r"^\*\*.*想涨知识.*\*\*\s*$"),
    re.compile(r"^\*\*.*关注虎嗅.*\*\*\s*$"),
    re.compile(r"^[\*_\s]+$"),                     # 纯 **** / __ 空行
    re.compile(r"^\]\(https?://.*\)\s*$"),         # 图链块的收尾 ](url)
    re.compile(r"^\[?!\[.*\]\(.*$"),               # [![ 开头的推广图链接
    re.compile(r"^\*\*\s*文\s*\*\*.*[｜|].*$"),     # ** 文 ｜ 作者 署名行
    re.compile(r"^.*来源[｜|].*封面来源.*$"),        # 来源｜xxx 封面来源｜xxx
]

MIN_CHINESE_CHARS = 1200  # 清洗后正文最低汉字数


def count_chinese(text: str) -> int:
    """统计汉字数"""
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def clean_article(text: str) -> str:
    """执行清洗规则，返回干净正文"""
    # 1. 评论区截断（取最早出现的标记位置）
    cut = len(text)
    for marker in COMMENT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut]

    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()

        # 2. 整行噪声
        if any(p.match(stripped) for p in LINE_DROP_PATTERNS):
            continue
        # 3. 转载引言行
        if INLINE_SOURCE_PATTERN.match(stripped):
            continue
        # 4. 公众号简介行（形如 **xxx** .  简介文字）
        if re.match(r"^\*\*.{1,20}\*\*\s*\.", stripped):
            continue
        # 5. 文末孤立噪声行
        if stripped in TAIL_DROP_EXACT:
            continue
        if any(p.match(stripped) for p in TAIL_DROP_PATTERNS):
            continue

        # 6. 修复错乱的小节标题：_ _ 苗圃 _ _ → ## 苗圃
        m = re.match(r"^_+\s*(_*\s*)?(.{1,30}?)\s*_+\s*(_+\s*)?$", stripped)
        if m and stripped.startswith("_"):
            title = m.group(2).strip().strip("_").strip()
            if title:
                kept.append(f"## {title}")
                continue

        kept.append(line.rstrip())

    cleaned = "\n".join(kept)
    # 5. 压缩连续空行为最多一个
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(RAW_DIR.glob("*.md"))
    stats = {"total": len(md_files), "zaobao": 0, "too_short": 0, "ok": 0}
    lengths: list[int] = []

    for path in md_files:
        # 跳过早报类新闻拼盘
        if path.stem.startswith("早报"):
            stats["zaobao"] += 1
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_article(text)
        n_chars = count_chinese(cleaned)

        if n_chars < MIN_CHINESE_CHARS:
            stats["too_short"] += 1
            continue

        (CLEANED_DIR / path.name).write_text(cleaned, encoding="utf-8")
        stats["ok"] += 1
        lengths.append(n_chars)

    lengths.sort()
    print("=" * 50)
    print("清洗完成")
    print("=" * 50)
    print(f"原始文件:      {stats['total']}")
    print(f"跳过早报:      {stats['zaobao']}")
    print(f"太短丢弃:      {stats['too_short']}（<{MIN_CHINESE_CHARS} 汉字）")
    print(f"清洗保留:      {stats['ok']} → {CLEANED_DIR}")
    if lengths:
        n = len(lengths)
        print(f"保留文章汉字数: 中位 {lengths[n//2]}, 最短 {lengths[0]}, 最长 {lengths[-1]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
