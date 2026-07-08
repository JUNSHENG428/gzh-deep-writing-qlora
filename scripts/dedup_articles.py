"""
Step 2-E-1：去重（data/cleaned 内部查重，重复的移入 data/duplicates）

原理：
  不同公众号常互相转载同一篇文章，训练数据重复会导致模型过拟合该篇风格。
  两级检测：
    1. 标题归一化完全相同 → 重复
    2. 内容近似重复：bottom-k MinHash 估算 5 字滑窗的 Jaccard 相似度，
       > 0.7 判定为转载/微改版本
  重复组内保留汉字数最多的版本（转载往往有删节）。

用法：
  python scripts/dedup_articles.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
DUP_DIR = ROOT / "data" / "duplicates"

SHINGLE = 5      # 滑窗长度（字）
BOTTOM_K = 128   # MinHash 签名大小
JACCARD_THRESHOLD = 0.7


def normalize_title(name: str) -> str:
    """文件名去标点、去空白，用于标题级查重"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", name.replace(".md", "")).lower()


def chinese_only(text: str) -> str:
    """只保留汉字，消除排版差异对指纹的影响"""
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


def bottom_k_signature(text: str) -> frozenset[int]:
    """bottom-k MinHash：取所有 5 字滑窗哈希中最小的 k 个作为签名"""
    cn = chinese_only(text)
    hashes = {
        int.from_bytes(
            hashlib.blake2b(cn[i:i + SHINGLE].encode(), digest_size=8).digest(), "big"
        )
        for i in range(max(1, len(cn) - SHINGLE + 1))
    }
    return frozenset(sorted(hashes)[:BOTTOM_K])


def estimated_jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """用两签名的并集 bottom-k 估算 Jaccard"""
    union_k = sorted(a | b)[:BOTTOM_K]
    inter = len(set(union_k) & a & b)
    return inter / len(union_k) if union_k else 0.0


def main():
    DUP_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CLEANED_DIR.glob("*.md"))
    print(f"待查重: {len(files)} 篇")

    texts = {f.name: f.read_text(encoding="utf-8") for f in files}
    cn_lens = {name: len(chinese_only(t)) for name, t in texts.items()}

    # ---- 1. 标题级查重 ----
    by_title: dict[str, list[str]] = {}
    for name in texts:
        by_title.setdefault(normalize_title(name), []).append(name)
    title_dups: set[str] = set()
    for group in by_title.values():
        if len(group) > 1:
            group.sort(key=lambda n: cn_lens[n], reverse=True)
            title_dups.update(group[1:])  # 保留最长的

    # ---- 2. 内容级查重（MinHash）----
    remaining = [n for n in texts if n not in title_dups]
    sigs = {n: bottom_k_signature(texts[n]) for n in remaining}

    content_dups: set[str] = set()
    names = sorted(remaining, key=lambda n: cn_lens[n], reverse=True)  # 长的优先保留
    kept: list[str] = []
    for name in names:
        dup_of = None
        for k in kept:
            if estimated_jaccard(sigs[name], sigs[k]) > JACCARD_THRESHOLD:
                dup_of = k
                break
        if dup_of:
            content_dups.add(name)
        else:
            kept.append(name)

    # ---- 移动重复文件 ----
    all_dups = title_dups | content_dups
    for name in all_dups:
        shutil.move(str(CLEANED_DIR / name), str(DUP_DIR / name))

    print("=" * 50)
    print(f"标题重复: {len(title_dups)}")
    print(f"内容重复: {len(content_dups)}")
    print(f"移入 duplicates: {len(all_dups)}")
    print(f"保留唯一文章: {len(files) - len(all_dups)} 篇")
    print("=" * 50)


if __name__ == "__main__":
    main()
