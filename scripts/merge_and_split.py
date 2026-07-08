"""
Step 2-F：合并全部数据源 → 校验 → 去重 → train/val 划分

数据源：
  reverse_distilled_selected.jsonl   375 条（Qwen/Kimi 批量，主力）
  reverse_distilled_gemini.jsonl       3 条（Gemini 试跑）
  reverse_distilled_deepseek.jsonl     3 条（DeepSeek 试跑）
  raw/manual_001.jsonl                 1 条（手工样本）
  gemini_*.jsonl                       1 条（正向蒸馏样本）

处理规则：
  1. 按 <article> 内容指纹去重（试跑样本可能与批量重复，保留批量版）
  2. 质量门禁：标签齐全、thinking 含「核心矛盾」、正文 >= 800 汉字
  3. 剥离 teacher 等额外字段（训练文件只留 instruction/input/output）
  4. 打乱后按 95/5 划分 train/val（固定种子，可复现）
  5. 统计 token 长度分布，为 Step 4 的 max_seq_length 提供依据

输出：
  data/final/train.jsonl / val.jsonl
  data/final/dataset_info.json      （LLaMA-Factory 数据集注册文件）
  data/final/provenance.jsonl       （每条数据的来源与教师，供消融实验）

用法：
  python scripts/merge_and_split.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generated"
FINAL = ROOT / "data" / "final"

SEED = 42
VAL_RATIO = 0.05
MIN_ARTICLE_CN = 800

# (文件, 来源标记, 优先级)  优先级小者在重复时保留
SOURCES = [
    (GEN / "reverse_distilled_selected.jsonl", "selected_batch", 0),
    (ROOT / "data" / "raw" / "manual_001.jsonl", "manual", 1),
    (GEN / "reverse_distilled_gemini.jsonl", "gemini_pilot", 2),
    (GEN / "reverse_distilled_deepseek.jsonl", "deepseek_pilot", 2),
]


def cn_count(text: str) -> int:
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def article_fingerprint(output: str) -> str:
    """取 <article> 内容的汉字指纹（前 500 汉字哈希，足以判重）"""
    m = re.search(r"<article>(.*?)</article>", output, re.DOTALL)
    body = m.group(1) if m else output
    cn = "".join(c for c in body if "\u4e00" <= c <= "\u9fff")[:500]
    return hashlib.md5(cn.encode()).hexdigest()


def quality_gate(record: dict) -> str | None:
    """返回不合格原因；None = 通过"""
    out = record.get("output", "")
    if not record.get("instruction") or not record.get("input"):
        return "缺 instruction/input"
    for tag in ("<thinking>", "</thinking>", "<article>", "</article>"):
        if tag not in out:
            return f"缺 {tag}"
    thinking = out.split("</thinking>")[0]
    if "核心矛盾" not in thinking:
        return "thinking 缺核心矛盾"
    m = re.search(r"<article>(.*?)</article>", out, re.DOTALL)
    if cn_count(m.group(1)) < MIN_ARTICLE_CN:
        return f"正文过短(<{MIN_ARTICLE_CN}字)"
    return None


def main():
    FINAL.mkdir(parents=True, exist_ok=True)

    # ---- 读取 + 正向蒸馏样本 ----
    forward_files = sorted(GEN.glob("gemini_*.jsonl"))
    sources = SOURCES + [(f, "gemini_forward", 2) for f in forward_files]

    raw_records: list[tuple[dict, str, int]] = []
    for path, tag, priority in sources:
        if not path.exists():
            print(f"[跳过] {path.name} 不存在")
            continue
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                raw_records.append((json.loads(line), tag, priority))
    print(f"合并前总数: {len(raw_records)}")

    # ---- 质量门禁 ----
    passed: list[tuple[dict, str, int]] = []
    rejected = Counter()
    for record, tag, priority in raw_records:
        reason = quality_gate(record)
        if reason:
            rejected[reason] += 1
        else:
            passed.append((record, tag, priority))
    if rejected:
        print(f"质量门禁淘汰: {dict(rejected)}")

    # ---- 去重（优先级小者保留）----
    passed.sort(key=lambda x: x[2])
    seen: dict[str, str] = {}
    unique: list[tuple[dict, str]] = []
    dup_count = 0
    for record, tag, _ in passed:
        fp = article_fingerprint(record["output"])
        if fp in seen:
            dup_count += 1
            continue
        seen[fp] = tag
        unique.append((record, tag))
    print(f"重复淘汰: {dup_count}，最终: {len(unique)} 条")

    # ---- token 长度统计（中文 1 字≈1.6 token 粗估）----
    lengths = []
    for record, _ in unique:
        total_cn = cn_count(record["output"]) + cn_count(record["input"])
        lengths.append(int(total_cn * 1.6))
    lengths.sort()
    n = len(lengths)
    over_4k = sum(1 for l in lengths if l > 4096)
    over_8k = sum(1 for l in lengths if l > 8192)
    print(f"\n预估 token: 中位 {lengths[n // 2]}, P90 {lengths[int(n * 0.9)]}, 最大 {lengths[-1]}")
    print(f"超 4096: {over_4k} 条 ({over_4k / n * 100:.0f}%), 超 8192: {over_8k} 条 ({over_8k / n * 100:.0f}%)")

    # ---- 打乱 + 划分 ----
    rng = random.Random(SEED)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * VAL_RATIO))
    val, train = unique[:n_val], unique[n_val:]

    def dump(items: list[tuple[dict, str]], path: Path):
        with path.open("w", encoding="utf-8") as f:
            for record, _ in items:
                clean = {k: record[k] for k in ("instruction", "input", "output")}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    dump(train, FINAL / "train.jsonl")
    dump(val, FINAL / "val.jsonl")

    # 来源追踪文件（消融实验用）
    with (FINAL / "provenance.jsonl").open("w", encoding="utf-8") as f:
        for split, items in (("train", train), ("val", val)):
            for record, tag in items:
                f.write(json.dumps({
                    "split": split,
                    "source": tag,
                    "teacher": record.get("teacher", tag),
                    "input_head": record["input"][:40],
                }, ensure_ascii=False) + "\n")

    # LLaMA-Factory 数据集注册
    dataset_info = {
        "gzh_writing_train": {
            "file_name": "train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "gzh_writing_val": {
            "file_name": "val.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    (FINAL / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    src_dist = Counter(tag for _, tag in unique)
    print(f"\n来源分布: {dict(src_dist)}")
    print(f"train: {len(train)} 条, val: {len(val)} 条 → {FINAL}")


if __name__ == "__main__":
    main()
