"""
Step 3-C：用 Qwen 真实分词器精确测量数据集 token 长度

原理：
  之前用「1 汉字 ≈ 1.6 token」粗估严重偏高——Qwen 分词器对中文
  压缩率高（约 1.5 汉字 = 1 token）。max_seq_length 直接决定
  训练显存，必须用真实分词器测完再定。

  每条样本的完整训练序列 = chat模板(instruction + input) + output，
  这里按同样结构拼接后统计。

用法：
  python -u scripts/measure_tokens.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
FINAL = ROOT / "data" / "final"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    lengths = []
    for split in ("train.jsonl", "val.jsonl"):
        for line in (FINAL / split).read_text(encoding="utf-8").strip().splitlines():
            d = json.loads(line)
            messages = [
                {"role": "system", "content": d["instruction"]},
                {"role": "user", "content": d["input"]},
                {"role": "assistant", "content": d["output"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            lengths.append(len(tokenizer(text)["input_ids"]))

    lengths.sort()
    n = len(lengths)
    p = lambda q: lengths[min(n - 1, int(n * q))]

    print(f"样本数: {n}")
    print(f"token 长度: 中位 {p(0.5)}, P75 {p(0.75)}, P90 {p(0.9)}, P95 {p(0.95)}, 最大 {lengths[-1]}")
    for cutoff in (2048, 4096, 6144, 8192, 12288):
        cover = sum(1 for l in lengths if l <= cutoff) / n * 100
        print(f"  max_seq_length={cutoff:>5}: 覆盖 {cover:.1f}% 样本")

    print("\n决策参考（7B QLoRA, batch=1, gradient_checkpointing 开）:")
    print("  4096 → 训练显存约 12-16 GB（最稳）")
    print("  6144 → 约 16-20 GB")
    print("  8192 → 约 20-26 GB（5090D 上限内，留意 OOM）")
    print("超长样本可截断（丢文章结尾）或丢弃，看覆盖率决定")


if __name__ == "__main__":
    main()
