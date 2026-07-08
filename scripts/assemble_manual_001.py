"""
Step 2-B：将 thinking.txt + article.txt 组装为 Alpaca JSONL

原理：
  - 读取分离的 thinking / article 文本，便于单独修改
  - output = <thinking> + <article> 标签块（训练用）
  - 写入 JSONL（每行一条 JSON，供后续训练读取）

用法：
  python scripts/assemble_manual_001.py
"""

import json
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# 全数据集共用的 instruction（固定不变）
INSTRUCTION = (
    "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
    "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
)

# 本条样本的 input（主题 + 受众 + 要求）
INPUT_TEXT = (
    "主题：为什么大模型「越开源」，创业越难\n"
    "受众：关注科技的互联网从业者\n"
    "要求：2000字左右，有明确论点，引用可验证的事实或公开数据，"
    "至少1个反直觉判断，避免营销话术"
)


def load_text(filename: str) -> str:
    """读取 raw 目录下的文本文件并去除首尾空白"""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_output(thinking: str, article: str) -> str:
    """
    拼接 output 字段

    格式：<thinking>中文5点</thinking> + 空行 + <article>正文</article>
    推理部署时会剥离 <thinking> 块，用户只看到 article
    """
    return f"<thinking>\n{thinking}\n</thinking>\n\n<article>\n{article}\n</article>"


def count_chinese_chars(text: str) -> int:
    """粗略统计汉字数量（用于检查正文长度）"""
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def main():
    thinking = load_text("thinking.txt")
    article = load_text("article.txt")
    output = build_output(thinking, article)

    sample = {
        "instruction": INSTRUCTION,
        "input": INPUT_TEXT,
        "output": output,
    }

    out_path = RAW_DIR / "manual_001.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 统计信息（帮助判断训练显存）
    article_chars = count_chinese_chars(article)
    thinking_chars = count_chinese_chars(thinking)
    total_chars = count_chinese_chars(output)
    est_tokens = int(total_chars * 1.6)  # 中文粗估：1 字 ≈ 1.5–1.8 token

    print("=" * 50)
    print("组装完成")
    print("=" * 50)
    print(f"输出文件: {out_path}")
    print(f"thinking 汉字数: {thinking_chars}（建议 300-500）")
    print(f"article  汉字数: {article_chars}（建议 1800-2500）")
    print(f"output 总汉字数: {total_chars}")
    print(f"预估 tokens:   ~{est_tokens}（4096 内较安全，超过需调 max_seq_length）")
    print()
    if article_chars > 2800:
        print("[WARN] 正文偏长，Step 4 训练时建议 max_seq_length=8192 或适当压缩正文")
    else:
        print("[OK] 长度在可接受范围")
    print("=" * 50)


if __name__ == "__main__":
    main()
