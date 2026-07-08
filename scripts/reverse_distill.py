"""
Step 2-D-2：反向蒸馏（多厂商版）
真实文章 → input + thinking → Alpaca JSONL

原理：
  1. 读取 data/cleaned/*.md（清洗后的真实公众号文章）
  2. 发给教师模型，反推「写作任务 input」和「动笔前构思 thinking」
  3. output = <thinking>反推构思</thinking> + <article>原文（一字不改）</article>
  4. 追加写入 data/generated/reverse_distilled_{厂商}.jsonl

多厂商与断点续跑：
  - --provider 选 deepseek / kimi / qwen / gemini（Key 配在 .env）
  - 每个厂商独立的 manifest（data/generated/reverse_manifest_{厂商}.json），
    中断后重跑自动跳过已完成的文章

用法（venv 已激活）：
  python scripts/reverse_distill.py --provider deepseek              # 试跑 3 篇
  python scripts/reverse_distill.py --provider deepseek --limit 20
  python scripts/reverse_distill.py --provider deepseek --limit 0    # 全部
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from llm_client import PROVIDERS, available_providers, chat, get_model

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
GENERATED_DIR = ROOT / "data" / "generated"
PROMPTS_DIR = ROOT / "data" / "prompts"

# 与 manual_001 / 正向蒸馏保持一致的固定 instruction
INSTRUCTION = (
    "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
    "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
)

# 读后感口吻关键词（thinking 质量过滤）
REVIEW_TONE_KEYWORDS = ("本文", "这篇文章", "文章讲述", "作者认为")


def out_paths(provider: str) -> tuple[Path, Path]:
    """返回该厂商的输出 JSONL 与断点 manifest 路径"""
    return (
        GENERATED_DIR / f"reverse_distilled_{provider}.jsonl",
        GENERATED_DIR / f"reverse_manifest_{provider}.json",
    )


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_reverse(raw: str) -> tuple[str, str]:
    """解析 <input> 和 <thinking> 标签"""
    input_match = re.search(r"<input>(.*?)</input>", raw, re.DOTALL)
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    if not input_match:
        raise ValueError("缺少 <input> 标签")
    if not thinking_match:
        raise ValueError("缺少 <thinking> 标签")
    return input_match.group(1).strip(), thinking_match.group(1).strip()


def validate(input_text: str, thinking: str) -> list[str]:
    """基础质量检查，返回警告列表"""
    warnings: list[str] = []
    for field in ("主题：", "受众：", "要求："):
        if field not in input_text:
            warnings.append(f"input 缺少「{field}」字段")
    if "核心矛盾" not in thinking:
        warnings.append("thinking 缺少「核心矛盾」")
    for kw in REVIEW_TONE_KEYWORDS:
        if kw in thinking:
            warnings.append(f"thinking 疑似读后感口吻（含「{kw}」）")
    n_cn = sum(1 for c in thinking if "\u4e00" <= c <= "\u9fff")
    if n_cn < 200:
        warnings.append(f"thinking 偏短（{n_cn} 汉字）")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="反向蒸馏真实文章（多厂商）")
    parser.add_argument(
        "--provider", type=str, default="deepseek",
        choices=list(PROVIDERS), help="教师模型厂商",
    )
    parser.add_argument("--limit", type=int, default=3, help="处理篇数，0=全部")
    args = parser.parse_args()

    if args.provider not in available_providers():
        key_env = PROVIDERS[args.provider]["key_env"]
        print(f"[ERROR] {args.provider} 未配置 Key，请在 .env 中填写 {key_env}")
        sys.exit(1)

    system = (PROMPTS_DIR / "reverse_system.txt").read_text(encoding="utf-8")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    out_jsonl, manifest_path = out_paths(args.provider)
    manifest = load_manifest(manifest_path)

    files = sorted(CLEANED_DIR.glob("*.md"))
    todo = [f for f in files if manifest.get(f.name) != "done"]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"厂商: {args.provider}  模型: {get_model(args.provider)}")
    print(f"待处理: {len(todo)} 篇（总 {len(files)}，该厂商已完成 {len(files) - len(todo)}）\n")

    ok, fail = 0, 0
    for i, path in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {path.stem[:30]}")
        article = path.read_text(encoding="utf-8").strip()
        try:
            raw = chat(
                system=system,
                user=f"请反推以下文章的写作任务和构思：\n\n{article}",
                provider=args.provider,
                temperature=0.4,   # 反推是忠实提炼任务，温度低更稳定
                max_tokens=4096,   # 只需 input+thinking
            )
            input_text, thinking = parse_reverse(raw)
            warnings = validate(input_text, thinking)

            record = {
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": f"<thinking>\n{thinking}\n</thinking>\n\n<article>\n{article}\n</article>",
            }
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            manifest[path.name] = "done"
            ok += 1
            print("  [WARN] " + "; ".join(warnings) if warnings else "  [OK]")
        except Exception as e:
            manifest[path.name] = f"failed: {str(e)[:80]}"
            fail += 1
            print(f"  [FAIL] {str(e)[:100]}")

        save_manifest(manifest_path, manifest)
        time.sleep(1)  # 限速

    print("\n" + "=" * 50)
    print(f"本轮完成: 成功 {ok}, 失败 {fail}")
    print(f"输出: {out_jsonl}")
    print(f"断点记录: {manifest_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
