"""
教师模型 A/B 测试：4 家模型 × 相同 3 篇文章做反向蒸馏，自动打分对比

评分维度（每篇满分 10）：
  格式合规（3 分）：<input>/<thinking> 标签齐全 1 分；主题/受众/要求三字段全 2 分
  构思完整（3 分）：核心矛盾/读者痛点/文章结构/语气定位/关键素材，5 点每点 0.6
  口吻正确（3 分）：出现「打算/准备/我会/先…再」计划口吻 +1.5；
                    无「本文/这篇文章」读后感口吻 +1.5
  长度达标（1 分）：thinking 300-500 汉字给满分，250-600 给 0.5

输出：
  data/ab_test/{provider}__{文章}.txt   各家原始响应
  data/ab_test/report.md                汇总报告（表格 + thinking 并排对照）

用法：
  python scripts/ab_test_providers.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from llm_client import chat, get_model

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
AB_DIR = ROOT / "data" / "ab_test"
PROMPTS_DIR = ROOT / "data" / "prompts"

# 3 篇不同题材的测试文章（科技 / 商业消费 / 社会观察）
ARTICLES = [
    "谷歌病了.md",
    "奈雪的体面被撕破了.md",
    "县城的老人何去何从.md",
]

TEST_PROVIDERS = ["deepseek", "kimi", "qwen", "gemini"]

FIVE_POINTS = ("核心矛盾", "读者痛点", "文章结构", "语气定位", "关键素材")
PLAN_TONE = re.compile(r"打算|准备|我会|先.{0,12}再")
REVIEW_TONE = re.compile(r"本文|这篇文章|文章讲述|作者认为")


def cn_count(text: str) -> int:
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def score_response(raw: str) -> tuple[float, dict]:
    """按四个维度打分，返回 (总分, 明细)"""
    detail: dict = {}

    input_match = re.search(r"<input>(.*?)</input>", raw, re.DOTALL)
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)

    # 1) 格式合规 3 分
    fmt = 0.0
    if input_match and thinking_match:
        fmt += 1.0
    input_text = input_match.group(1) if input_match else ""
    fields = sum(1 for f in ("主题：", "受众：", "要求：") if f in input_text)
    fmt += (fields / 3) * 2.0
    detail["格式"] = round(fmt, 2)

    thinking = thinking_match.group(1).strip() if thinking_match else ""

    # 2) 构思完整 3 分
    points = sum(1 for p in FIVE_POINTS if p in thinking)
    completeness = points * 0.6
    detail["构思5点"] = f"{points}/5 ({completeness:.1f})"

    # 3) 口吻 3 分
    tone = 0.0
    if PLAN_TONE.search(thinking):
        tone += 1.5
    review_hits = REVIEW_TONE.findall(thinking)
    if not review_hits:
        tone += 1.5
    detail["口吻"] = round(tone, 2)
    detail["读后感词"] = ",".join(set(review_hits)) if review_hits else "-"

    # 4) 长度 1 分
    n = cn_count(thinking)
    if 300 <= n <= 500:
        length = 1.0
    elif 250 <= n <= 600:
        length = 0.5
    else:
        length = 0.0
    detail["thinking字数"] = n

    total = fmt + completeness + tone + length
    return round(total, 2), detail


def main():
    AB_DIR.mkdir(parents=True, exist_ok=True)
    system = (PROMPTS_DIR / "reverse_system.txt").read_text(encoding="utf-8")

    results: list[dict] = []
    thinking_excerpts: dict[str, dict[str, str]] = {a: {} for a in ARTICLES}

    for article_name in ARTICLES:
        article = (CLEANED_DIR / article_name).read_text(encoding="utf-8").strip()
        for provider in TEST_PROVIDERS:
            tag = f"{provider} × {article_name[:12]}"
            print(f"[{tag}] ...", flush=True)
            start = time.time()
            try:
                raw = chat(
                    system=system,
                    user=f"请反推以下文章的写作任务和构思：\n\n{article}",
                    provider=provider,
                    temperature=0.4,
                    max_tokens=4096,
                )
                elapsed = time.time() - start
                total, detail = score_response(raw)

                out = AB_DIR / f"{provider}__{article_name.replace('.md', '')}.txt"
                out.write_text(raw, encoding="utf-8")

                tm = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
                thinking_excerpts[article_name][provider] = (
                    tm.group(1).strip() if tm else "(解析失败)"
                )

                results.append({
                    "provider": provider,
                    "article": article_name,
                    "score": total,
                    "elapsed": round(elapsed, 1),
                    **detail,
                })
                print(f"  得分 {total}/10, 耗时 {elapsed:.1f}s")
            except Exception as e:
                results.append({
                    "provider": provider, "article": article_name,
                    "score": 0.0, "elapsed": round(time.time() - start, 1),
                    "格式": 0, "构思5点": "-", "口吻": 0,
                    "读后感词": "-", "thinking字数": 0,
                    "error": str(e)[:80],
                })
                print(f"  FAIL: {str(e)[:100]}")
            time.sleep(1)

    # ---- 生成报告 ----
    lines = ["# 教师模型 A/B 测试报告", ""]
    lines.append("## 总分排名（3 篇平均）")
    lines.append("")
    lines.append("| 厂商 | 模型 | 平均分 | 平均耗时 |")
    lines.append("|------|------|--------|----------|")
    for provider in TEST_PROVIDERS:
        rows = [r for r in results if r["provider"] == provider]
        avg = sum(r["score"] for r in rows) / len(rows)
        avg_t = sum(r["elapsed"] for r in rows) / len(rows)
        lines.append(f"| {provider} | {get_model(provider)} | {avg:.2f}/10 | {avg_t:.1f}s |")

    lines.append("")
    lines.append("## 单篇明细")
    lines.append("")
    lines.append("| 厂商 | 文章 | 总分 | 格式 | 构思5点 | 口吻 | 读后感词 | thinking字数 | 耗时 |")
    lines.append("|------|------|------|------|---------|------|----------|--------------|------|")
    for r in results:
        lines.append(
            f"| {r['provider']} | {r['article'][:14]} | {r['score']} | {r.get('格式', '-')} "
            f"| {r.get('构思5点', '-')} | {r.get('口吻', '-')} | {r.get('读后感词', '-')} "
            f"| {r.get('thinking字数', '-')} | {r['elapsed']}s |"
        )

    lines.append("")
    lines.append("## thinking 并排对照（人工评审用）")
    for article_name in ARTICLES:
        lines.append("")
        lines.append(f"### 《{article_name.replace('.md', '')}》")
        for provider in TEST_PROVIDERS:
            lines.append("")
            lines.append(f"#### {provider}")
            lines.append("")
            lines.append("```text")
            lines.append(thinking_excerpts[article_name].get(provider, "(失败)"))
            lines.append("```")

    report = AB_DIR / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已生成: {report}")


if __name__ == "__main__":
    main()
