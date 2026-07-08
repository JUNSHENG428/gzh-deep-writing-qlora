"""
Step 2-C-1：用 Gemini 蒸馏【单条】训练样本

原理：
  1. 读取 distill_system.txt 作为 system_instruction（教师角色）
  2. 读取 distill_user_template.txt，填入主题后作为 user 消息
  3. 调用 Gemini API 生成 <thinking> + <article>
  4. 解析标签 → 组装 Alpaca JSONL → 写入 data/generated/

前置条件：
  1. pip install google-genai python-dotenv
  2. 复制 .env.example 为 .env，填入 GEMINI_API_KEY
  3. 在 Google AI Studio 申请 API Key：https://aistudio.google.com/apikey

用法（Windows，venv 已激活）：
  python scripts/distill_one_gemini.py
  python scripts/distill_one_gemini.py --topic "Agent 是下一代 App，还是过度包装"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── 路径常量 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "data" / "prompts"
GENERATED_DIR = ROOT / "data" / "generated"
RAW_DIR = ROOT / "data" / "raw"
TOPICS_FILE = ROOT / "data" / "topics_seed.txt"

# 全数据集共用的 instruction（与 manual_001 保持一致）
INSTRUCTION = (
    "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
    "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
)

# 默认受众与写作要求（科技评论）
DEFAULT_AUDIENCE = "关注科技的互联网从业者"
DEFAULT_REQUIREMENTS = (
    "1800-2200字，有明确论点，引用可验证的事实或公开数据，"
    "至少1个反直觉判断，避免营销话术"
)

# 英文元分析关键词（质量过滤用）
BAD_THINKING_KEYWORDS = (
    "Analyze the Request",
    "Thesis Generation",
    "Drafting Process",
    "Final Polish",
)


def load_prompt(name: str) -> str:
    """读取 Prompt 模板文件"""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"找不到 Prompt 文件: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_first_topic() -> str:
    """从 topics_seed.txt 读取第一个有效主题（跳过 # 注释行）"""
    if not TOPICS_FILE.exists():
        raise FileNotFoundError(f"找不到选题文件: {TOPICS_FILE}")
    for line in TOPICS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError("topics_seed.txt 中没有有效主题")


def build_user_prompt(topic: str, audience: str, requirements: str) -> str:
    """把主题填入 user 模板"""
    template = load_prompt("distill_user_template.txt")
    return template.format(
        topic=topic,
        audience=audience,
        requirements=requirements,
    )


def parse_output(raw_text: str) -> tuple[str, str]:
    """
    从 Gemini 原始输出中解析 thinking 和 article

    原理：用正则提取 <thinking>...</thinking> 和 <article>...</article>
    若标签缺失，抛出 ValueError 便于重试
    """
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)
    article_match = re.search(r"<article>(.*?)</article>", raw_text, re.DOTALL)

    if not thinking_match:
        raise ValueError("输出中缺少 <thinking>...</thinking> 标签")
    if not article_match:
        raise ValueError("输出中缺少 <article>...</article> 标签")

    thinking = thinking_match.group(1).strip()
    article = article_match.group(1).strip()
    return thinking, article


def validate_sample(thinking: str, article: str) -> list[str]:
    """
    基础质量检查，返回警告列表（非空则建议重跑）

    检查项：
      - thinking 是否含英文元分析
      - thinking 是否含中文 5 点结构
      - article 是否以 # 标题开头
    """
    warnings: list[str] = []

    for keyword in BAD_THINKING_KEYWORDS:
        if keyword in thinking:
            warnings.append(f"thinking 含英文元分析关键词: {keyword}")

    if "核心矛盾" not in thinking:
        warnings.append("thinking 缺少「核心矛盾」要点")

    if not article.startswith("#"):
        warnings.append("article 未以 Markdown 标题 (# ...) 开头")

    # 粗略统计汉字数
    chinese_chars = sum(1 for c in article if "\u4e00" <= c <= "\u9fff")
    if chinese_chars < 1200:
        warnings.append(f"article 偏短（约 {chinese_chars} 汉字，建议 ≥1800）")
    elif chinese_chars > 3200:
        warnings.append(f"article 偏长（约 {chinese_chars} 汉字，训练 token 可能超 4096）")

    return warnings


def build_jsonl_record(topic: str, thinking: str, article: str) -> dict:
    """组装 Alpaca 格式的一条记录"""
    input_text = (
        f"主题：{topic}\n"
        f"受众：{DEFAULT_AUDIENCE}\n"
        f"要求：{DEFAULT_REQUIREMENTS}"
    )
    output_text = f"<thinking>\n{thinking}\n</thinking>\n\n<article>\n{article}\n</article>"
    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": output_text,
    }


def call_gemini(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    """
    调用 Gemini API

    参数说明：
      model: 模型 ID；gemini-2.5-pro 长文质量最好；flash 系列可能 503 高峰不可用
      max_output_tokens: 上限 token，长文需 8192+，否则会被截断
      temperature: 0.5–0.7 适合评论写作；越低越稳定，越高越发散
    """
    client = genai.Client(api_key=api_key)

    # 503 高峰时自动重试（最多 3 次，间隔递增）
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=16384,  # 8192 不够，长文+thinking 会被截断
                    temperature=0.6,
                ),
            )
            break
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 5 * (attempt + 1)
                print(f"[WARN] 模型繁忙，{wait}s 后重试 ({attempt + 1}/3)...")
                time.sleep(wait)
            else:
                raise
    else:
        raise last_error

    # 优先取 text；若为空则从 candidates 手动拼接
    text = response.text
    if not text and response.candidates:
        parts = response.candidates[0].content.parts or []
        text = "".join(getattr(p, "text", "") or "" for p in parts).strip()

    if not text:
        finish = getattr(response.candidates[0], "finish_reason", "unknown") if response.candidates else "no_candidates"
        raise RuntimeError(f"Gemini 返回空内容，finish_reason={finish}，可能被安全策略拦截或 token 不足")

    return text.strip()


def save_outputs(
    topic: str,
    raw_text: str,
    record: dict,
    warnings: list[str],
) -> Path:
    """保存原始响应、JSONL 样本，并打印统计"""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)[:30]

    # 原始响应（调试 Prompt 用）
    raw_path = RAW_DIR / f"gemini_response_{safe_topic}_{timestamp}.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    # JSONL 样本
    out_path = GENERATED_DIR / f"gemini_{safe_topic}_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    article = record["output"].split("<article>")[1].split("</article>")[0]
    chinese_chars = sum(1 for c in article if "\u4e00" <= c <= "\u9fff")

    print("=" * 55)
    print("蒸馏完成")
    print("=" * 55)
    print(f"主题:     {topic}")
    print(f"JSONL:    {out_path}")
    print(f"原始响应: {raw_path}")
    print(f"article 汉字数: ~{chinese_chars}")
    print(f"预估 tokens:    ~{int(chinese_chars * 1.6)}（不含 thinking）")

    if warnings:
        print("\n[WARN] 质量警告（建议检查后决定是否重跑）:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] 基础质量检查通过")

    print("=" * 55)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Gemini 蒸馏单条训练样本")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="指定主题；不填则使用 topics_seed.txt 第一条",
    )
    args = parser.parse_args()

    # 加载 .env（GEMINI_API_KEY / GEMINI_MODEL）
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    if not api_key or api_key == "your-api-key-here":
        print("[ERROR] 请先在 .env 中配置 GEMINI_API_KEY")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 到 https://aistudio.google.com/apikey 申请 Key")
        sys.exit(1)

    topic = args.topic or load_first_topic()
    system_prompt = load_prompt("distill_system.txt")
    user_prompt = build_user_prompt(topic, DEFAULT_AUDIENCE, DEFAULT_REQUIREMENTS)

    print(f"模型: {model}")
    print(f"主题: {topic}")
    print("正在调用 Gemini API（长文约需 30-90 秒）...\n")

    raw_text = call_gemini(system_prompt, user_prompt, model, api_key)

    try:
        thinking, article = parse_output(raw_text)
    except ValueError as e:
        # 解析失败时保存原始响应，方便排查截断/格式问题
        debug_path = RAW_DIR / f"gemini_failed_{datetime.now():%Y%m%d_%H%M%S}.txt"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw_text, encoding="utf-8")
        print(f"[ERROR] {e}")
        print(f"原始响应已保存: {debug_path}")
        print("常见原因: max_output_tokens 不足导致 </article> 被截断，请重试")
        sys.exit(1)

    warnings = validate_sample(thinking, article)
    record = build_jsonl_record(topic, thinking, article)
    save_outputs(topic, raw_text, record, warnings)


if __name__ == "__main__":
    main()
