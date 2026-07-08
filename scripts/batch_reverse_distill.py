"""
Step 2-E-4：精选集批量反向蒸馏（多教师派单）

派单规则：
  - 超长文（>8000 汉字）→ kimi（长上下文强项）
  - 其余按文件名哈希稳定分配：约 70% qwen（同族蒸馏主力），30% kimi（教师多样性）
  - 每条记录 teacher 字段，便于训练后做「单教师 vs 多教师」消融实验

并发与断点：
  - 4 线程并发（约 40-60 分钟跑完 375 篇）
  - manifest 断点续跑，中断后重跑自动跳过

用法：
  python -u scripts/batch_reverse_distill.py            # 全部
  python -u scripts/batch_reverse_distill.py --limit 10 # 试跑
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import chat, get_model

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "selected"
PROMPTS_DIR = ROOT / "data" / "prompts"
OUT_JSONL = ROOT / "data" / "generated" / "reverse_distilled_selected.jsonl"
MANIFEST = ROOT / "data" / "generated" / "reverse_manifest_selected.json"

WORKERS = 4
LONG_ARTICLE_CHARS = 8000  # 超过此汉字数派给 kimi
QWEN_RATIO = 7             # 哈希尾数 0-6 → qwen（70%），7-9 → kimi

INSTRUCTION = (
    "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
    "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
)

REVIEW_TONE_KEYWORDS = ("本文", "这篇文章", "文章讲述", "作者认为")

_lock = threading.Lock()


def cn_count(text: str) -> int:
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def assign_teacher(name: str, n_chars: int) -> str:
    """按长度与稳定哈希派单"""
    if n_chars > LONG_ARTICLE_CHARS:
        return "kimi"
    digest = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return "qwen" if digest % 10 < QWEN_RATIO else "kimi"


def parse_reverse(raw: str) -> tuple[str, str]:
    input_match = re.search(r"<input>(.*?)</input>", raw, re.DOTALL)
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    if not input_match:
        raise ValueError("缺少 <input> 标签")
    if not thinking_match:
        raise ValueError("缺少 <thinking> 标签")
    return input_match.group(1).strip(), thinking_match.group(1).strip()


def validate(input_text: str, thinking: str) -> list[str]:
    warnings: list[str] = []
    for field in ("主题：", "受众：", "要求："):
        if field not in input_text:
            warnings.append(f"缺「{field}」")
    if "核心矛盾" not in thinking:
        warnings.append("缺「核心矛盾」")
    for kw in REVIEW_TONE_KEYWORDS:
        if kw in thinking:
            warnings.append(f"读后感腔（{kw}）")
    return warnings


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def process_one(path: Path, system: str, teacher: str) -> tuple[dict, list[str]]:
    article = path.read_text(encoding="utf-8").strip()
    raw = chat(
        system=system,
        user=f"请反推以下文章的写作任务和构思：\n\n{article}",
        provider=teacher,
        temperature=0.4,   # kimi-k2 会被客户端自动改为 1.0
        max_tokens=4096,
    )
    input_text, thinking = parse_reverse(raw)
    warnings = validate(input_text, thinking)
    record = {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": f"<thinking>\n{thinking}\n</thinking>\n\n<article>\n{article}\n</article>",
        "teacher": teacher,
    }
    return record, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="处理篇数，0=全部")
    args = parser.parse_args()

    system = (PROMPTS_DIR / "reverse_system.txt").read_text(encoding="utf-8")
    manifest = load_manifest()

    files = sorted(SRC_DIR.glob("*.md"))
    todo = [f for f in files if manifest.get(f.name, {}).get("status") != "done"]
    n_done = len(files) - len(todo)
    if args.limit > 0:
        todo = todo[: args.limit]

    assignments = {
        f: assign_teacher(f.name, cn_count(f.read_text(encoding="utf-8")))
        for f in todo
    }
    n_qwen = sum(1 for t in assignments.values() if t == "qwen")
    print(f"待处理: {len(todo)} 篇（总 {len(files)}，已完成 {n_done}）")
    print(f"派单: qwen({get_model('qwen')}) {n_qwen} 篇, "
          f"kimi({get_model('kimi')}) {len(todo) - n_qwen} 篇, 并发 {WORKERS}\n")
    if not todo:
        return

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ok, fail = 0, 0

    def worker(path: Path) -> tuple[Path, dict | None, list[str], str | None]:
        teacher = assignments[path]
        try:
            record, warnings = process_one(path, system, teacher)
            return path, record, warnings, None
        except Exception as e:
            return path, None, [], str(e)[:100]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(worker, f) for f in todo]
        for i, future in enumerate(as_completed(futures), 1):
            path, record, warnings, error = future.result()
            with _lock:
                if record:
                    with OUT_JSONL.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest[path.name] = {
                        "status": "done",
                        "teacher": record["teacher"],
                        "warnings": warnings,
                    }
                    ok += 1
                else:
                    manifest[path.name] = {"status": f"failed: {error}"}
                    fail += 1
                    print(f"  [FAIL] {path.name[:30]}: {error}")
                MANIFEST.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - start
                eta = elapsed / i * (len(todo) - i)
                print(f"  进度 {i}/{len(todo)}  成功 {ok} 失败 {fail}  "
                      f"用时 {elapsed / 60:.1f}min  预计剩余 {eta / 60:.0f}min")

    n_warn = sum(1 for v in manifest.values()
                 if isinstance(v, dict) and v.get("warnings"))
    print("\n" + "=" * 50)
    print(f"完成: 成功 {ok}, 失败 {fail}, 带警告 {n_warn}")
    print(f"输出: {OUT_JSONL}")
    print("=" * 50)


if __name__ == "__main__":
    main()
