"""
Step 5-B（评估）：对比 base / v1 / v2 三方生成质量，产出量化报告

三个指标：
  1. ROUGE-L（字符级）：生成文与参考文的最长公共子序列 F1，
     衡量结构/用词重合度。中文按字符算 LCS，无需分词。
     注意：写作任务 ROUGE 不宜过高（过高=抄参考），看的是相对变化。
  2. 版式垃圾命中率：直接统计输出里「作者/封面来源/点赞/参考文献链接」等，
     这是本次数据清洗价值的最直接证据（v1 应高、v2 应趋近 0）。
  3. LLM 裁判：Gemini 按 思考深度/论据充分/文风地道/无版式垃圾 四维各 1-5 分。
     （需 .env 配 GEMINI_API_KEY；未配则跳过）

用法：
  python -u scripts/eval_compare.py                # 只算 ROUGE + 版式（快，离线）
  python -u scripts/eval_compare.py --judge        # 额外调 Gemini 裁判
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "eval"
sys.path.insert(0, str(ROOT / "scripts"))

TAGS = ["base", "v1", "v2"]

BOILERPLATE = [
    re.compile(r"封面来源|题图来源|图片来源|封面图"),
    re.compile(r"^\s*[*_\s]*(作者|编辑|排版|校对|责编|监制|策划)\s*[:：|｜]", re.M),
    re.compile(r"未经授权|不得转载|本文来[自源]|授权发布"),
    re.compile(r"点个|在看|小爱心|求关注|扫码|微信号"),
    re.compile(r"^\[?\d+\]?\s*.{0,80}https?://", re.M),
]

JUDGE_SYSTEM = (
    "你是资深公众号主编，负责评审科技/商业深度文章。"
    "请严格、挑剔地按四个维度各打 1-5 分（5 最好），只输出 JSON。"
)
JUDGE_TEMPLATE = """请评审下面这篇文章，按四维打分并只返回 JSON：
- depth: 思考深度（是否有核心论点、反直觉判断，而非罗列常识）
- evidence: 论据充分（是否有具体数据/事实/案例支撑）
- style: 文风地道（是否像人写的公众号深度文，而非车轱辘话/AI腔）
- clean: 无版式垃圾（1=满是作者栏/求赞/参考链接等杂质，5=纯正文干净）

返回格式：{{"depth":x,"evidence":x,"style":x,"clean":x}}

文章主题：{topic}

文章正文：
{article}
"""


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: str, ref: str) -> float:
    hyp = re.sub(r"\s+", "", hyp)
    ref = re.sub(r"\s+", "", ref)
    if not hyp or not ref:
        return 0.0
    l = lcs_len(hyp, ref)
    prec, rec = l / len(hyp), l / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def boilerplate_hits(text: str) -> int:
    return sum(1 for p in BOILERPLATE if p.search(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="调用 Gemini 裁判打分")
    args = ap.parse_args()

    data = {}
    for tag in TAGS:
        f = EVAL_DIR / f"gen_{tag}.json"
        if not f.exists():
            print(f"[跳过] 缺少 {f.name}，先运行 gen_local.py --tag {tag}")
            continue
        data[tag] = json.loads(f.read_text(encoding="utf-8"))

    if not data:
        print("没有可评估的生成结果")
        return

    judge = None
    if args.judge:
        from llm_client import chat, _get_key
        if _get_key("gemini"):
            judge = chat
        else:
            print("[WARN] 未配置 GEMINI_API_KEY，跳过裁判打分\n")

    print("=" * 64)
    print(f"{'指标':<22}{'base':>12}{'v1':>12}{'v2':>12}")
    print("=" * 64)

    # ---- ROUGE-L ----
    rouge = {}
    for tag, items in data.items():
        rouge[tag] = sum(rouge_l(it["generated_article"], it["reference"]) for it in items) / len(items)
    row = "".join(f"{rouge.get(t, float('nan')):>12.4f}" for t in TAGS)
    print(f"{'ROUGE-L(字符)':<20}{row}")

    # ---- 版式垃圾命中率（平均每篇命中模式数）----
    bp = {}
    for tag, items in data.items():
        bp[tag] = sum(boilerplate_hits(it["generated_raw"]) for it in items) / len(items)
    row = "".join(f"{bp.get(t, float('nan')):>12.3f}" for t in TAGS)
    print(f"{'版式垃圾/篇':<20}{row}")

    # ---- 平均文章字数 ----
    length = {}
    for tag, items in data.items():
        length[tag] = sum(len(re.findall(r"[\u4e00-\u9fff]", it["generated_article"])) for it in items) / len(items)
    row = "".join(f"{length.get(t, float('nan')):>12.0f}" for t in TAGS)
    print(f"{'平均汉字数':<20}{row}")

    # ---- LLM 裁判 ----
    if judge:
        print("-" * 64)
        print("Gemini 裁判打分中（每篇约 10s）...")
        dims = ["depth", "evidence", "style", "clean"]
        scores = {tag: {d: [] for d in dims} for tag in data}
        n = min(len(v) for v in data.values())
        for i in range(n):
            for tag, items in data.items():
                topic = items[i]["input"].split("\n")[0][:60]
                try:
                    reply = judge(
                        JUDGE_SYSTEM,
                        JUDGE_TEMPLATE.format(topic=topic, article=items[i]["generated_article"][:6000]),
                        provider="gemini", temperature=0.1, max_tokens=2048,
                    )
                    m = re.search(r"\{.*?\}", reply, re.S)
                    js = json.loads(m.group(0))
                    for d in dims:
                        scores[tag][d].append(float(js.get(d, 0)))
                except Exception as e:
                    print(f"  [WARN] {tag}#{i} 打分失败: {str(e)[:60]}")
            print(f"  裁判进度 {i+1}/{n}")

        print("=" * 64)
        for d in dims:
            avg = {tag: (sum(scores[tag][d]) / len(scores[tag][d]) if scores[tag][d] else float("nan")) for tag in data}
            row = "".join(f"{avg.get(t, float('nan')):>12.2f}" for t in TAGS)
            label = {"depth": "思考深度", "evidence": "论据充分", "style": "文风地道", "clean": "无版式垃圾"}[d]
            print(f"{label:<20}{row}")
        # 总均分
        overall = {}
        for tag in data:
            all_s = [s for d in dims for s in scores[tag][d]]
            overall[tag] = sum(all_s) / len(all_s) if all_s else float("nan")
        row = "".join(f"{overall.get(t, float('nan')):>12.2f}" for t in TAGS)
        print("-" * 64)
        print(f"{'综合均分':<20}{row}")

    print("=" * 64)


if __name__ == "__main__":
    main()
