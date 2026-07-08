# 数据管线与脚本对照（脚本变动时更新）

最后更新：2026-07-07

所有脚本在 `scripts/`，运行前激活 venv，并加 `-X utf8`：
`C:\Users\7\Desktop\gzhllm\venv\Scripts\python.exe -X utf8 scripts\xxx.py`

## 管线总览

```
data/raw/*.md（抓取的公众号原文）
  → [2-D-1] clean_raw_articles.py  清洗 → data/cleaned/
  → [2-E-1] dedup_articles.py      去重（MinHash Jaccard>0.7）→ 重复移入 data/duplicates/
  → [2-E-2] score_articles.py      质量打分（deepseek-v4-flash，depth/evidence/genre/topic）
                                    → data/generated/article_scores.jsonl（断点续跑）
  → [2-E-3] select_articles.py     按分筛精选 → data/selected/ + selection_report.txt
  → [2-D-2] reverse_distill.py     反向蒸馏：真文章 → 反推 input+thinking
                                    → data/generated/reverse_distilled_{provider}.jsonl
                                    （每厂商独立 manifest 断点续跑）
  → SFT 训练数据（Alpaca JSONL，output = <thinking>…</thinking><article>原文</article>）
```

## 各脚本速查

| 脚本 | 阶段 | 作用 |
|------|------|------|
| `check_env.py` | Step 1 | 训练机 GPU/CUDA/PyTorch 检查（5090D sm_120 专用，在 Linux 训练机跑） |
| `assemble_manual_001.py` | 2-B | 手写 thinking+article 组装为第一条 Alpaca 样本 |
| `distill_one_gemini.py` | 2-C | Gemini 正向蒸馏单条试跑（已验证可行） |
| `clean_raw_articles.py` | 2-D-1 | 清洗 raw→cleaned：删图片/链接/日期/推荐尾巴，<1200 字丢弃 |
| `reverse_distill.py` | 2-D-2 | 反向蒸馏主脚本，`--provider gemini` `--limit 0`（全部） |
| `dedup_articles.py` | 2-E-1 | 内容去重，保留字数最多的版本 |
| `score_articles.py` | 2-E-2 | 批量质检打分（6 并发，只发前 3000 字省 token） |
| `select_articles.py` | 2-E-3 | 筛选：depth≥4 且 genre=深度评论 直接入选；depth=3 且 evidence≥4 补足；`--target 700` |
| `show_scores.py` | 辅助 | 查看打分分布 |
| `llm_client.py` | 基础 | 统一 4 厂商 chat() 接口 |
| `test_providers.py` | 辅助 | 4 家 API 连通性测试 |
| `ab_test_providers.py` | 辅助 | 教师模型 A/B 对比（已完成，见 04_decisions） |

## 数据格式约定
- Alpaca JSONL，固定 instruction：「你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，再撰写一篇有论据、有观点、不蹭热度的科技评论文章。」
- output 结构：`<thinking>动笔前构思（第一人称"打算/准备"口吻，300-500 字，5 要点：核心矛盾/读者痛点/文章结构/语气定位/关键素材）</thinking>` + `<article>文章正文</article>`
- 反向蒸馏中 article 必须是原文一字不改；thinking 若出现读后感口吻（"本文/这篇文章"等）会被质量过滤
