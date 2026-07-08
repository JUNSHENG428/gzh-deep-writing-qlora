# 数据管线与脚本对照（脚本变动时更新）

最后更新：2026-07-08

所有脚本在 `scripts/`，运行前激活 venv 并设 `$env:PYTHONUTF8="1"`（或 `python -X utf8`）。

## 管线总览（全部已跑通）

```
data/raw/*.md（1096 篇公众号原文）
  → [2-D-1] clean_raw_articles.py     清洗 → data/cleaned/（843）
  → [2-E-1] dedup_articles.py         去重 → 13 篇移入 data/duplicates/
  → [2-E-2] score_articles.py         DeepSeek flash 打分 → article_scores.jsonl（843 全量完成）
  → [2-E-3] select_articles.py        筛选 → data/selected/（375 篇）
  → [2-E-4] batch_reverse_distill.py  批量反向蒸馏（~70% qwen + 30% kimi，带 teacher 字段）
                                       → reverse_distilled_selected.jsonl（375 条）
  → [2-F]   merge_and_split.py        合并所有数据源 + 质量闸门 + 去重
                                       → data/final/train.jsonl(358) + val.jsonl(18)
  → [5-A]   scan_boilerplate.py       诊断训练集残留版式噪声
            clean_final_boilerplate.py 清洗 data/final（旧版备份到 backup_v1/）→ v2 数据集
  → [4]     llamafactory-cli train configs/qwen25_7b_qlora_sft.yaml  → output/…-v2
  → [5-B]   gen_local.py --tag base|v1|v2   本地批量生成（val 集 prompt）
            eval_compare.py --judge          ROUGE-L + 版式噪声 + Gemini 裁判
  → [6-A]   llamafactory-cli export configs/merge_lora.yaml → models/qwen25-7b-gzh-merged
```

## 各脚本速查

| 脚本 | 阶段 | 作用 |
|------|------|------|
| `check_env.py` | 1 | GPU/CUDA/PyTorch/bitsandbytes 自检（5090D sm_120） |
| `clean_raw_articles.py` | 2-D-1 | raw→cleaned：删图片/链接/推荐尾巴，<1200 字丢弃 |
| `dedup_articles.py` | 2-E-1 | MinHash Jaccard>0.7 去重 |
| `score_articles.py` | 2-E-2 | deepseek-v4-flash 打分 depth/evidence/genre/topic，断点续跑 |
| `select_articles.py` | 2-E-3 | depth≥4 深度评论直入 + depth=3 evidence≥4 候补 |
| `reverse_distill.py` | 2-D-2 | 反向蒸馏单厂商版（早期，已被 batch 版取代） |
| `batch_reverse_distill.py` | 2-E-4 | 批量蒸馏：超长文给 kimi，其余按 hash 70% qwen/30% kimi；4 线程；manifest 断点 |
| `merge_and_split.py` | 2-F | 合并 5 个数据源、按 article 指纹去重、质量闸门、切 train/val |
| `measure_tokens.py` | 3-C | 用 Qwen 真实分词器测数据集 token 长度（定 cutoff_len 依据） |
| `load_model_4bit.py` | 3-B | 4bit NF4 加载基座 + 显存/推理验证 |
| `test_finetuned.py` | 4 | 基座 4bit + LoRA 适配器挂载，微调前后同 prompt 对比 |
| `scan_boilerplate.py` | 5-A | 扫描 data/final 残留版式噪声（诊断） |
| `clean_final_boilerplate.py` | 5-A | 清洗训练集版式噪声（行级删除+尾部截断+字数闸门） |
| `gen_local.py` | 5-B | base/v1/v2 三方在 val 集上批量生成，固定 temperature/seed |
| `eval_compare.py` | 5-B | ROUGE-L(字符) + 版式噪声统计 + Gemini 裁判 4 维打分 |
| `llm_client.py` | 基础 | 统一 4 厂商 chat() 接口 |
| `test_providers.py` / `ab_test_providers.py` | 辅助 | API 连通性 / 教师 A/B 测试 |
| `show_scores.py` | 辅助 | 打分分布查看 |
| `assemble_manual_001.py` / `distill_one_gemini.py` | 早期 | 手工样本 / Gemini 单条试跑 |

## 数据格式约定
- Alpaca JSONL，固定 instruction：「你是一位科技领域的深度评论员……」
- output：`<thinking>`第一人称"打算/准备"口吻构思（5 要点：核心矛盾/读者痛点/文章结构/语气定位/关键素材）`</thinking>` + `<article>`原文一字不改`</article>`
- 蒸馏记录带 `teacher` 字段，支持后续"哪家教师效果好"消融
- `data/final/dataset_info.json` 注册了 `gzh_writing_train` / `gzh_writing_val` 供 LLaMA-Factory 使用
