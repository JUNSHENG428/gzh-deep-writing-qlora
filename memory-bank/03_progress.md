# 当前进度与下一步（最常更新，动手前先读这份）

最后更新：2026-07-08 10:20（UTC+8）

## 大局：核心闭环已完成 ✅
数据工程 → 蒸馏 → QLoRA 训练（v1/v2 两版）→ 三方评估 → LoRA 合并导出 → 发布 HuggingFace + GitHub，全部走通。
项目进入「收尾/迭代」阶段。

## 关键成果
- 数据：1096 raw → 843 cleaned → 375 selected → **train 358 / val 18**（data/final/）
- 训练两版（各 3 epoch，~35-42 分钟，峰值显存 26GB）：
  - v1 = 未清洗数据（`output/qwen25-7b-gzh-qlora`），eval_loss 2.546
  - v2 = 清洗版式噪声后（`output/qwen25-7b-gzh-qlora-v2`），eval_loss 2.574
- 评估（eval_compare.py --judge，Gemini 裁判）：
  | 指标 | base | v1 | v2 |
  |---|---|---|---|
  | 综合质量分(1-5) | 2.44 | 2.09 | **2.47** |
  | 版式垃圾数/篇 | 0.06 | 1.33 | **0.11** |
  | 论据充分度 | 1.38 | 1.53 | **1.62** |
  | 平均正文字数 | 1752 | 2716 | 2878 |
- **核心发现（简历亮点）**：数据消融实验证明 garbage in garbage out——v1 学入版式噪声综合分反低于基座，清洗后 v2 反超基座
- LoRA 已合并导出：`models/qwen25-7b-gzh-merged`（~15GB）
- 已发布：HuggingFace 适配器 junshengma/qwen2.5-7b-gzh-writing-qlora；GitHub 仓库已推送（2 commits），README 完整

## 昨天（07-07 晚）~ 今早动态
- 打分 843 篇全量完成；筛出 375 篇（核心档 362 + 候补 13，字数中位 3113）
- batch_reverse_distill 375 篇全部蒸馏完成（qwen 为主 + kimi 超长文）
- 训练、评估、合并、发布均已完成
- 今早（07-08）在装 GitHub CLI 并做 `gh auth login`（网页授权流程，可能未完成——如果用户提 gh/GitHub 相关需求，先 `gh auth status` 确认）

## 未提交的改动
- `configs/merge_lora.yaml` 还没 commit（git status 显示 untracked）

## 可能的下一步（等用户定，勿擅自开工）
1. 部署链路收尾：merged 模型 → GPTQ/AWQ 量化 → FastAPI 封装 → Docker（.cursorrules 里的既定远期路径，还没做）
2. 质量提升迭代：扩充数据量（>358 条）、调 lora_rank/epoch、教师消融（利用 teacher 字段）
3. GitHub 仓库完善（gh CLI 刚装，可能想做 release / PR 流程）
4. 提交 `configs/merge_lora.yaml`

## 注意事项 / 坑（历史教训）
- Windows 必设 `PYTHONUTF8=1`，训练/推理加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- cutoff_len 用 6144，8192 会 WDDM 换页速度掉 20 倍
- LoRA 合并必须在 fp16/bf16 下做（merge_lora.yaml 用 CPU 合并），不能在 4bit 上合并
- Gemini 裁判偶发空响应/限流，eval_compare 有 WARN 跳过逻辑
- 7B 模型有事实幻觉；思考深度绝对分仍低（1.75/5），受限于小数据+rank16
- 用户要求「先教后做」：给方案前先解释原理和取舍（见 .cursorrules）
