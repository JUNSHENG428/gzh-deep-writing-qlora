# 关键技术决策记录（追加式，别推翻已定决策）

最后更新：2026-07-08

## D1. 微调方案：QLoRA on Qwen2.5-7B-Instruct（项目启动时定）
- 单卡 32GB 唯一现实选择；LLaMA-Factory 作为训练框架
- 理由：Qwen2.5 中文写作强；QLoRA 4-bit 后 7B 训练显存约 12-16GB，留有余量

## D2. 数据策略：以「反向蒸馏」为主（2026-07-07）
- 正向蒸馏（教师直接写文章）质量上限低，改为：拿真实优质公众号文章 → 教师模型反推 input + thinking，output 用原文
- 好处：文章质量有保证（真人写的），教师只负责生成 COT 构思

## D3. 教师模型 A/B 测试结论（2026-07-07，见 data/ab_test/report.md）
- 4 家 × 3 篇，自动打分（格式/构思 5 要点/口吻/字数）：
  - **gemini-2.5-pro 9.83**（最优）> qwen3.7-max 9.67 > kimi-k2.6 9.17 > deepseek-v4-pro 8.67
- deepseek 偶发读后感口吻扣分；qwen thinking 偏短且最慢（44.8s/篇）
- 结论：反向蒸馏主力用 **Gemini**，deepseek 便宜可作补充/对照
- 打分质检用最便宜的 **deepseek-v4-flash**

## D4. 数据质量三重把关（2026-07-07）
1. 规则清洗（clean_raw_articles.py）：删噪声、<1200 字丢弃 —— 1096 raw → 843 cleaned
2. MinHash 去重（dedup_articles.py）：Jaccard>0.7 视为转载 —— 移出 13 篇
3. 模型打分（score_articles.py）：清洗救不了「干净但平庸」的文章（通稿/软文/拼盘），必须模型级判断深度

## D5. 筛选分数线（select_articles.py）
- 核心档：depth≥4 且 genre=深度评论 → 直接入选
- 候补档：depth=3 且 evidence≥4 且 genre=深度评论 → 按 evidence 降序补足到 target（默认 700）
- 访谈即使 depth≥4 也不要（对话体不是目标文体）；软文/拼盘/新闻报道一律不要

## D6. Windows 编码坑（2026-07-07）
- 控制台 GBK 导致中文乱码/写文件报错，所有脚本用 `python -X utf8` 运行

## D7. 批量蒸馏教师改用 Qwen+Kimi 混合，而非 Gemini（2026-07-07，修正 D3）
- A/B 测试 Gemini 分最高，但全量 375 篇考虑限流/成本/稳定性，实际采用：
  超长文（>8000 字）给 kimi（长上下文），其余按文件名哈希 ~70% qwen / 30% kimi
- 每条记录 teacher 字段，保留教师消融实验的可能；Gemini 转做评估裁判（避免既当教练又当裁判）

## D8. 训练直接在 Windows 5090D 本机进行（2026-07-07，修正最初计划）
- 最初设想的 Linux 训练机不存在；实测 Windows + venv + LLaMA-Factory 可行
- 代价：WDDM 显存换页问题 → cutoff_len 从 8192 降到 6144（覆盖 95.5% 样本，见 D9）

## D9. 关键超参定案（2026-07-07，configs/qwen25_7b_qlora_sft.yaml）
- lora_rank=16 / alpha=32（放大系数 2）/ dropout=0.05 / lora_target=all
- cutoff_len=6144（8192 触发 WDDM 换页速度掉 20 倍，实测）
- batch=1 × grad_accum=8（有效 batch 8），lr=1e-4，3 epoch，峰值显存 26GB

## D10. 数据消融：v1（未清洗）vs v2（清洗后）双版本训练（2026-07-07）
- 发现训练集 output 里残留公众号版式噪声（污染率 67%）→ 写 scan/clean_final_boilerplate 清洗到 4.5%
- 故意保留 v1 做对照：v1 综合分 2.09 低于基座 2.44，v2 2.47 反超 → 简历级消融结论
- data/final/backup_v1/ 保留清洗前数据

## D11. 发布策略（2026-07-07/08）
- LoRA 适配器发 HuggingFace（junshengma/qwen2.5-7b-gzh-writing-qlora）
- 代码开源 GitHub（Apache-2.0），爬取语料与派生训练集因版权不入库，只给格式样例
- 合并模型（merge_lora.yaml，CPU 上 fp16 合并）留本地 models/qwen25-7b-gzh-merged 供后续量化部署
