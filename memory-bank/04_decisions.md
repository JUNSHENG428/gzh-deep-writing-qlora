# 关键技术决策记录（追加式，别推翻已定决策）

最后更新：2026-07-07

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
