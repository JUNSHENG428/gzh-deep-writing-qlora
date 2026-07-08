# 当前进度与下一步（最常更新，动手前先读这份）

最后更新：2026-07-07 17:30（UTC+8）

## 当前状态快照
- 数据量：`data/raw` 1096 篇 → 清洗后 `data/cleaned` 843 篇 → 去重移出 13 篇
- **打分进行中**：`score_articles.py` 正在终端后台跑（17:16 启动，命令
  `venv\Scripts\python.exe -X utf8 scripts\score_articles.py`），
  `article_scores.jsonl` 已写入 367/843 条，支持断点续跑，中断了直接重跑同一命令即可
- 已打分部分的分布（趋势参考）：
  - genre：深度评论 202 / 拼盘杂烩 79 / 软文广告 66 / 新闻报道 13 / 访谈 7
  - depth：4 分 151 / 3 分 89 / 2 分 69 / 1 分 56 / 5 分 2
  - → 按此比例，843 篇里「深度评论且 depth≥4」预计 300-400 篇，target 700 可能不够，
    届时要么降档补足、要么接受更小但更精的数据集
- 反向蒸馏：仅试跑（gemini 3 篇、deepseek 3 篇），**全量还没跑**
- `data/selected/` 尚不存在（select_articles.py 还没运行）
- 训练侧：还没开始。LLaMA-Factory / 训练机环境未动工（check_env.py 要在 Linux 5090D 机器上跑）

## 已完成 ✅
1. 四厂商 API 打通（llm_client.py + test_providers.py）
2. 手工样本 manual_001 组装、Gemini 单条正向蒸馏验证
3. 教师模型 A/B 测试 → 选定 Gemini 做反向蒸馏主力（见 04_decisions D3）
4. 全量文章清洗（含多轮残留模式修补：评论区、微信链接、图片、JS 链接等）
5. 去重完成
6. 反向蒸馏脚本 + 断点续跑机制验证通过

## 下一步（按顺序）
1. **等打分跑完**（843 篇，剩约 476 篇；deepseek-v4-flash 6 并发）。
   完成后用 `show_scores.py` 看分布
2. 跑 `select_articles.py --target 700`（若核心档不够 700，先看 selection_report.txt 再决定降不降档）
3. 对 `data/selected/` 全量反向蒸馏：
   `python -X utf8 scripts\reverse_distill.py --provider gemini --limit 0`
   （注意 Gemini 免费额度/限流，量大可分天跑或 deepseek 分担一部分）
4. 蒸馏产物质检 + 合并成最终 SFT 训练集（Alpaca JSONL）
5. 转战 Linux 训练机：跑 `check_env.py`，装 LLaMA-Factory，写 QLoRA 配置
   （4-bit、gradient_checkpointing、paged_adamw_8bit、小 batch + 梯度累积，显存预估 <32GB）
6. 训练 → 评估（微调前后 ROUGE + 人工评分对比，留简历素材）

## 注意事项 / 坑
- Windows 跑脚本必须 `-X utf8`，否则 GBK 乱码
- 打分/蒸馏都是逐条追加 + manifest 断点续跑，重跑不会浪费 token
- 不要在 Windows/Mac 上跑实际训练，只做数据工程
- 用户要求「先教后做」：给方案前先解释原理和取舍（见 .cursorrules）
