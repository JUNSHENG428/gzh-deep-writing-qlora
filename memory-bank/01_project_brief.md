# 项目简介（很少变动）

最后更新：2026-07-07

## 一句话
微调 Qwen2.5-7B-Instruct，让它写出「有思考、有深度、有逻辑论证」的中文公众号文章。

## 用户背景
- 数据分析师，Python/SQL 熟练，LLM 微调新手但想学透原理（要求「先教后做」，见 `.cursorrules`）
- 目标转向 AI 基础设施 / MLOps，重视可写进简历的量化成果
- 沟通语言：中文

## 技术路线
- 基座：Qwen2.5-7B-Instruct
- 方法：QLoRA（4-bit 量化 + LoRA），框架 LLaMA-Factory
- 数据：无现成 SFT 数据，用教师模型蒸馏生成带 `<thinking>`（COT 构思）+ `<article>` 的 Alpaca JSONL
- 部署路径（远期）：合并 LoRA → GPTQ 量化 → FastAPI → Docker

## 硬件与环境
- **训练机**：单卡 RTX 5090D，32GB 显存，Linux（所有实际训练在这台机器）
- **当前数据工程机**：Windows（本仓库 `c:\Users\7\Desktop\gzhllm`），venv 在 `venv/`
  - ⚠️ Windows 控制台默认 GBK，跑脚本务必 `python -X utf8 xxx.py`，否则中文乱码
- 显存红线：任何训练配置必须适配单卡 32GB（4-bit、gradient_checkpointing、小 batch + 梯度累积）

## API 资源（Key 在 `.env`，勿提交 Git）
- Gemini（gemini-2.5-pro）、DeepSeek、Kimi/Moonshot、Qwen/DashScope 四家均已配好并连通
- 统一客户端：`scripts/llm_client.py` 的 `chat(system, user, provider=...)`

## 目录结构
- `scripts/` —— 数据管线脚本（详见 `02_pipeline.md`）
- `data/raw/` —— 抓取的公众号原文 md
- `data/cleaned/` —— 清洗后可用文章
- `data/duplicates/` —— 去重时移出的重复文章
- `data/generated/` —— 蒸馏产物、打分结果（jsonl）
- `data/prompts/` —— 蒸馏用 system prompt 模板
- `data/ab_test/` —— 教师模型 A/B 测试原始输出与报告
- `data/selected/` —— （待生成）按打分筛出的精选文章
