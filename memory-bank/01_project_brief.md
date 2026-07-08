# 项目简介（很少变动）

最后更新：2026-07-08

## 一句话
微调 Qwen2.5-7B-Instruct，让它写出「有思考、有深度、有逻辑论证」的中文公众号文章。
**核心训练已完成**，LoRA 适配器已发布：https://huggingface.co/junshengma/qwen2.5-7b-gzh-writing-qlora

## 用户背景
- 数据分析师，Python/SQL 熟练，LLM 微调新手但想学透原理（要求「先教后做」，见 `.cursorrules`）
- 目标转向 AI 基础设施 / MLOps，重视可写进简历的量化成果
- 沟通语言：中文

## 技术路线
- 基座：Qwen2.5-7B-Instruct（modelscope 下载到 `models/Qwen2.5-7B-Instruct`）
- 方法：QLoRA（4-bit NF4 + LoRA rank=16, alpha=32, target=all），框架 LLaMA-Factory（本地克隆 `LLaMA-Factory/`）
- 数据：反向蒸馏——真实优质公众号文章反推 input + `<thinking>`（COT 构思），output=`<thinking>`+`<article>原文</article>`，Alpaca JSONL
- 多模型分工：DeepSeek 打分质检、Qwen/Kimi 批量蒸馏教师、Gemini 评估裁判
- 部署路径（远期）：合并 LoRA（已完成）→ GPTQ 量化 → FastAPI → Docker

## 硬件与环境（⚠️ 与最初设想不同，已修正）
- **训练就在本 Windows 机器上**：RTX 5090D（Blackwell sm_120），32GB 显存，Windows + venv
  - 最初计划的「Linux 训练机」不存在，全流程都在这台 Windows 机器完成
- Windows 训练/推理前必须设：`$env:PYTHONUTF8="1"; $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`
- 显存实测：QLoRA 训练峰值 26GB；cutoff_len=8192 会触发 WDDM 换页速度掉 20 倍，定为 6144
- bitsandbytes 需 0.45.3+ 适配 Blackwell；PyTorch 用 cu128 版

## API 资源（Key 在 `.env`，勿提交 Git）
- Gemini（gemini-2.5-pro）、DeepSeek、Kimi/Moonshot、Qwen/DashScope 四家均配好
- 统一客户端：`scripts/llm_client.py` 的 `chat(system, user, provider=...)`

## 目录结构
- `scripts/` —— 全流程脚本（详见 `02_pipeline.md`）
- `configs/` —— `qwen25_7b_qlora_sft.yaml`（训练）、`merge_lora.yaml`（合并导出）
- `data/raw|cleaned|duplicates|selected/` —— 原文 1096 → 清洗 843 → 精选 375
- `data/generated/` —— 蒸馏产物、打分结果
- `data/final/` —— 最终训练集：train.jsonl 358 条 / val.jsonl 18 条 + dataset_info.json（backup_v1/ 是清洗前旧版）
- `models/` —— 基座 + `qwen25-7b-gzh-merged`（合并后完整模型 ~15GB）
- `output/` —— 训练产物：`qwen25-7b-gzh-qlora`（v1 未清洗数据）、`qwen25-7b-gzh-qlora-v2`（v2 清洗后）、`eval/`（三方生成结果）
- `LLaMA-Factory/` —— 训练框架源码
- 已建 Git 仓库并推送 GitHub（origin/main），公开仓库不含爬取语料（版权）
