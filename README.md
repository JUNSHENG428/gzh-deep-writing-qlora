# 中文公众号深度写作模型（Qwen2.5-7B + QLoRA）

在**单卡 RTX 5090D（32GB）** 上，用 QLoRA 微调 Qwen2.5-7B-Instruct，训练一个能写"有思考、有深度、有论证"的中文公众号深度长文的写作模型。

本仓库是完整的**端到端微调工程**：数据爬取后处理 → 多模型蒸馏 → 训练 → 评估 → 数据消融迭代。

> 微调后的 LoRA 适配器已发布至 HuggingFace：**https://huggingface.co/junshengma/qwen2.5-7b-gzh-writing-qlora**

---

## 亮点成果

| 指标 | 基座 Qwen2.5-7B | v1（未清洗数据） | **v2（清洗后数据）** |
|---|---|---|---|
| 综合质量分（Gemini 裁判 1-5） | 2.44 | 2.09 | **2.47** |
| 无版式垃圾（1-5，越高越干净） | 5.00 | 3.29 | **4.81** |
| 版式垃圾数 / 篇（越低越好） | 0.06 | 1.33 | **0.11** |
| 论据充分度（1-5） | 1.38 | 1.53 | **1.62** |
| 平均正文字数 | 1752 | 2716 | 2878 |

**核心发现（数据消融实验）**：同样的超参与训练流程下，未清洗数据训练的 v1 因学入版式噪声，综合分反而**低于基座**（2.09 < 2.44）；清洗训练数据后（版式噪声污染率 67% → 4.5%），v2 综合分回升并**反超基座**。这直观印证了 "garbage in, garbage out"——数据质量对小样本 SFT 的决定性影响。

## 技术栈

- **基座模型**：Qwen2.5-7B-Instruct
- **微调方法**：QLoRA（4-bit NF4 量化 + LoRA rank=16），单卡 32GB 可行
- **训练框架**：LLaMA-Factory
- **数据策略**：逆向蒸馏（Reverse Distillation）——用真实优质文章反推出"任务指令 + 规划式思考链（CoT）"，再让模型学"从思考到成文"
- **多模型协作**：DeepSeek 打分筛选、Qwen/Kimi 做蒸馏教师、Gemini 当评估裁判

## 微调前后 / 数据消融对比

同一 prompt（"AI Agent 会取代 App 吗"）下：
- **基座**：思考仅 3 句空泛提纲，正文"首先/其次/最后"车轱辘话
- **v1**：学会了公众号篇幅与结构，但夹带"作者：XX / 封面来源 / 点个小爱心"等版式垃圾
- **v2**：结构与深度保留，版式垃圾基本消除

## 项目结构

```
gzhllm/
├── scripts/                    # 全流程脚本
│   ├── check_env.py            # 环境自检（PyTorch/CUDA/bitsandbytes）
│   ├── clean_raw_articles.py   # 原始文章清洗（去 boilerplate/图片/评论）
│   ├── dedup_articles.py       # 标题 + MinHash 内容去重
│   ├── score_articles.py       # DeepSeek 质量打分（深度/论据/体裁）
│   ├── select_articles.py      # 按质量分筛选
│   ├── llm_client.py           # 统一 4 厂商 LLM 客户端
│   ├── reverse_distill.py      # 逆向蒸馏（单篇）
│   ├── batch_reverse_distill.py# 多教师批量蒸馏
│   ├── merge_and_split.py      # 合并/质量闸门/训练验证集切分
│   ├── measure_tokens.py       # 真实分词器测 token 长度
│   ├── load_model_4bit.py      # 4bit 加载 + 显存/推理验证
│   ├── scan_boilerplate.py     # 训练集版式噪声扫描（诊断）
│   ├── clean_final_boilerplate.py # 训练集版式噪声清洗
│   ├── gen_local.py            # 本地模型批量生成（评估用）
│   └── eval_compare.py         # 三方对比评估（ROUGE-L + 版式 + LLM 裁判）
├── configs/
│   └── qwen25_7b_qlora_sft.yaml# QLoRA 训练配置（超参含中文注释）
├── data/
│   ├── prompts/                # 蒸馏 system/user 提示词模板
│   └── examples/sample.jsonl   # 数据格式脱敏样例
├── requirements.txt
└── README.md
```

> 注：爬取的公众号原文与派生训练集因第三方版权原因**未包含**在本仓库，仅提供数据格式样例与处理脚本。

## 快速开始

```bash
# 1. 环境（Blackwell 架构需 CUDA 12.8 版 PyTorch）
python -m venv venv && venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/check_env.py            # 自检

# 2. 下载基座
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models/Qwen2.5-7B-Instruct

# 3. 准备数据（需自备语料，格式见 data/examples/sample.jsonl）
#    配置 API Key: cp .env.example .env  然后填入各厂商 Key
python scripts/clean_raw_articles.py
python scripts/dedup_articles.py
python scripts/score_articles.py
python scripts/select_articles.py
python scripts/batch_reverse_distill.py
python scripts/merge_and_split.py

# 4. 训练（Windows 需设两个环境变量，详见配置文件头注释）
git clone https://github.com/hiyouga/LLaMA-Factory.git && pip install -e LLaMA-Factory
$env:PYTHONUTF8="1"; $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
llamafactory-cli train configs/qwen25_7b_qlora_sft.yaml

# 5. 评估（微调前后 + 数据消融对比）
python scripts/gen_local.py --tag base
python scripts/gen_local.py --tag v2 --adapter output/qwen25-7b-gzh-qlora-v2
python scripts/eval_compare.py --judge
```

## 部署为 API 服务

微调后的模型可合并、量化并封装成 HTTP 服务（详见 [`serve/README.md`](serve/README.md)）：

```bash
# 1. 合并 LoRA 到基座
llamafactory-cli export configs/merge_lora.yaml

# 2a. 本地起服务（4bit 加载，约 6GB 显存）
uvicorn serve.app:app --host 127.0.0.1 --port 8010

# 2b. 或 Docker 一键部署（需 nvidia-container-toolkit）
docker compose up -d --build

# 3. 调用
curl -X POST http://localhost:8010/generate -H "Content-Type: application/json" \
  -d '{"topic":"AI Agent 会取代 App 吗","max_new_tokens":3072}'
```

提供 `/generate`（一次性）与 `/generate/stream`（SSE 流式）两个接口。

## 数据格式（Alpaca JSONL）

```json
{
  "instruction": "你是一位科技领域的深度评论员……",
  "input": "主题：……\n受众：……\n要求：……",
  "output": "<thinking>规划式思考链……</thinking>\n\n<article>正文……</article>"
}
```

## 工程实战记录

- **单卡 32GB 显存管理**：4bit 量化 + gradient_checkpointing + paged_adamw_8bit + 梯度累积，训练峰值 26GB
- **cutoff_len 取舍**：实测 8192 会打满显存触发 Windows WDDM 换页（速度掉 20 倍），定为 6144（覆盖 95.5% 样本）
- **Windows 踩坑**：PowerShell GBK 编码需 `PYTHONUTF8=1`；`bitsandbytes` 需 0.45.3+ 适配 Blackwell

## 局限与后续

- 7B 模型存在事实幻觉，写作输出需人工核查事实
- 思考深度/文风绝对分仍偏低（受限于 358 条小数据、rank=16、3 epoch）
- 提升路径：扩充数据量 > 调 rank/epoch > 更强基座

## License

代码采用 Apache-2.0。基座模型遵循 [Qwen2.5 License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。
