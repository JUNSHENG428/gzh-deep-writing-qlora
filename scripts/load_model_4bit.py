"""
Step 3-B：4bit 量化加载 Qwen2.5-7B-Instruct + 显存测量 + 试推理

原理（QLoRA 的「Q」）：
  磁盘上的 bf16 权重（~15GB）加载时被 bitsandbytes 压成 NF4 4bit（~5GB 显存）。
  NF4 = NormalFloat4：按正态分布分位数设计的 4bit 数据类型，
  比均匀 int4 更贴合神经网络权重的分布，精度损失更小。

关键参数解释：
  load_in_4bit=True                 权重以 4bit 存显存
  bnb_4bit_quant_type="nf4"         用 NF4 而非均匀 fp4
  bnb_4bit_compute_dtype=bfloat16   计算时临时反量化为 bf16
                                    （存储 4bit，算的时候还原精度，兼顾省显存和效果）
  bnb_4bit_use_double_quant=True    二次量化：量化常数本身再量化一次，
                                    每参数再省 ~0.4bit，7B 约省 0.35GB

预期显存（仅推理）：
  模型权重 ~5.2GB + KV cache/激活 ~1-2GB ≈ 6-7GB（训练时另算）

用法：
  python -u scripts/load_model_4bit.py
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-7B-Instruct"


def vram_gb() -> float:
    return torch.cuda.memory_allocated(0) / 1024**3


def main():
    assert torch.cuda.is_available(), "CUDA 不可用"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模型路径: {MODEL_DIR}\n")

    # ---- 4bit 量化配置（QLoRA 训练时也用同一份）----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("4bit 加载模型（首次约 1-3 分钟）...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map="cuda:0",   # 单卡直接放 GPU
        dtype=torch.bfloat16,
    )
    load_time = time.time() - start
    print(f"加载完成: {load_time:.0f}s")
    print(f"模型权重显存: {vram_gb():.2f} GB\n")

    # ---- 试推理：用训练时的同款 prompt 结构 ----
    messages = [
        {"role": "system", "content": (
            "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
            "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
        )},
        {"role": "user", "content": (
            "主题：AI Agent 会取代 App 吗\n受众：关注科技的互联网从业者\n"
            "要求：先输出 <thinking> 再输出 <article>，正文 300 字即可（这只是加载测试）"
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print("生成中（微调前的基座水平，仅验证推理可用）...")
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("-" * 50)
    print(reply[:800])
    print("-" * 50)
    print(f"\n生成 {new_tokens} tokens 用时 {gen_time:.1f}s（{new_tokens / gen_time:.1f} tok/s）")
    print(f"峰值显存: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    print("\n[OK] 4bit 加载与推理验证通过，可进入 Step 4 训练配置")


if __name__ == "__main__":
    main()
