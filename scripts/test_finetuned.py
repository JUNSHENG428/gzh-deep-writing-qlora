"""
Step 4 收尾：加载 基座(4bit) + LoRA 适配器，验证微调效果

原理：
  LoRA 训练产物只是 ~80MB 的适配器权重（output/ 下的 adapter_model.safetensors），
  推理时先 4bit 加载基座，再用 peft 把适配器挂上去，
  前向计算变成 W_4bit + (alpha/r) * B @ A。

  用与 scripts/load_model_4bit.py 完全相同的 prompt，
  形成「微调前 vs 微调后」的直接对比。

用法：
  python -u scripts/test_finetuned.py
"""

import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
ADAPTER_DIR = ROOT / "output" / "qwen25-7b-gzh-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("4bit 加载基座...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print(f"挂载 LoRA 适配器: {ADAPTER_DIR.name}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    # 与 Step 3 基座测试完全相同的 prompt，形成前后对比
    messages = [
        {"role": "system", "content": (
            "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
            "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
        )},
        {"role": "user", "content": (
            "主题：AI Agent 会取代 App 吗\n受众：关注科技的互联网从业者\n"
            "要求：先输出 <thinking> 再输出 <article>，观点鲜明、有论据"
        )},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print("生成中（长文约 3-6 分钟）...\n")
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    out_file = ROOT / "output" / "finetuned_sample.txt"
    out_file.write_text(reply, encoding="utf-8")

    print(reply)
    print("-" * 50)
    print(f"生成 {new_tokens} tokens 用时 {gen_time:.1f}s（{new_tokens / gen_time:.1f} tok/s）")
    print(f"完整输出已存: {out_file}")


if __name__ == "__main__":
    main()
