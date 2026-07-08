"""
Step 5-B（生成）：用本地模型对验证集批量生成，供后续对比评估

原理：
  同一批 val.jsonl 的 instruction+input，分别喂给三种模型：
    base  —— 纯基座 Qwen2.5-7B（微调前基线）
    v1    —— 未清洗数据训练的适配器
    v2    —— 清洗后数据训练的适配器
  生成结果存 JSON，交给 eval_compare.py 算指标。
  用固定 temperature/seed 保证可复现对比。

用法：
  python -u scripts/gen_local.py --tag base
  python -u scripts/gen_local.py --tag v1 --adapter output/qwen25-7b-gzh-qlora
  python -u scripts/gen_local.py --tag v2 --adapter output/qwen25-7b-gzh-qlora-v2
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
VAL = ROOT / "data" / "final" / "val.jsonl"
OUT_DIR = ROOT / "output" / "eval"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def extract_article(text: str) -> str:
    m = re.search(r"<article>\n?(.*?)\n?</article>", text, re.S)
    return m.group(1).strip() if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="标识：base / v1 / v2")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；base 不填")
    ap.add_argument("--max_new_tokens", type=int, default=3072,
                    help="生成上限；3072 够放 2500 字文章，正常模型会提前停")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"[{args.tag}] 4bit 加载基座...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=bnb_config, device_map="cuda:0", dtype=torch.bfloat16,
    )
    if args.adapter:
        from peft import PeftModel
        print(f"[{args.tag}] 挂载适配器: {args.adapter}")
        model = PeftModel.from_pretrained(model, ROOT / args.adapter)
    model.eval()

    samples = [json.loads(l) for l in VAL.read_text(encoding="utf-8").strip().splitlines()]
    results = []
    torch.manual_seed(42)  # 固定采样，保证可复现

    for i, d in enumerate(samples):
        messages = [
            {"role": "system", "content": d["instruction"]},
            {"role": "user", "content": d["input"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, temperature=0.7, top_p=0.9,
                repetition_penalty=1.05, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "idx": i,
            "input": d["input"],
            "reference": extract_article(d["output"]),
            "generated_raw": gen,
            "generated_article": extract_article(gen),
        })
        print(f"  [{args.tag}] {i+1}/{len(samples)} 生成 {output.shape[1]-inputs['input_ids'].shape[1]} tok, {time.time()-start:.0f}s")

    out_file = OUT_DIR / f"gen_{args.tag}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{args.tag}] 完成，已存 {out_file}")


if __name__ == "__main__":
    main()
