"""
6-B：公众号深度写作模型 FastAPI 服务

设计要点：
  - lifespan 启动时一次性 4bit 加载模型到显存，所有请求复用（不重复加载）
  - /generate       一次性返回完整文章
  - /generate/stream SSE 流式逐字返回（打字机效果）
  - 单卡 GPU 用 asyncio.Lock 串行化生成，避免并发抢显存 OOM

环境变量：
  MODEL_PATH   合并后模型目录（默认 models/qwen25-7b-gzh-merged）
  LOAD_4BIT    是否 4bit 量化加载（默认 1；显存充裕可设 0 用 bf16）

启动：
  uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from threading import Thread

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/qwen25-7b-gzh-merged")
LOAD_4BIT = os.getenv("LOAD_4BIT", "1") == "1"

DEFAULT_SYSTEM = (
    "你是一位科技领域的深度评论员。请先分析技术趋势背后的产业逻辑与争议点，"
    "再撰写一篇有论据、有观点、不蹭热度的科技评论文章。"
)

# 全局单例：模型/分词器 + 生成锁
STATE: dict = {}
GEN_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载模型，关闭时释放"""
    print(f"[启动] 加载模型: {MODEL_PATH} (4bit={LOAD_4BIT})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
    if LOAD_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()
    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    vram = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[启动] 完成，显存占用 {vram:.2f} GB，服务就绪")
    yield
    STATE.clear()
    torch.cuda.empty_cache()


app = FastAPI(title="公众号深度写作模型 API", version="1.0", lifespan=lifespan)


class WriteRequest(BaseModel):
    topic: str = Field(..., description="文章主题")
    audience: str = Field("关注科技的互联网从业者", description="目标受众")
    requirements: str = Field(
        "2000字左右，有明确论点，引用可验证事实，至少1个反直觉判断，避免营销话术",
        description="写作要求",
    )
    system: str | None = Field(None, description="自定义系统提示；不填用默认科技评论员")
    max_new_tokens: int = Field(3072, ge=256, le=4096)
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)


def build_inputs(req: WriteRequest):
    tokenizer = STATE["tokenizer"]
    user = f"主题：{req.topic}\n受众：{req.audience}\n要求：{req.requirements}"
    messages = [
        {"role": "system", "content": req.system or DEFAULT_SYSTEM},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to("cuda")


@app.get("/health")
async def health():
    ready = "model" in STATE
    vram = torch.cuda.memory_allocated(0) / 1024**3 if ready else 0
    return {"status": "ready" if ready else "loading", "vram_gb": round(vram, 2)}


@app.post("/generate")
async def generate(req: WriteRequest):
    """一次性生成完整文章"""
    async with GEN_LOCK:
        model, tokenizer = STATE["model"], STATE["tokenizer"]
        inputs = build_inputs(req)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"text": gen, "article": _extract_article(gen)}


@app.post("/generate/stream")
async def generate_stream(req: WriteRequest):
    """SSE 流式生成（打字机效果）"""
    async with GEN_LOCK:
        model, tokenizer = STATE["model"], STATE["tokenizer"]
        inputs = build_inputs(req)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        # 生成在子线程跑，主线程从 streamer 逐块取文本
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        def event_stream():
            for token in streamer:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")


def _extract_article(text: str) -> str:
    import re
    m = re.search(r"<article>\n?(.*?)\n?</article>", text, re.S)
    return m.group(1).strip() if m else text.strip()
