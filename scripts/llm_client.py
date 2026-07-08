"""
统一 LLM 客户端：一个 chat() 函数调用 4 家厂商

原理：
  DeepSeek / Kimi / Qwen 都兼容 OpenAI API 格式，只需换 base_url 和模型名；
  Gemini 用 google-genai SDK，单独封装成相同接口。

用法：
  from llm_client import chat, available_providers
  text = chat("你是编辑", "写一句话", provider="deepseek")
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 各厂商配置：key/model 从 .env 读取，base_url 固定
PROVIDERS = {
    "deepseek": {
        # 默认官方 API；用阿里云 MaaS 网关时在 .env 配 DEEPSEEK_BASE_URL 覆盖
        "base_url": "https://api.deepseek.com",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "base_url_env": "KIMI_BASE_URL",
        "key_env": "KIMI_API_KEY",
        "model_env": "KIMI_MODEL",
        "default_model": "kimi-latest",
    },
    "qwen": {
        # 默认公共 DashScope；若用阿里云 MaaS 专属端点，在 .env 配 QWEN_BASE_URL 覆盖
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "QWEN_BASE_URL",
        "key_env": "DASHSCOPE_API_KEY",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen-max",
    },
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-pro",
    },
}

# 可重试的临时性错误关键字
RETRYABLE = ("503", "UNAVAILABLE", "429", "rate", "timeout", "Timeout")


def _get_key(provider: str) -> str | None:
    """读取厂商 Key；未配置或仍是占位符则返回 None"""
    key = os.getenv(PROVIDERS[provider]["key_env"], "")
    if not key or key == "your-api-key-here":
        return None
    return key


def get_model(provider: str) -> str:
    """返回该厂商实际使用的模型名（.env 可覆盖默认值）"""
    cfg = PROVIDERS[provider]
    return os.getenv(cfg["model_env"], "") or cfg["default_model"]


def available_providers() -> list[str]:
    """返回已配置 Key 的厂商列表"""
    return [p for p in PROVIDERS if _get_key(p)]


def _chat_openai_compatible(
    provider: str, system: str, user: str,
    temperature: float, max_tokens: int,
    model: str | None = None,
) -> str:
    """DeepSeek / Kimi / Qwen：走 OpenAI 兼容接口"""
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    base_url = os.getenv(cfg.get("base_url_env", ""), "") or cfg["base_url"]
    client = OpenAI(api_key=_get_key(provider), base_url=base_url)

    use_model = model or get_model(provider)
    # Moonshot 官方的 kimi-k2 系列只接受 temperature=1
    if provider == "kimi" and use_model.startswith("kimi-k2"):
        temperature = 1.0

    response = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError(f"{provider} 返回空内容")
    return text.strip()


def _chat_gemini(system: str, user: str, temperature: float, max_tokens: int) -> str:
    """Gemini：google-genai SDK"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_get_key("gemini"))
    response = client.models.generate_content(
        model=get_model("gemini"),
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    text = response.text
    if not text and response.candidates:
        parts = response.candidates[0].content.parts or []
        text = "".join(getattr(p, "text", "") or "" for p in parts)
    if not text:
        raise RuntimeError("gemini 返回空内容")
    return text.strip()


def chat(
    system: str,
    user: str,
    provider: str = "deepseek",
    temperature: float = 0.4,
    max_tokens: int = 4096,
    retries: int = 3,
    model: str | None = None,
) -> str:
    """
    统一入口：向指定厂商发送 system + user 消息，返回文本

    参数：
      provider:    deepseek / kimi / qwen / gemini
      temperature: 反推提炼任务建议 0.4，创作任务 0.6
      max_tokens:  输出上限；反向蒸馏 4096 够用，正向写文章需 16384
      retries:     临时性错误（503/429/超时）自动重试次数
      model:       覆盖 .env 中的默认模型（如打分任务用便宜的 flash 版）
    """
    if provider not in PROVIDERS:
        raise ValueError(f"未知厂商: {provider}，可选: {list(PROVIDERS)}")
    if not _get_key(provider):
        raise RuntimeError(f"{provider} 未配置 Key，请在 .env 中填写 {PROVIDERS[provider]['key_env']}")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if provider == "gemini":
                return _chat_gemini(system, user, temperature, max_tokens)
            return _chat_openai_compatible(provider, system, user, temperature, max_tokens, model)
        except Exception as e:
            last_error = e
            if any(k in str(e) for k in RETRYABLE) and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  [WARN] {provider} 临时错误，{wait}s 后重试 ({attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    raise last_error  # type: ignore[misc]
