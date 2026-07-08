"""
多厂商 API 连通性测试

对每个已配置 Key 的厂商发送一条极小的测试消息（消耗 <100 tokens），
验证 Key 有效、模型名正确、网络可达，并报告延迟。

用法：
  python scripts/test_providers.py
"""

import time

from llm_client import PROVIDERS, available_providers, chat, get_model


def main():
    configured = available_providers()
    missing = [p for p in PROVIDERS if p not in configured]

    print("=" * 55)
    print("厂商 API 连通性测试")
    print("=" * 55)
    if missing:
        print(f"未配置 Key（跳过）: {', '.join(missing)}")
    print()

    results = []
    for provider in configured:
        model = get_model(provider)
        print(f"[{provider}] 模型: {model} ... ", end="", flush=True)
        start = time.time()
        try:
            reply = chat(
                system="你是测试助手。",
                user="请只回复两个字：正常",
                provider=provider,
                temperature=0.1,
                # 注意：思考型模型（如 gemini-2.5-pro）会先消耗内部思考 token，
                # 上限太小会导致正文为空，故给足余量
                max_tokens=2048,
            )
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}s) 回复: {reply[:20]}")
            results.append((provider, model, "OK", f"{elapsed:.1f}s"))
        except Exception as e:
            print(f"FAIL: {str(e)[:120]}")
            results.append((provider, model, "FAIL", str(e)[:60]))

    print()
    print("=" * 55)
    print("汇总")
    print("=" * 55)
    for provider, model, status, info in results:
        mark = "[OK]  " if status == "OK" else "[FAIL]"
        print(f"{mark} {provider:<10} {model:<25} {info}")
    ok_count = sum(1 for r in results if r[2] == "OK")
    print(f"\n可用厂商: {ok_count}/{len(configured)}（另有 {len(missing)} 家未配置 Key）")


if __name__ == "__main__":
    main()
