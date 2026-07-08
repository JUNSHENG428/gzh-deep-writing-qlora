"""
Step 1 环境检查脚本（RTX 5090D / Blackwell 专用）

用途：
  1. 确认 GPU 驱动、CUDA、PyTorch 是否正常
  2. 检测 sm_120（Blackwell）是否被 PyTorch 识别
  3. 粗估 QLoRA 训练显存是否够用

在【装有 5090D 的 Windows 机器】上运行：
  python scripts/check_env.py
"""

import sys


def check_python_version():
    """Python 3.10+ 与新版 PyTorch 生态兼容性更好"""
    print("=" * 55)
    print("1. Python 版本")
    print("=" * 55)
    major, minor = sys.version_info[:2]
    print(f"当前版本: Python {major}.{minor}.{sys.version_info.micro}")
    if major < 3 or (major == 3 and minor < 10):
        print("⚠️  建议升级到 Python 3.10 或 3.11（5090D + 新 PyTorch 更稳）")
    else:
        print("✅ Python 版本 OK")


def check_pytorch_and_cuda():
    """PyTorch 必须带 cu128+ 才能跑 Blackwell（sm_120）"""
    print("\n" + "=" * 55)
    print("2. PyTorch 与 CUDA")
    print("=" * 55)

    try:
        import torch
    except ImportError:
        print("❌ 未安装 PyTorch，请先执行安装命令（见下方说明）")
        return False

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"PyTorch 内置 CUDA: {torch.version.cuda}")

    # 5090D 关键：不能用 cu124 的旧轮子
    cuda_ver = torch.version.cuda or ""
    if cuda_ver and float(cuda_ver[:3]) < 12.8:
        print("⚠️  CUDA 版本 < 12.8，5090D（sm_120）可能无法运行！")
        print("   请重装: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    else:
        print("✅ CUDA 版本满足 Blackwell 要求（≥ 12.8）")

    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() = False")
        print("   常见原因：装了 CPU 版 PyTorch / 驱动未装 / 版本不匹配")
        return False

    print("✅ CUDA 可用")
    return True


def check_gpu_details():
    """读取 GPU 型号、显存、计算能力"""
    import torch

    print("\n" + "=" * 55)
    print("3. GPU 信息（5090D 应为 ~32GB，计算能力 12.0）")
    print("=" * 55)

    device_count = torch.cuda.device_count()
    print(f"检测到 GPU 数量: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"\n  GPU {i}: {props.name}")
        print(f"  总显存: {total_gb:.1f} GB")
        print(f"  计算能力: sm_{props.major}{props.minor}")

        # Blackwell = sm_120（major=12, minor=0）
        if props.major == 12:
            print("  ✅ Blackwell 架构已识别（sm_120）")
        elif "5090" in props.name:
            print("  ⚠️  型号是 5090 但计算能力不是 12.x，PyTorch 轮子可能不对")

        if total_gb >= 30:
            print("  ✅ 显存充足，适合 7B QLoRA 微调")
        elif total_gb >= 20:
            print("  ⚠️  显存偏紧，需控制 batch_size 和 max_seq_length")
        else:
            print("  ❌ 显存不足，无法做 7B QLoRA 训练")


def check_sm120_kernel():
    """
    实际跑一次 GPU 矩阵乘法，验证 sm_120 内核能执行。
    若 PyTorch 不含 sm_120 编译目标，这里会直接报错。
    """
    import torch

    print("\n" + "=" * 55)
    print("4. sm_120 内核实测（最关键一步）")
    print("=" * 55)

    try:
        torch.cuda.empty_cache()
        before_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)

        # 在 GPU 上做一次矩阵乘法（显存增量约 12 MB，可忽略）
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()

        after_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"矩阵运算成功，显存占用: {after_mb:.1f} MB（增量 {after_mb - before_mb:.1f} MB）")
        print("✅ sm_120 GPU 计算正常")

        del x, y
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ GPU 计算失败: {e}")
        print("   典型原因：PyTorch 轮子不含 sm_120，需重装 cu128 版本")
        return False


def check_bitsandbytes():
    """QLoRA 依赖 bitsandbytes 做 4-bit 量化"""
    print("\n" + "=" * 55)
    print("5. bitsandbytes（QLoRA 4-bit 量化）")
    print("=" * 55)

    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 版本: {bnb.__version__}")

        # 尝试加载 CUDA 库（不加载模型，开销极小）
        import bitsandbytes.cextension as ce
        lib = ce.lib
        print(f"加载的 CUDA 库: {getattr(lib, 'name', '已加载')}")
        print("✅ bitsandbytes 导入成功")
        print("   建议版本 ≥ 0.45.3（含 sm_120 支持）")
        return True
    except Exception as e:
        print(f"❌ bitsandbytes 失败: {e}")
        print("   QLoRA 无法进行 4-bit 量化，需修复后再训练")
        return False


def estimate_qlora_vram():
    """根据 5090D 32GB 给出 QLoRA 显存预估"""
    print("\n" + "=" * 55)
    print("6. QLoRA 显存预估（Qwen2.5-7B-Instruct）")
    print("=" * 55)
    print("""
  组件                  预估显存      说明
  ─────────────────────────────────────────────────
  基座模型 (4-bit)      ~4-5 GB       NF4 量化后
  LoRA 适配器           ~0.1 GB       rank=64 时
  优化器 (paged_8bit)   ~2-4 GB       只优化 LoRA 参数
  激活值 + 梯度         ~6-12 GB      取决于 seq_len / batch
  ─────────────────────────────────────────────────
  合计（典型）          ~14-20 GB     5090D 32GB 充裕
  危险线                > 28 GB       需减 batch 或开 checkpointing
""")


def main():
    print("\n🔍 RTX 5090D 微调环境检查\n")

    check_python_version()

    cuda_ok = check_pytorch_and_cuda()
    if cuda_ok:
        check_gpu_details()
        kernel_ok = check_sm120_kernel()
    else:
        kernel_ok = False

    bnb_ok = check_bitsandbytes()
    estimate_qlora_vram()

    print("=" * 55)
    print("检查总结")
    print("=" * 55)
    all_ok = cuda_ok and kernel_ok and bnb_ok
    if all_ok:
        print("✅ 全部通过！可以进入 Step 2（数据准备）")
    else:
        print("❌ 存在问题，请根据上方提示修复后重新运行")
    print()


if __name__ == "__main__":
    main()
