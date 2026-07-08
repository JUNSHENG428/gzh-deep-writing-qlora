# ============================================================
# 公众号深度写作模型 API 服务镜像
# 基础镜像：CUDA 12.8 运行时（Blackwell/sm120 必需）
# 模型不打进镜像，运行时通过 -v 挂载到 /models（见 docker-compose.yml）
# 构建：docker build -f serve/Dockerfile -t gzh-writer:latest .
# ============================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/models/qwen25-7b-gzh-merged \
    LOAD_4BIT=1

# Python 3.10（Ubuntu 22.04 自带）+ pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 先装 PyTorch（cu128），单独一层便于缓存
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu128

# 再装服务依赖（与 requirements.txt 对齐，仅推理所需）
RUN pip install \
        "transformers>=4.55,<=5.7" \
        "accelerate>=1.3,<=1.11" \
        "bitsandbytes>=0.45.3" \
        fastapi "uvicorn[standard]" pydantic

# 拷贝服务代码（模型走挂载，不 COPY）
COPY serve/ /app/serve/

EXPOSE 8000

# 健康检查：轮询 /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status==200 else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
