# 模型服务化（FastAPI + Docker）

把微调后的写作模型封装成可调用的 HTTP API。

## 目录

```
serve/
├── app.py          # FastAPI 服务（4bit 加载 + /generate + 流式）
├── test_api.py     # 冒烟测试脚本
├── Dockerfile      # CUDA 12.8 运行时镜像
└── README.md
```

## 前置：合并模型

服务加载的是合并后的完整模型（LoRA 已融进基座）：

```bash
llamafactory-cli export configs/merge_lora.yaml
# 产物：models/qwen25-7b-gzh-merged/（约 14.5GB）
```

## 方式一：本地直接运行

```bash
# 启动服务（4bit 加载约占 5-6GB 显存）
uvicorn serve.app:app --host 127.0.0.1 --port 8010

# 另开终端测试
python serve/test_api.py
```

> 注：Windows 上 8000 端口可能被占用，示例用 8010。

## 方式二：Docker 容器

前置：Docker Desktop 运行中 + NVIDIA 驱动 + nvidia-container-toolkit（Windows 走 WSL2 后端）。

```bash
docker compose up -d --build      # 构建并后台启动
docker compose logs -f            # 看加载日志
curl http://localhost:8010/health # 健康检查
docker compose down               # 停止
```

## API 说明

### `GET /health`
```json
{"status": "ready", "vram_gb": 5.18}
```

### `POST /generate` —— 一次性返回
请求：
```json
{
  "topic": "AI Agent 会取代 App 吗",
  "audience": "关注科技的互联网从业者",
  "requirements": "2000字左右，观点鲜明，有论据",
  "max_new_tokens": 3072,
  "temperature": 0.7
}
```
响应：
```json
{"text": "<thinking>...</thinking><article>...</article>", "article": "纯正文"}
```

### `POST /generate/stream` —— SSE 流式
返回 `text/event-stream`，逐块 `data: <片段>`，结束 `data: [DONE]`。适合前端打字机效果。

调用示例：
```bash
curl -N -X POST http://localhost:8010/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"topic":"AI 编程助手会让初级程序员消失吗","max_new_tokens":2048}'
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | `models/qwen25-7b-gzh-merged` | 合并模型路径 |
| `LOAD_4BIT` | `1` | 1=4bit量化(~6GB)，0=bf16(~15GB) |

## 关于量化的进阶说明

当前用 **bitsandbytes 运行时 4bit 量化**（加载即量化，稳定可用）。若追求更高推理速度，可选 **GPTQ/AWQ** 离线量化产出专用权重——但在 Windows/Blackwell 上 `auto-gptq` 编译较麻烦，建议在 Linux 环境做。生产部署也可考虑 **vLLM** 提升吞吐（PagedAttention + 连续批处理）。
