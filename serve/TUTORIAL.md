# 用 Docker 服务生产文章 · 使用教程

微调好的写作模型已封装成 Docker 服务。本教程教你怎么用它批量产出公众号文章。

## 零、前提：确认服务在跑

```powershell
cd C:\Users\7\Desktop\gzhllm
docker compose ps                          # STATUS 应为 Up (healthy)
curl http://localhost:8010/health          # 返回 {"status":"ready",...}
```

若没在跑：`docker compose up -d`，首次加载模型约 2 分钟（看 `docker compose logs -f`）。

---

## 一、最快上手：命令行客户端（推荐）

`serve/write.py` 是傻瓜式客户端，一行命令出文章。

```powershell
# 激活环境（客户端只用到标准库，用系统 python 也行）
$env:PYTHONUTF8="1"

# 最简：只给主题
python serve/write.py "为什么英伟达的护城河是CUDA而不是芯片"

# 存成文件
python serve/write.py "县城青年的三条出路" --save output/县城青年.md

# 完整控制
python serve/write.py "AI Agent 会取代 App 吗" `
    --audience "科技从业者" `
    --requirements "2500字，观点鲜明，至少1个反直觉判断，引用可验证数据" `
    --max 3072 --temp 0.8 --rep 1.15

# 流式输出（打字机效果，边生成边显示）
python serve/write.py "免费的代价" --stream
```

参数速查：

| 参数 | 含义 | 建议值 |
|---|---|---|
| `--audience` | 目标受众 | 按主题调整 |
| `--requirements` | 字数/风格/硬性要求 | 越具体越好 |
| `--max` | 最大生成长度(token) | 2000字文≈2600，长文3072 |
| `--temp` | 温度，越高越发散 | 0.7-0.9 |
| `--rep` | 重复惩罚，越大越不复读 | 1.15（复读就调到1.2-1.3） |
| `--system` | 换写作角色 | 见下文 |
| `--save` | 保存正文到文件 | 可选 |

---

## 二、直接调 HTTP 接口（接入你自己的程序）

### 一次性生成

```powershell
curl -X POST http://localhost:8010/generate `
  -H "Content-Type: application/json" `
  -d '{\"topic\":\"数据分析师如何转型AI工程\",\"requirements\":\"2000字，务实，有可操作建议\",\"max_new_tokens\":3072}'
```

返回 JSON：`{"text": "<thinking>...</thinking><article>...</article>", "article": "纯正文"}`
- `text`：含思考链的完整输出
- `article`：只要正文（发公众号用这个）

### Python 调用

```python
import requests
r = requests.post("http://localhost:8010/generate", json={
    "topic": "为什么大厂都在做AI硬件",
    "audience": "科技投资人",
    "requirements": "2500字，有产业分析，避免营销话术",
    "max_new_tokens": 3072,
    "temperature": 0.7,
    "repetition_penalty": 1.15,
})
print(r.json()["article"])
```

### 流式（SSE，做前端打字机效果）

```python
import requests
with requests.post("http://localhost:8010/generate/stream",
                   json={"topic": "AI泡沫论", "max_new_tokens": 2048}, stream=True) as r:
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            chunk = line[6:]
            if chunk == "[DONE]": break
            print(chunk, end="", flush=True)
```

---

## 三、换写作角色（不止科技评论）

默认角色是"科技领域深度评论员"。用 `system` / `--system` 可切换：

```powershell
python serve/write.py "如何看待年轻人整顿职场" `
  --system "你是一位关注社会议题的深度观察者，善于从个体现象洞察结构性问题，观点犀利有同理心。"
```

> 注意：模型是在"科技/商业深度评论"语料上微调的，写这类题材最擅长；偏离太远（如诗歌、菜谱）效果会下降。

---

## 四、批量生产（一次多篇）

新建 `topics.txt`，一行一个主题，然后：

```powershell
$env:PYTHONUTF8="1"
Get-Content topics.txt | ForEach-Object {
    $name = $_ -replace '[\\/:*?"<>|]', '_'
    python serve/write.py "$_" --save "output/articles/$name.md"
}
```

---

## 五、常见问题

**Q：生成结尾在复读/车轱辘话？**
调高 `--rep` 到 1.2~1.3。原理：重复惩罚对已生成过的 token 降概率，越大越不易陷入复读循环。

**Q：文章太短/没写完？**
`--max` 调大（上限 4096）。模型偶尔会提前收尾，可在 `--requirements` 里强调"不少于XX字"。

**Q：出现"作者：XX""封面来源"这类杂质？**
这是训练数据的残留噪声（已从 v1 的每篇1.33处降到0.11处，未完全归零）+ 7B 幻觉。发布前人工删一下即可，或等扩充数据后重训改善。

**Q：涉及数据/事实的内容可信吗？**
7B 模型会编造数字和出处（如假的融资额、假参考链接）。**凡是具体数据、人名、引用，务必人工核实**。模型的价值在于结构、论证框架和文笔，不在事实准确性。

**Q：GPU 要留给别的任务用？**
`docker compose down` 停服务释放约 6GB 显存；用完再 `docker compose up -d`。

**Q：改了 serve/app.py 怎么生效？**
`docker compose up -d --build` 重建（依赖有缓存，只重打代码层，约 15 秒 + 模型重新加载约 2 分钟）。
