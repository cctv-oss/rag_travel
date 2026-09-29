# 智能旅行助手（Travel_Assistant）

基于 **RAG（检索增强生成）** 的旅行知识库问答系统。上传旅行资料后，系统自动切分、向量化并写入向量数据库；提问时通过「**向量 + BM25 混合检索 → RRF 融合 → CrossEncoder 精排**」召回相关知识片段，结合多轮对话历史，由大模型流式生成回答。

> 个人学习项目，用于展示 RAG 全链路与混合检索的实现。

## 技术栈

- **接口 / 界面**：FastAPI、Streamlit
- **编排**：LangChain（`RunnableWithMessageHistory`、链式调用）
- **向量库**：Chroma
- **向量化**：阿里 DashScope `text-embedding-v2`
- **大模型**：通义千问 `qwen-plus`
- **混合检索**：向量检索 + BM25（`rank_bm25` + `jieba` 中文分词）+ RRF 融合 + CrossEncoder（`bge-reranker-large`）精排

## 核心特性

- **混合检索**：向量检索（语义匹配）与 BM25（关键词匹配）双路召回，用 RRF 融合排序，再经 CrossEncoder 精排，兼顾语义与精确匹配、速度与精度
- **RAG 全链路**：文档加载 → 递归切分 → 向量化 → 入库 → 检索 → 生成
- **多轮对话**：按 `session_id` 隔离会话历史，本地持久化
- **流式响应**：`StreamingResponse` 逐字返回
- **MD5 去重**：相同内容不重复入库

## 项目结构

```
Travel_Assistant/
├── main.py               # FastAPI 接口（/upload、/chat）
├── app_qa.py             # Streamlit 聊天界面
├── app_file_uploader.py  # Streamlit 上传界面
├── knowledge_base.py     # 知识库：MD5 去重、切分、写入 Chroma
├── vector_stores.py      # 混合检索：向量 + BM25 + RRF + rerank
├── rag.py                # RAG 链：检索 → 拼 prompt → 生成
├── file_history_store.py # 会话历史本地存储
├── config_data.py        # 全部配置项
├── data/                 # 语料文件
├── chroma/               # 向量库本地存储
└── chat_history/         # 会话历史
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置阿里云 DashScope API Key
export DASHSCOPE_API_KEY=sk-xxx          # Linux / macOS
set DASHSCOPE_API_KEY=sk-xxx             # Windows

# 3. 启动 FastAPI 接口
uvicorn main:app --reload
```

> 首次运行会自动从 HuggingFace 镜像下载重排模型（`bge-reranker-large`，约 1.3GB）。

**界面方式**（二选一，与 FastAPI 共用同一套服务）：

```bash
streamlit run app_file_uploader.py   # 上传知识库
streamlit run app_qa.py              # 对话问答
```

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/docs` | GET | Swagger 文档，可视化测试 |
| `/upload` | POST | 上传 txt 文件入库（multipart，字段名 `file`） |
| `/chat` | POST | 对话（JSON：`{"message": "...", "session_id": "user_001"}`，流式返回） |

示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "上海外滩附近有什么好玩的景点？", "session_id": "user_001"}'
```

## 检索流程

```
用户提问
   │
   ├── 向量检索（Chroma）        → Top5 ─┐
   │                                     ├── RRF 融合去重排序 ── CrossEncoder 精排 ── Top3
   └── BM25 检索（jieba 分词）    → Top5 ─┘
                                          ↓
                              拼接历史对话 + 召回片段 → 通义千问 → 流式输出
```

## 配置说明

检索相关参数集中在 `config_data.py`：

| 参数 | 含义 |
|------|------|
| `chunk_size` / `chunk_overlap` | 文本切分长度 / 重叠 |
| `vector_k` / `bm25_k` | 向量 / BM25 各自召回条数 |
| `final_k` | 融合重排后最终条数 |
| `rrf_c` | RRF 平滑常数 |
| `rerank_model` | 重排模型 |
