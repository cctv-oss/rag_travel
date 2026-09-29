# 混合检索验证脚本:验证 向量+BM25+RRF+rerank 全链路
import os
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

print("=" * 50)
print("1. 初始化 VectorStoreService(加载rerank模型,可能首次下载)")
vs = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))
print("   语料文件:", config.raw_text_path, "| 存在:", os.path.exists(config.raw_text_path))

print("=" * 50)
print("2. 混合检索测试")
retriever = vs.get_retriever()

queries = [
    "上海外滩附近有什么好玩的景点?",
    "南京有什么旅游景点?",
]
for q in queries:
    print("-" * 50)
    print("问题:", q)
    docs = retriever.invoke(q)
    print("返回条数:", len(docs))
    for i, d in enumerate(docs):
        text = d.page_content.replace("\n", " ")[:70]
        print(f"  #{i+1} [{d.metadata.get('source')}] {text}")

print("=" * 50)
print("验证完成")
