import os # ⚠️必须放在所有import最前面!!hf镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder

import config_data as config
import jieba
class VectorStoreService(object):
    def __init__(self,embedding):
        self.embeddings = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_directory,
        )
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": config.vector_k})
        self._ensure_raw_text()#语料文件不存在时从Chroma一次性导出(兼容改造前已入库资料)
        self.reranker = CrossEncoder(config.rerank_model, max_length=512)#重排模型,启动时加载一次
        #BM25索引缓存:语料不变就不重建,避免每次提问都重新分词建索引
        self._bm25_retriever = None
        self._bm25_mtime = None
    def jieba_preprocess(self,text:str):
        return list(jieba.cut(text))
    def _ensure_raw_text(self):
        #兼容改造前已入库的资料:语料文件不存在时,从Chroma导出纯文本chunk(带来源)
        if not os.path.exists(config.raw_text_path):
            all_data = self.vector_store.get()
            with open(config.raw_text_path,"w",encoding="utf-8") as f:
                for chunk, meta in zip(all_data["documents"], all_data["metadatas"]):
                    f.write(json.dumps({"text": chunk, "source": (meta or {}).get("source")}, ensure_ascii=False)+"\n")

    def _load_chunks(self):
        #读取BM25语料文件(每行一个JSON对象{"text","source"})
        if not os.path.exists(config.raw_text_path):
            return []
        with open(config.raw_text_path,"r",encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_bm25_retriever(self):
        #用文件修改时间做缓存:语料没变直接复用,变了(上传新资料)才重建
        if not os.path.exists(config.raw_text_path):
            return None
        mtime = os.path.getmtime(config.raw_text_path)
        if self._bm25_retriever is not None and self._bm25_mtime == mtime:
            return self._bm25_retriever

        items = self._load_chunks()
        if not items:
            return None
        texts = [i["text"] for i in items]
        metadatas = [{"source": i.get("source")} for i in items]
        self._bm25_retriever = BM25Retriever.from_texts(
            texts,
            metadatas=metadatas,
            preprocess_func=self.jieba_preprocess,
            k=config.bm25_k,
        )
        self._bm25_mtime = mtime
        return self._bm25_retriever

    def rrf_fuse(self, vector_docs, bm25_docs, k=60):
        score_map = defaultdict(float)

        for rank, doc in enumerate(vector_docs, start=1):
            score_map[doc.page_content] += 1 / (k + rank)

        for rank, doc in enumerate(bm25_docs, start=1):
            score_map[doc.page_content] += 1 / (k + rank)
        seen_text = set()

        combined_docs = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen_text:
                seen_text.add(doc.page_content)
                combined_docs.append(doc)
        combined_docs.sort(
            key=lambda d: score_map.get(d.page_content, 0),
            reverse=True
        )

        return combined_docs

    def get_retriever(self):
        def retrieve(query: str):
            vector_docs = self.vector_retriever.invoke(query)
            bm25_retriever = self.get_bm25_retriever()
            if bm25_retriever is None:
                return vector_docs[:config.final_k]
            hybrid_docs = self.rrf_fuse(vector_docs, bm25_retriever.invoke(query), k=config.rrf_c)
            #rerank重排
            pairs = [[query, doc.page_content] for doc in hybrid_docs]
            scores = self.reranker.predict(pairs)
            sorted_docs = [d for _, d in sorted(zip(scores, hybrid_docs), key=lambda x: x[0], reverse=True)]
            return sorted_docs[:config.final_k]
        return RunnableLambda(retrieve)
