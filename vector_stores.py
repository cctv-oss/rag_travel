from langchain_chroma import Chroma
import config_data as config
class VectorStoreService(object):
    def __init__(self,embedding):
        self.embeddings = embedding
        self.vector_store = Chroma(
            collection_name=config.cllection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        #返回向量检索器,方便加入chain
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})