import os # ⚠️必须放在所有import最前面!!hf镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.tongyi import Tongyi
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",config.system_template),
                ("system","并且我提供用户的对话历史记录,如下:"),
                MessagesPlaceholder("history"),
                ("user",config.user_template)
            ]
        )

        self.chat_model = Tongyi(model = config.chat_model)

        self.chain = self._get_chain()

    def _get_chain(self):
        #获得最终执行链
        retriever = self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return"无相关资料"
            formatted_str = ""
            for doc in docs:
                formatted_str +=f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str



        def format_for_retriever(value:dict)->str:
            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | self.chat_model | StrOutputParser()
        )


        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain
