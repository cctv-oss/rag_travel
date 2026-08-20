
##app_file_uploader
supported_file_type_desc = "TXT"#支持文件类型列表的中文说明
supported_file_type = ['txt']#支持文件类型列表


##knowledge_base
#md5
md5_path = "./md5.text"#md5字符串文件路径

#Chroma
cllection_name = "RAG"#数据库表名
EmneddingModel = "text-embedding-v2"#数据库向量化模型
persist_directory = "./chroma"#数据库本地存储路径
collection_metadata = "l2"#评分方式["cosine"|"l2"]


#splitter
chunk_size = 500 #每段文本最大长度
chunk_overlap = 50 #连续文本间的字符重叠数
separators = ['\n\n','\n','.','!','?','。','，',',',' ',''] #自然段落划分符号(从左往右,优先级逐步降低)
min_split_char_number = 1000 #文本分割处理的最小长度


##vector_store
similarity_threshold = 3 #检索返回的匹配文档数

##rag
#vector_service
embedding_model_name = "text-embedding-v2"#rag向量化模型
#prompt_template
system_template ="""
以我提供的已知参考资料为主,简洁且专业的回答用户问题
参考资料:{context}
"""
user_template = "请回答用户提问{input}"
#chat model
chat_model = "qwen-plus"

##file history store
storage_path ="./chat_history"

##app qa
#session config
session_config = {
    "configurable":{
        "session_id":"user_001",
    }
}
