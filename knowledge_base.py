'''
知识库
'''
import os
from langchain_chroma import Chroma
import hashlib


from datetime import datetime
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
def check_md5(md5_str:str):#检查字符串是否已经被处理过(去重)
    if not os.path.exists(config.md5_path):#如果文件不存在,则创建一个空文件,返回False
        open (config.md5_path,'w',encoding='utf-8').close()
        return False
    else:#如果文件存在,则读取文件内容,判断是否包含md5
         for line in open(config.md5_path,'r',encoding='utf-8').readlines():
             line = line.strip()#处理字符串前后的空格和回车
             if line == md5_str:#如果包含,则返回True
                 return True #处理过
         return False




def save_md5(md5_str:str):#保存md5字符串到数据库
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')


def get_string_md5(input_str:str,encoding='utf-8'):#将传入的字符串转换为md5字符串
    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)
    # 创建md5对象
    md5_obj = hashlib.md5()  # 得到md5对象
    md5_obj.update(str_bytes)  # 更新内容(传入即将要转换的字节数组)
    md5_hex = md5_obj.hexdigest()  # 得到md5的十六进制字符串
    return md5_hex
class KnowledgeBaseService(object):
    def __init__(self):
        #文件不存在则创建,存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name= config.cllection_name,
            embedding_function= DashScopeEmbeddings(model = config.EmneddingModel),
            persist_directory= config.persist_directory,
            collection_metadata={"hnsw:space": config.collection_metadata}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,#长度统计的依据
        )#文本分割器对象(智能递归分割)

    def upload_by_str(self,data:str,filename): #传入字符串向量化,存入向量数据库中
        #先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return"[跳过]内容已经存在知识库中"
        if len(data)>config.min_split_char_number:
            knowledge_chunks:list[str]=self.splitter.split_text(data)
        else:
            knowledge_chunks:list[str]=[data]
        metadata = {
            "source": filename,#数据来源文件
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),#知识库加入的时间点
            "operator": "admin"#操作员

        }

        self.chroma.add_texts(         #将内容加载到向量库中
            knowledge_chunks,
            metadata=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)
        return"[成功]内容已加载到数据库中"

if __name__ == "__main__":
    service = KnowledgeBaseService()


