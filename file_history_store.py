import os,json
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
import config_data as config



def get_history(session_id):
    return FileChatMessageHistory(session_id,storage_path=config.storage_path)
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id      # 会话id
        self.storage_path = storage_path  # 不同会话id的存放文件,文件夹的路径
            #完整文件路径
        self.file_path = os.path.join(self.storage_path,self.session_id)
            #确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self,messages: Sequence[BaseMessage]) -> None:
        #Sequence序列 类似list,tuple
        all_messages = list(self.messages)   #已有的消息列表
        all_messages.extend(messages)        #新的和已知的融合成一个list
    #将数据同步写入到本地文件中
    #类对象写入文件->一堆二进制
    #为了方便,可以将BaseMessage转换为字典(借助json字符串写入文件)
    #官方message_to_dict,单个消息对象(BaseMessage类实例)->字典
    #官方,messages_to_dict,多个消息对象(BaseMessage类实例)->)->字典
        new_messages = [message_to_dict(message)for message in all_messages]
        #将数据写入文件
        with open (self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)
    @property      #@property装饰器,将messages方法转换为成员属性
    def messages(self)->list[BaseMessage]:
        #当前文件内:list[字典]
        try:
            with open (self.file_path,"r",encoding="utf-8") as f:
                messages_date = json.load(f)
                return messages_from_dict(messages_date)
        except FileNotFoundError:
                return []
    def clear(self)->None:
        with open (self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)