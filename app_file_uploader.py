
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService
from config_data import supported_file_type_desc,supported_file_type

st.title("知识库上传/更新服务")
uploader_file = st.file_uploader(
    f"请上传文件 支持文件类型:{supported_file_type_desc}",
    type=supported_file_type,
    accept_multiple_files=True,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None and len(uploader_file) > 0:
    for single_file in uploader_file:
        file_name = single_file.name
        file_type = single_file.type
        file_size = single_file.size / 1024

        st.subheader(f"文件名{file_name}")
        st.write(f"格式:{file_type}|大小:{file_size:.2f}KB")
#get_value->bytes->decode(utf-8)
        text = single_file.getvalue().decode("utf-8", errors="ignore")

        with st.spinner("载入知识库中..."): #代码执行过程中增加转圈动画
            time.sleep(1)  #转圈动画最少时间  /秒
            result = st.session_state["service"].upload_by_str(text,file_name)
            st.write(result)

