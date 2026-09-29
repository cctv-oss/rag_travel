from typing import Annotated
from fastapi.responses import StreamingResponse
from fastapi import FastAPI , UploadFile , File
from knowledge_base import KnowledgeBaseService
from pydantic import BaseModel
from rag import RagService

app = FastAPI()
rag = RagService()

@app.post("/upload")
async def upload(file: Annotated[UploadFile,File()]):
    text = (await file.read()).decode("utf-8", errors="ignore")
    kb = KnowledgeBaseService()
    result = kb.upload_by_str(text, file.filename)
    return {"result": result}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "user_001"


@app.post("/chat")
async def chat(body: ChatRequest):
    session_config = {"configurable": {"session_id": body.session_id}}
    def generate():
        for chunk in rag.chain.stream({"input": body.message},session_config):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")


