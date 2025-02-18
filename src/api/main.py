import os, sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval import basic_query_from_documents, chat_with_llm_pure
from src.utils import get_chat_file_name
from typing import List
# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    index_names: List[str]
    similarity_top_k: int = 12
    chat_record_dir: str = "/Users/Daglas/dalong.github/dalong.chatrecord/chatrecord-origin/"

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        # Generate chat record file path
        chat_record_file = os.path.join(
            request.chat_record_dir,
            f"{get_chat_file_name(request.question)}.md"
        )
        
        # Call the basic query function
        basic_query_from_documents(
            question=request.question,
            index_names=request.index_names,
            similarity_top_k=request.similarity_top_k,
            chat_record_file=chat_record_file
        )
        
        # Read and return the chat record
        with open(chat_record_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "status": "success",
            "data": content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)