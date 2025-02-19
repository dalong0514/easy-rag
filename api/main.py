import os, sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.indexing import build_basic_fixed_size_index, build_automerging_index, build_sentence_window_index
from src.retrieval import basic_query_from_documents
from src.utils import get_chat_file_name, get_all_files_from_directory, print_data_sources, get_timestamp
from typing import Union, List, Optional
# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from helper import get_api_key, get_api_key_google, get_api_key_grok

app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


# api_key = get_api_key_grok()
# base_url= "https://api.x.ai/v1"
# model_name = "grok-2-1212"

# Settings.llm = OpenAI(
#     api_base="https://api.302.ai/v1",
#     api_key=api_key,
#     model_name="deepseek-v3-huoshan"
# )

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# api_key_google = get_api_key_google()
# genai.configure(api_key=api_key_google, transport="rest")
# gemini_model = genai.GenerativeModel(
#     model_name = "gemini-2.0-flash-thinking-exp-01-21",
# )
# def chat_with_gemini(question, context):
#     system_template = f"You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. Question: {question} context: {context}  Helpful answer:"
#     response = gemini_model.generate_content(
#                 contents=system_template, 
#                 stream=True)
#     for chunk in response:
#         print(chunk.text, end="", flush=True)
#     print(response.text)


api_key = get_api_key()
base_url= "https://api.302.ai/v1"
model_name = "deepseek-r1-huoshan"


model = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    streaming=True
)

def chat_with_llm(question, context):
    full_response = ""
    system_template = "You are a helpful AI assistant. Output in Simplified Chinese. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. {context}  Question: {question} Helpful answer:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{context}")]
    )
    prompt = prompt_template.invoke({"context": context, "question": question})
    response = model.stream(prompt)
    for chunk in response:
        print(chunk.content, end='', flush=True)
        full_response += chunk.content
    return full_response

def chat_with_llm_pure(question, chat_record_file=None):
    full_response = ""
    response = model.stream(question)
    for chunk in response:
        print(chunk.content, end='', flush=True)
        full_response += chunk.content
    if chat_record_file:
        with open(chat_record_file, 'w', encoding='utf-8') as f:
            f.write(f"[question]:\n\n{question}\n\n[answer]:\n\n{full_response}")
    return full_response


class QueryRequest(BaseModel):
    question: str
    index_names: List[str]
    similarity_top_k: int = 12
    chat_record_dir: str = "/Users/Daglas/dalong.github/dalong.chatrecord/chatrecord-origin/"

class ChatRequest(BaseModel):
    question: str
    chat_record_dir: str = "/Users/Daglas/dalong.github/dalong.chatrecord/chatrecord-origin/"

class BuildIndexRequest(BaseModel):
    input_path: Union[List[str], str]  # 支持文件路径列表或目录路径
    index_name: str
    index_type: str = "basic"  # "basic", "automerging", or "sentence_window"
    file_extension: Optional[str] = None  # 用于目录扫描时的文件扩展名
    chunk_size: Optional[int] = 1024
    chunk_overlap: Optional[int] = 200
    chunk_sizes: Optional[List[int]] = None

@app.post("/query")
async def query_from_documents_api(request: QueryRequest):
    try:
        file_name = f"{get_timestamp()}RAG-{get_chat_file_name(request.question)}"
        chat_record_file = os.path.join(
            request.chat_record_dir,
            f"{file_name}.md"
        )
        
        async def generate():
            # Call the basic query function
            source_nodes = basic_query_from_documents(
                question=request.question,
                index_names=request.index_names,
                similarity_top_k=request.similarity_top_k
            )

            context = "\n".join([n.text for n in source_nodes])
            
            print_data = print_data_sources(source_nodes)
            print(f"Number of source nodes: {len(source_nodes)}")

            # 流式返回 LLM 的响应
            full_response = ""
            system_template = "You are a helpful AI assistant..."
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", system_template), ("user", "{context}")]
            )
            prompt = prompt_template.invoke({"context": context, "question": request.question})
            response = model.stream(prompt)
            
            for chunk in response:
                yield chunk.content
                full_response += chunk.content
            
            # 最后写入文件
            with open(chat_record_file, 'w', encoding='utf-8') as f:
                f.write(f"{file_name}\n\n[question]:\n\n{request.question}\n\n[answer]:\n\n{full_response}\n\n[source_datas]:\n\n{print_data}")

        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_llm_api(request: ChatRequest):
    try:
        file_name = f"{get_timestamp()}Chat-{get_chat_file_name(request.question)}"
        chat_record_file = os.path.join(
            request.chat_record_dir,
            f"{file_name}.md"
        )

        async def generate():
            full_response = ""
            response = model.stream(request.question)
            
            for chunk in response:
                yield chunk.content
                full_response += chunk.content
            
            # 最后写入文件
            with open(chat_record_file, 'w', encoding='utf-8') as f:
                f.write(f"{file_name}\n\n[question]:\n\n{request.question}\n\n[answer]:\n\n{full_response}")

        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build-index")
async def build_index_api(request: BuildIndexRequest):
    try:
        # 处理输入路径
        if isinstance(request.input_path, str):
            # 如果是目录路径，使用 get_all_files_from_directory
            input_files = get_all_files_from_directory(
                request.input_path,
                file_extension=request.file_extension
            )
        else:
            # 如果是文件路径列表，直接使用
            input_files = request.input_path

        if not input_files:
            raise ValueError("No valid input files found")

        # 根据索引类型调用相应的构建函数
        if request.index_type == "basic":
            build_basic_fixed_size_index(
                input_files=input_files,
                index_name=request.index_name,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap
            )
        elif request.index_type == "automerging":
            build_automerging_index(
                input_files=input_files,
                index_name=request.index_name,
                chunk_sizes=request.chunk_sizes
            )
        elif request.index_type == "sentence_window":
            build_sentence_window_index(
                input_files=input_files,
                index_name=request.index_name
            )
        else:
            raise ValueError(f"Invalid index type: {request.index_type}")
        
        return {
            "status": "success",
            "message": f"Index '{request.index_name}' built successfully",
            "num_files": len(input_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)