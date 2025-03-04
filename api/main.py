import os, sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.indexing import build_basic_fixed_size_index, build_automerging_index, build_sentence_window_index, delete_document_collections
from src.retrieval import basic_query_from_documents, get_all_index_names
from src.utils import get_chat_file_name, get_all_files_from_directory, print_data_sources, get_timestamp
from typing import Union, List, Optional
# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from helper import get_api_key, get_base_url, get_chat_record_dir

app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

api_key = get_api_key("deepseek")
base_url= get_base_url("deepseek")
model_name = "deepseek-reasoner"
chat_record_dir= get_chat_record_dir()

model = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    temperature=0.6,
    streaming=True
)

class QueryRequest(BaseModel):
    question: str
    index_names: List[str]
    similarity_top_k: int = 12
    chat_record_dir: str = chat_record_dir

class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None  # 新增上下文字段
    chat_record_dir: str = chat_record_dir

class BuildIndexRequest(BaseModel):
    input_path: Union[List[str], str]  # 支持文件路径列表或目录路径
    index_name: str
    index_type: str = "basic"  # "basic", "automerging", or "sentence_window"
    file_extension: Optional[str] = None  # 用于目录扫描时的文件扩展名
    chunk_size: Optional[int] = 1024
    chunk_overlap: Optional[int] = 200
    chunk_sizes: Optional[List[int]] = None

class GetIndexNamesRequest(BaseModel):
    pass


class DeleteIndexRequest(BaseModel):
    index_names: List[str]

@app.post("/query")
async def query_from_documents_api(request: QueryRequest):
    try:
        file_name = f"{get_timestamp()}RAG-{get_chat_file_name(request.question)}"
        chat_record_file = os.path.join(
            request.chat_record_dir,
            f"{file_name}.md"
        )
        
        async def generate():
            # 立即发送一个初始消息，告知用户正在处理
            yield "<think>\n正在检索相关文档...\n</think>\n"
            
            # 异步调用文档检索函数
            source_nodes = basic_query_from_documents(
                question=request.question,
                index_names=request.index_names,
                similarity_top_k=request.similarity_top_k
            )

            # 通知用户文档检索完成
            yield "<think>\n文档检索完成，正在生成回答...\n</think>\n"
            
            context = "\n".join([n.text for n in source_nodes])
            
            # 在后台处理数据源信息，不阻塞响应流
            print_data = print_data_sources(source_nodes)
            print(f"Number of source nodes: {len(source_nodes)}")

            # 流式返回 LLM 的响应
            full_response = "<think>\n正在检索相关文档...\n</think>\n<think>\n文档检索完成，正在生成回答...\n</think>\n"
            prompt_template = ChatPromptTemplate([
                ("user", "**response with \"<think>\n\" at the beginning of every output.**\nUse the following pieces of context to answer the question at the end.\n{context}\nQuestion: {question}")
            ])
            prompt = prompt_template.invoke({"context": context, "question": request.question})
            response = model.stream(prompt)
            
            for chunk in response:
                yield chunk.content
                full_response += chunk.content
            
            # 异步写入文件，不阻塞响应流
            async def write_record_file():
                index_names_str = ', '.join(request.index_names)
                with open(chat_record_file, 'w', encoding='utf-8') as f:
                    f.write(f"{file_name}\n\n[question]:\n\n{request.question}\n\n[index_names]:\n\n{index_names_str}\n\n[answer]:\n\n{full_response}\n\n[source_datas]:\n\n{print_data}")
            
            # 在后台启动文件写入任务
            import asyncio
            asyncio.create_task(write_record_file())

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
            # 立即发送一个初始消息，告知用户正在处理
            yield "<think>\n正在处理您的请求...\n</think>\n"
            
            full_response = "<think>\n正在处理您的请求...\n</think>\n"
            prompt_template = ChatPromptTemplate([
                ("user", "**response with \"\<think\>\n\" at the beginning of every output**\nContext: {context}\nQuestion: {question}")
            ])
            prompt = prompt_template.invoke({
                "context": request.context or "",
                "question": request.question
            })
            response = model.stream(prompt)
            
            for chunk in response:
                yield chunk.content
                full_response += chunk.content
            
            # 异步写入文件，不阻塞响应流
            async def write_record_file():
                with open(chat_record_file, 'w', encoding='utf-8') as f:
                    f.write(f"{file_name}\n\n[question]:\n\n{request.question}\n\n[answer]:\n\n{full_response}")
            
            # 在后台启动文件写入任务
            import asyncio
            asyncio.create_task(write_record_file())

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


@app.post("/get-index-names")
async def get_index_names_api(request: GetIndexNamesRequest):
    try:
        index_names = get_all_index_names()
        return {
            "status": "success",
            "index_names": index_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-index")
async def delete_index_api(request: DeleteIndexRequest):
    try:
        # 调用删除函数
        delete_document_collections(request.index_names)
        
        return {
            "status": "success",
            "message": f"Indexes {request.index_names} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
