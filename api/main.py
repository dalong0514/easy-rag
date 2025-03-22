import os, sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.indexing import build_basic_fixed_size_index, build_automerging_index, build_sentence_window_index, delete_document_collections
from src.retrieval import basic_query_from_documents, get_all_index_names
from src.utils import get_chat_file_name, get_all_files_from_directory, print_data_sources, get_timestamp, utils_remove_punctuation
from typing import Union, List, Optional
# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from helper import get_api_key, get_base_url, get_chat_record_dir
import requests
import json

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
brave_api_key = get_api_key("brave")
base_url= get_base_url("deepseek")
model_name = "deepseek-reasoner"
chat_record_dir= get_chat_record_dir()

search_answer_zh_template = \
    '''**每次回复必须以 \"\<think\>\n\" 开始。**
    \# 以下内容是基于用户发送的消息的搜索结果:
    {context}
    在我给你的搜索结果中，每个结果都是[indexpage X begin]...[indexpage X end]格式的，X代表每篇文章的数字索引。请在适当的情况下在句子末尾引用上下文。请按照引用编号[citation:X]的格式在答案中对应部分引用上下文。如果一句话源自多个上下文，请列出所有相关的引用编号，例如[citation:3][citation:5]，切记不要将引用集中在最后返回引用编号，而是在答案对应部分列出。
    在回答时，请注意以下几点：
    - 并非搜索结果的所有内容都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选。
    - 对于列举类的问题（如列举所有航班信息），尽量将答案控制在10个要点以内，并告诉用户可以查看搜索来源、获得完整信息。优先提供信息完整、最相关的列举项；如非必要，不要主动告诉用户搜索结果未提供的内容。
    - 对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，例如[citation:3][citation:5]，不能只在文章末尾引用。你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
    - 如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
    - 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
    - 你需要根据用户要求和回答内容选择合适、美观的回答格式，确保可读性强。
    - 你的回答应该综合多个相关搜索结果来回答，不能重复引用一个搜索结果。
    - 除非用户要求，否则你回答的语言需要和用户提问的语言保持一致。
    \# 用户消息为：
    {question}'''

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

class WebSearchRequest(BaseModel):
    question: str
    chat_record_dir: str = chat_record_dir

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
            
            context = "\n".join([f"[indexpage {i} begin]{n.text}[indexpage {i} end]" for i, n in enumerate(source_nodes)])
            
            # 在后台处理数据源信息，不阻塞响应流
            print_data = print_data_sources(source_nodes)
            print(f"Number of source nodes: {len(source_nodes)}")

            # 流式返回 LLM 的响应
            full_response = "<think>\n正在检索相关文档...\n</think>\n<think>\n文档检索完成，正在生成回答...\n</think>\n"
            prompt_template = ChatPromptTemplate([
                ("user", search_answer_zh_template)
            ])
            
            prompt = prompt_template.invoke({"context": context, "question": request.question})
            response = model.stream(prompt)
            
            for chunk in response:
                yield chunk.content
                full_response += chunk.content
            
            # 添加引用信息到响应末尾
            citations = "<references>\n"
            for i, n in enumerate(source_nodes):
                citations += f"[{i}]: "
                if hasattr(n, 'metadata') and n.metadata:
                    # 将元数据格式化为易读的形式
                    metadata_dict = n.metadata
                    if isinstance(metadata_dict, dict):
                        metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata_dict.items() if k and v])
                    else:
                        metadata_str = str(n.metadata).replace('\n', ' ')
                    citations += f"Metadata: {metadata_str}\n"
                else:
                    citations += "\n"  # 确保即使没有元信息也添加一个换行
                citations += f"{n.text}\n\n"  # 在每个引用项之后添加额外的空行
            citations += "</references>"
            
            yield f"\n\n{citations}"
            full_response += f"\n\n{citations}"
            
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


@app.post("/web-search")
async def web_search_api(request: WebSearchRequest):
    try:
        file_name = f"{get_timestamp()}Web-{get_chat_file_name(request.question)}"
        chat_record_file = os.path.join(
            request.chat_record_dir,
            f"{file_name}.md"
        )
        
        async def generate():
            # 立即发送一个初始消息，告知用户正在处理
            yield "<think>\n正在使用Brave搜索引擎搜索...\n</think>\n"
            
            # 调用Brave搜索API
            brave_search_results = []
            try:
                clean_question = utils_remove_punctuation(request.question)
                # 这里需要替换为实际的Brave搜索API密钥和URL
                brave_api_url = "https://api.search.brave.com/res/v1/web/search"
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": brave_api_key  # 实际使用时需要替换为正确的API密钥
                }
                params = {
                    "q": clean_question,
                    "count": 10
                }
                
                search_response = requests.get(brave_api_url, headers=headers, params=params)
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    print(clean_question)
                    print(search_data)
                    
                    # 提取搜索结果
                    if 'web' in search_data and 'results' in search_data['web']:
                        brave_search_results = search_data['web']['results']
                else:
                    yield "<think>\n搜索API请求失败，状态码: {}\n</think>\n".format(search_response.status_code)
            except Exception as e:
                yield "<think>\n搜索过程中出错: {}\n</think>\n".format(str(e))
            
            # 构建搜索结果的上下文
            context = ""
            for i, result in enumerate(brave_search_results):
                context += f"[indexpage {i} begin]"
                context += f"标题: {result.get('title', '')}\n"
                context += f"链接: {result.get('url', '')}\n"
                context += f"描述: {result.get('description', '')}\n"
                context += f"[indexpage {i} end]\n\n"
            
            # 通知用户搜索完成
            yield "<think>\n搜索完成，正在生成回答...\n</think>\n"
            
            # 如果没有搜索结果，通知用户
            if not brave_search_results:
                context = "未找到相关搜索结果。"
                
            # 创建完整的显示结果
            print(f"Number of search results: {len(brave_search_results)}")
            
            # 流式返回 LLM 的响应
            full_response = "<think>\n正在使用Brave搜索引擎搜索...\n</think>\n<think>\n搜索完成，正在生成回答...\n</think>\n"
            prompt_template = ChatPromptTemplate([
                ("user", search_answer_zh_template)
            ])
            
            prompt = prompt_template.invoke({"context": context, "question": request.question})
            response = model.stream(prompt)
            
            for chunk in response:
                yield chunk.content
                full_response += chunk.content
            
            # 添加引用信息到响应末尾
            citations = "<references>\n"
            for i, result in enumerate(brave_search_results):
                citations += f"[{i}]: "
                citations += f"标题: {result.get('title', '')}\n"
                citations += f"链接: {result.get('url', '')}\n"
                citations += f"描述: {result.get('description', '')}\n\n"
            citations += "</references>"
            
            yield f"\n\n{citations}"
            full_response += f"\n\n{citations}"
            
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
