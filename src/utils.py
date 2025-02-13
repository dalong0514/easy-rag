import os
import weaviate
from pathlib import Path
from helper import get_api_key
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.objects import ObjectIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.classes.config import Configure
from typing import List, Optional

api_key = get_api_key()

base_url = "https://api.302.ai/v1"
model_name = "deepseek-v3-aliyun"
Settings.llm = OpenAI(
    api_base=base_url,
    api_key=api_key,
    model_name=model_name
)

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"]
# model_name = "models/gemini-2.0-flash-exp"
# Settings.llm = Gemini(
#     api_key=api_key,
#     model_name=model_name
# )

Settings.embed_model = resolve_embed_model("local:/Users/Daglas/dalong.modelsets/bge-m3")

# -------------------------------
# 初始化本地 Weaviate 客户端和向量存储对象
# -------------------------------
weaviate_client = weaviate.connect_to_local()
if weaviate_client.is_ready():
    print("Local Weaviate is ready.")
    # 创建本地向量存储对象，使用 bge-m3 模型进行向量化
    vector_store = WeaviateVectorStore(
        client=weaviate_client,
        index_name="RagVectorIndex",
        text2vec_config=Configure.Vectorizer.text2vec_ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="bge-m3:latest"
        )
    )
else:
    print("Local Weaviate is not ready.")
    exit(1)

def get_vector_nodes():
    # 加载文档
    documents = SimpleDirectoryReader(input_files=["/Users/Daglas/Downloads/metagpt.pdf"]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def get_router_query_engine():
    nodes = get_vector_nodes()
    summary_index = SummaryIndex(nodes)
    # 修改处：传入 vector_store 参数，实现将向量数据存入本地数据库
    vector_index = VectorStoreIndex(nodes, vector_store=vector_store)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to MetaGPT",
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context from the MetaGPT paper.",
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True
    )
    return query_engine

def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int:
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)

def tool_call():
    add_tool = FunctionTool.from_defaults(fn=add)
    mystery_tool = FunctionTool.from_defaults(fn=mystery)
    response = Settings.llm.predict_and_call(
        [add_tool, mystery_tool],
        "Tell me the output of the mystery function on 2 and 9",
        verbose=True
    )
    print(str(response))

def vector_query(query: str, page_numbers: List[str]) -> str:
    """
    在索引上执行向量搜索。
    
    Args:
        query (str): 要嵌入的查询字符串。
        page_numbers (List[str]): 指定页码过滤。如果为空，则在所有页面中搜索。
    """
    metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
    nodes = get_vector_nodes()
    # 修改处：传入 vector_store 参数
    vector_index = VectorStoreIndex(nodes, vector_store=vector_store)
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
    )
    response = query_engine.query(query)
    return response

def get_summary_tool():
    nodes = get_vector_nodes()
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to MetaGPT",
    )
    return summary_tool

def auto_retrieval_tool_call():
    vector_query_tool = FunctionTool.from_defaults(name="vector_tool", fn=vector_query)
    summary_tool = get_summary_tool()
    prompt = "What is a summary of the paper?"
    response = Settings.llm.predict_and_call(
        [vector_query_tool, summary_tool],
        prompt,
        verbose=True
    )
    for n in response.source_nodes:
        print(n.metadata)

def get_doc_tools(file_path: str, name: str) -> str:
    """
    获取文档对应的向量查询和摘要查询工具。
    """
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    # 修改处：传入 vector_store 参数
    vector_index = VectorStoreIndex(nodes, vector_store=vector_store)
    
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """
        针对 MetaGPT 文档回答问题的向量查询函数。
        """
        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
        )
        response = query_engine.query(query)
        return response
        
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of MetaGPT. "
            "Do NOT use if you have specific questions over MetaGPT."
        ),
    )
    return vector_query_tool, summary_tool

def router_query_engine():
    query_engine = get_router_query_engine()
    response = query_engine.query("What is the summary of the document?")
    print(str(response))
    print(len(response.source_nodes))
    response = query_engine.query("How do agents share information with other agents?")
    print(len(response.source_nodes))
    print(str(response))
    response = query_engine.query("Tell me about the ablation study results?")
    print(len(response.source_nodes))
    print(str(response))

def agent_reasoning_loop():
    vector_tool, summary_tool = get_doc_tools("/Users/Daglas/Downloads/metagpt.pdf", "metagpt")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool],
        llm=Settings.llm,
        verbose=True
    )
    agent = AgentRunner(agent_worker)
    response = agent.query(
        "Tell me about the agent roles in MetaGPT, and then how they communicate with each other."
    )
    print(response.source_nodes[0].get_content(metadata_mode="all"))
    response = agent.chat("Tell me about the evaluation datasets used.")
    print(response.source_nodes[0].get_content(metadata_mode="all"))
    response = agent.chat("Tell me the results over one of the above datasets.")
    print(response.source_nodes[0].get_content(metadata_mode="all"))

def agent_reasoning_loop_step():
    vector_tool, summary_tool = get_doc_tools("/Users/Daglas/Downloads/metagpt.pdf", "metagpt")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool],
        llm=Settings.llm,
        verbose=True
    )
    agent = AgentRunner(agent_worker)
    task = agent.create_task(
        "Tell me about the agent roles in MetaGPT, and then how they communicate with each other."
    )
    step_output = agent.run_step(task.task_id)
    completed_steps = agent.get_completed_steps(task.task_id)
    print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
    print(completed_steps[0].output.sources[0].raw_output)
    upcoming_steps = agent.get_upcoming_steps(task.task_id)
    print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
    upcoming_steps[0]
    step_output = agent.run_step(task.task_id, input="What about how agents share information?")
    step_output = agent.run_step(task.task_id)
    print(step_output.is_last)
    response = agent.finalize_response(task.task_id)
    print(str(response))

def multi_document_agent_rag():
    papers = [
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/metagpt.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/longlora.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/selfrag.pdf",
    ]
    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]
    initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    agent_worker = FunctionCallingAgentWorker.from_tools(
        initial_tools,
        llm=Settings.llm,
        verbose=True
    )
    agent = AgentRunner(agent_worker)
    response = agent.query(
        "Tell me about the evaluation dataset used in LongLoRA, and then tell me about the evaluation results"
    )
    response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
    print(str(response))

def multi_document_agent_rag_rank():
    papers = [
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/metagpt.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/longlora.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/selfrag.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/knowledge_card.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/loftq.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/swebench.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/zipformer.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/values.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/finetune_fair_diffusion.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/metra.pdf",
        # "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/vr_mcl.pdf"
    ]
    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]
    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    # 定义一个“object”索引，并在其上构造检索器，注意传入构造 VectorStoreIndex 时的 vector_store 参数
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=lambda nodes: VectorStoreIndex(nodes, vector_store=vector_store),
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    tools = obj_retriever.retrieve("Tell me about the eval dataset used in MetaGPT and SWE-Bench")
    print(tools[2].metadata)
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=Settings.llm,
        system_prompt="""\
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
""",
        verbose=True
    )
    agent = AgentRunner(agent_worker)
    response = agent.query(
        "Tell me about the evaluation dataset used in MetaGPT and compare it against SWE-Bench"
    )
    print(str(response))
    response = agent.query(
        "Compare and contrast the LoRA papers (LongLoRA, LoftQ). Analyze the approach in each paper first."
    )

if __name__ == "__main__":
    # 示例调用
    router_query_engine()
    # agent_reasoning_loop()
    # multi_document_agent_rag_rank()
