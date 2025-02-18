import os, time
import string
import weaviate
import json, requests
from pathlib import Path
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
from weaviate.classes.config import ConsistencyLevel
from llama_index.core import StorageContext
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery
from typing import List, Optional
from helper import get_api_key, get_api_key_weaviate, get_wcd_url_weaviate


api_key = get_api_key()
wcd_api_key = get_api_key_weaviate()
wcd_url = get_api_key_weaviate()

base_url= "https://api.302.ai/v1"
model_name = "deepseek-v3-aliyun"
Settings.llm = OpenAI(
    api_base=base_url,
    api_key=api_key,
    model_name=model_name
)
Settings.embed_model = resolve_embed_model("local:/Users/Daglas/dalong.modelsets/bge-m3")
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# model_name = "models/gemini-2.0-flash-exp"
# Settings.llm = Gemini(
#     api_key=api_key,
#     model_name=model_name
# )


def get_vector_nodes():
    # load documents
    documents = SimpleDirectoryReader(input_files=["../data/papers/metagpt.pdf"]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    return nodes

def get_router_query_engine():
    nodes = get_vector_nodes()

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )

    return query_engine


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


def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    
    nodes = get_vector_nodes()
    vector_index = VectorStoreIndex(nodes)
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
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
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )

    return summary_tool

def auto_retrieval_tool_call():
    vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
    )
    summary_tool = get_summary_tool()
    # prompt = "What are the high-level results of MetaGPT as described on page 2?"
    # prompt = "What are the MetaGPT comparisons with ChatDev described on page 8?"
    prompt = "What is a summary of the paper?"
    response = Settings.llm.predict_and_call(
        [vector_query_tool, summary_tool], 
        prompt, 
        verbose=True
    )
    for n in response.source_nodes:
        print(n.metadata)


def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the MetaGPT paper.
    
        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
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

def agent_reasoning_loop():
    vector_tool, summary_tool = get_doc_tools("/Users/Daglas/Downloads/metagpt.pdf", "metagpt")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool], 
        llm=Settings.llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)
    response = agent.query(
        "Tell me about the agent roles in MetaGPT, "
        "and then how they communicate with each other."
    )

    print(response.source_nodes[0].get_content(metadata_mode="all"))
    response = agent.chat(
    "Tell me about the evaluation datasets used."
    )
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
        "Tell me about the agent roles in MetaGPT, "
        "and then how they communicate with each other."
    )
    step_output = agent.run_step(task.task_id)

    completed_steps = agent.get_completed_steps(task.task_id)
    print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
    print(completed_steps[0].output.sources[0].raw_output)

    upcoming_steps = agent.get_upcoming_steps(task.task_id)
    print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
    upcoming_steps[0]

    step_output = agent.run_step(
        task.task_id, input="What about how agents share information?"
    )

    step_output = agent.run_step(task.task_id)
    print(step_output.is_last)

    response = agent.finalize_response(task.task_id)
    print(str(response))

def multi_document_agent_rag():
    papers = [
        "../data/papers/metagpt.pdf",
        "../data/papers/longlora.pdf",
        "../data/papers/selfrag.pdf",
    ]

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    len(initial_tools)

    agent_worker = FunctionCallingAgentWorker.from_tools(
        initial_tools, 
        llm=Settings.llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    response = agent.query(
        "Tell me about the evaluation dataset used in LongLoRA, "
        "and then tell me about the evaluation results"
    )

    response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
    print(str(response))


def multi_document_agent_rag_rank():
    papers = [
        "../data/papers/metagpt.pdf",
        "../data/papers/longlora.pdf",
        "../data/papers/selfrag.pdf",
        "../data/papers/knowledge_card.pdf",
        "../data/papers/loftq.pdf", 
        "../data/papers/swebench.pdf",
        "../data/papers/zipformer.pdf",
        "../data/papers/values.pdf",
        "../data/papers/finetune_fair_diffusion.pdf",
        "../data/papers/metra.pdf",
        # "../data/papers/vr_mcl.pdf"
    ]

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    # define an "object" index and retriever over these tools
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )

    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    tools = obj_retriever.retrieve(
        "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
    )
    tools[2].metadata

    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=Settings.llm, 
        system_prompt=""" \
    You are an agent designed to answer queries over a set of given papers.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    response = agent.query(
        "Tell me about the evaluation dataset used "
        "in MetaGPT and compare it against SWE-Bench"
    )
    print(str(response))

    response = agent.query(
        "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
        "Analyze the approach in each paper first. "
    )

def get_chat_file_name(input_str: str) -> str:
    """处理字符串：
    1) 剔除所有标点符号
    2) 截取前6个英文单词或中文字符
    3) 将空格替换为'-'
    
    Args:
        input_str (str): 输入字符串
        
    Returns:
        str: 处理后的字符串
    """
    
    # 1. 剔除标点符号（包括中文和英文）
    # 定义中文标点符号集合
    chinese_punctuation = '，。！？；：（）《》【】“”‘’、·…—'
    # 合并英文和中文标点符号
    all_punctuation = string.punctuation + chinese_punctuation
    # 创建翻译表并删除所有标点符号
    translator = str.maketrans('', '', all_punctuation)
    cleaned_str = input_str.translate(translator)
    
    # 2. 截取前6个单词/字符
    # 对于英文：按空格分割取前6个单词
    if any(char.isalpha() for char in cleaned_str):
        words = cleaned_str.split()[:5]
        truncated_str = ' '.join(words)
    # 对于中文：直接取前6个字符
    else:
        truncated_str = cleaned_str[:5]
    
    # 3. 将空格替换为'-'
    final_str = truncated_str.replace(' ', '-')

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    final_str = f"{timestamp}-{final_str}"
    
    return final_str

if __name__ == "__main__":
    result = get_chat_file_name("储罐隔热层相关计算时，什么是隔热层的折减系数Ri")
    print(result)