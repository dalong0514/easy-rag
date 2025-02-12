from helper import get_api_key
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List

api_key = get_api_key()
base_url= "https://api.302.ai/v1"
local_base_url= "https://24d5-115-236-24-30.ngrok-free.app"
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-ada-002"
# embedding_model_name = "bge-m3:latest"
# model_name = "gemini-2.0-flash-exp"

Settings.llm = OpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name
)
Settings.embed_model = OpenAIEmbedding(
    base_url=base_url,
    api_key=api_key,
    model_name=embedding_model_name
)

def get_vector_nodes():
    # load documents
    documents = SimpleDirectoryReader(input_files=["/Users/Daglas/Downloads/metagpt.pdf"]).load_data()

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
