from helper import get_api_key
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

api_key = get_api_key()
base_url= "https://api.302.ai/v1"
local_base_url= "https://24d5-115-236-24-30.ngrok-free.app"
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-ada-002"
# embedding_model_name = "bge-m3:latest"
# model_name = "gemini-2.0-flash-exp"

def get_router_query_engine(input_files):
    # load documents
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

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