import weaviate
from helper import get_api_key
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore

api_key = get_api_key()
base_url= "https://api.302.ai/v1"
model_name = "deepseek-r1-aliyun"

Settings.llm = OpenAI(
    api_base=base_url,
    api_key=api_key,
    model_name=model_name
)

def retrieval_from_documents(prompt, index_name, similarity_top_k):
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()

    vector_store = WeaviateVectorStore(
        weaviate_client=client, 
        index_name=index_name
    )

    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = vector_index.as_query_engine(similarity_top_k=similarity_top_k)

    response = query_engine.query(prompt)
    print(str(response))
    for n in response.source_nodes:
        print(n.metadata)
        print(n.text)
        print("----------------------------------------------------------")

    client.close()  # Free up resources

def search_from_weaviate():
    # 连接本地 Weaviate
    client = weaviate.connect_to_local()

    vector_store = WeaviateVectorStore(
        weaviate_client=client, 
        index_name="AgentRAGDocument"
    )

    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = vector_index.as_query_engine(similarity_top_k=3)

    response = query_engine.query("How do agents share information with other agents?")
    print(str(response))
    for n in response.source_nodes:
        print(n.metadata)
    # print(len(response.source_nodes))

    client.close()  # Free up resources