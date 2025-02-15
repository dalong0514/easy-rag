import weaviate
from helper import get_api_key
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

api_key = get_api_key()
# base_url= "https://api.302.ai/v1"
# model_name = "deepseek-r1-aliyun"
base_url= "http://127.0.0.1:11434/v1"
model_name = "deepseek-r1:14b"

Settings.llm = OpenAI(
    api_base="https://api.302.ai/v1",
    api_key=api_key,
    model_name="deepseek-v3-aliyun"
)

model = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    streaming=True
)

system_template = "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. {context}  Question: {question} Helpful answer:"

def retrieval_from_documents(question, index_name, similarity_top_k):
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=index_name
        )

        vector_index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine = vector_index.as_query_engine(similarity_top_k=similarity_top_k)

        response = query_engine.query(question)
        context = "\n".join([n.text for n in response.source_nodes])
        source_datas = response.source_nodes

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{context}")]
        )
        prompt = prompt_template.invoke({"context": context, "question": question})

        response = model.stream(prompt)
        for chunk in response:
            print(chunk.content, end='', flush=True)
        # print(response.content)

        print_data_sources(source_datas)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("Weaviate connection closed.")


def retrieval_from_documents_llamaindex(prompt, index_name, similarity_top_k):
    try:
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
        # print(len(response.source_nodes))
        for n in response.source_nodes:
            print(n.metadata)
            print(n.text)
            print("----------------------------------------------------------------------------------------")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("Weaviate connection closed.")

def print_data_sources(source_datas):
    print("\n\nsource_datas----------------------------------------------------------------source_datas")
    for n in source_datas:
        print(n.metadata)
        print(n.text)
        print("----------------------------------------------------------------------------------------")