import weaviate, os
import google.generativeai as genai
from helper import get_api_key, get_api_key_google
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

api_key = get_api_key()
base_url= "https://api.302.ai/v1"
model_name = "deepseek-r1-huoshan"
reranker_model_name = "/Users/Daglas/dalong.modelsets/bge-reranker-v2-m3"

# Settings.llm = OpenAI(
#     api_base="https://api.302.ai/v1",
#     api_key=api_key,
#     model_name="deepseek-v3-huoshan"
# )

model = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    streaming=True
)

def chat_with_llm(question, context):
    full_response = ""
    system_template = "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. {context}  Question: {question} Helpful answer:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{context}")]
    )
    prompt = prompt_template.invoke({"context": context, "question": question})
    response = model.stream(prompt)
    for chunk in response:
        print(chunk.content, end='', flush=True)
        full_response += chunk.content
    return full_response

def chat_with_llm_pure(question):
    full_response = ""
    response = model.stream(question)
    for chunk in response:
        print(chunk.content, end='', flush=True)
        full_response += chunk.content
    return full_response

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
api_key_google = get_api_key_google()
genai.configure(api_key=api_key_google, transport="rest")
gemini_model = genai.GenerativeModel(
    model_name = "gemini-2.0-flash-thinking-exp-01-21",
)
def chat_with_gemini(question, context):
    system_template = f"You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. Question: {question} context: {context}  Helpful answer:"
    response = gemini_model.generate_content(
                contents=system_template, 
                stream=True)
    for chunk in response:
        print(chunk.text, end="", flush=True)
    # print(response.text)

def basic_query_from_documents(question, index_names, similarity_top_k, chat_record_file):
    try:
        client = weaviate.connect_to_local()
        
        if isinstance(index_names, str):
            index_names = [index_names]
            
        vector_indices = []
        for index_name in index_names:
            vector_store = WeaviateVectorStore(
                weaviate_client=client, 
                index_name=index_name
            )
            vector_index = VectorStoreIndex.from_vector_store(vector_store)
            vector_indices.append(vector_index)
        
        # 创建每个索引的检索器
        retrievers = [index.as_retriever(similarity_top_k=similarity_top_k) for index in vector_indices]
        
        # 自定义复合检索器
        # 修改后的 MultiIndexRetriever 实现
        class MultiIndexRetriever(BaseRetriever):
            def __init__(self, retrievers, similarity_top_k):
                super().__init__()
                self.retrievers = retrievers
                self.similarity_top_k = similarity_top_k  # 存储全局 top_k 值

            def _retrieve(self, query, **kwargs):
                all_nodes = []
                # 收集所有检索器的节点
                for retriever in self.retrievers:
                    retrieved_nodes = retriever.retrieve(query, **kwargs)
                    all_nodes.extend(retrieved_nodes)
                
                # 按相似度分数降序排序（分数越高越相关）
                sorted_nodes = sorted(all_nodes, key=lambda x: x.score, reverse=True)
                
                # 截取前 similarity_top_k 个节点
                return sorted_nodes[:self.similarity_top_k]

        # 创建复合检索器时传入 similarity_top_k 参数
        combined_retriever = MultiIndexRetriever(retrievers, similarity_top_k=similarity_top_k)  # 关键修改

        # 创建查询引擎时不需要再设置 similarity_top_k（已在检索器层处理）
        query_engine = RetrieverQueryEngine.from_args(combined_retriever)
        
        response = query_engine.query(question)
        context = "\n".join([n.text for n in response.source_nodes])
        source_datas = response.source_nodes
        
        print_data = print_data_sources(source_datas)
        print(f"Number of source nodes: {len(source_datas)}")
        
        chat_record = chat_with_llm(question, context)
        with open(chat_record_file, 'w', encoding='utf-8') as f:
            f.write(f"[question]:\n\n{question}\n\n[answer]:\n\n{chat_record}\n\n[source_datas]:\n\n{print_data}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()
            print("\nWeaviate connection closed.")


def basic_query_from_documents_for_one_collection(question, index_name, similarity_top_k):
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
        print_data_sources(source_datas)
        print(f"Number of source nodes: {len(source_datas)}")

        chat_with_llm(question, context)
        # chat_with_gemini(question, context)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("\nWeaviate connection closed.")


def sentence_window_query_from_documents(
    question,
    index_name,
    similarity_top_k=12,
    rerank_top_n=4,
):
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=index_name
        )

        sentence_index = VectorStoreIndex.from_vector_store(vector_store)

        # define postprocessors
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model=reranker_model_name
        )

        sentence_window_engine = sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
        )

        response = sentence_window_engine.query(question)

        context = "\n".join([n.text for n in response.source_nodes])
        source_datas = response.source_nodes

        print_data_sources(source_datas)

        chat_with_llm(question, context)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("\nWeaviate connection closed.")

# for auto-merging retriever
def automerging_query_from_documents(
    question,
    index_name,
    similarity_top_k=12,
    rerank_top_n=2,
):
    try:
        # 连接本地 Weaviate
        client = weaviate.connect_to_local()

        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=index_name
        )

        automerging_index = VectorStoreIndex.from_vector_store(vector_store)

        base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)

        retriever = AutoMergingRetriever(
            base_retriever, 
            automerging_index.storage_context, 
            verbose=True
        )

        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model=reranker_model_name
        )

        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, 
            node_postprocessors=[rerank]
        )

        response = auto_merging_engine.query(question)

        context = "\n".join([n.text for n in response.source_nodes])
        source_datas = response.source_nodes

        print_data_sources(source_datas)

        chat_with_llm(question, context)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()  # Ensure client is always closed, Free up resources
            print("\nWeaviate connection closed.")
        # query_engine = automerging_index.as_query_engine(similarity_top_k=similarity_top_k)
        # response = query_engine.query(question)
        # # print(response.source_nodes)
        # with open("/Users/Daglas/Downloads/output.json", 'a', encoding='utf-8') as file:
        #     file.write(str(response.source_nodes) + '\n\n')


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
            print("\nWeaviate connection closed.")

def print_data_sources(source_datas):
    full_content = ""
    print("\n\nsource_datas----------------------------------------------------------------source_datas")
    for n in source_datas:
        print(n.metadata)
        print(n.text)
        print("----------------------------------------------------------------------------------------")
        full_content += f"{n.metadata}\n\n{n.text}\n----------------------------------------------------------------------------------------\n"
    return full_content

if __name__ == "__main__":
    chat_with_llm_pure("中国唐朝最著名的四位诗人")