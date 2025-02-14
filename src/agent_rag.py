import os, time
from utils import router_query_engine, auto_retrieval_tool_call, multi_document_agent_rag
from indexing import create_document_index, get_all_files_from_directory, delete_document_collection
from retrieval import retrieval_from_documents

def agent_rag():
    input_files = get_all_files_from_directory("/Users/Daglas/Desktop/test-documents")
    index_name = "Yangzhiping"
    prompt = "大模型领域有哪些机会，特别是 DeepSeek 生态相关的。详细阐述，且用简体中文问答。"
    similarity_top_k = 8

    # create_document_index(input_files, index_name)
    retrieval_from_documents(prompt, index_name, similarity_top_k)

if __name__ == "__main__":
    start_time = time.time()
    print('waiting...\n')
    agent_rag()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        print(f'Time Used: {elapsed_time:.2f} seconds')
    else:
        print(f'Time Used: {elapsed_time/60:.2f} minutes')