import os, time
from retrieval import basic_query_from_documents, automerging_query_from_documents, sentence_window_query_from_documents
from indexing import build_basic_fixed_size_index, get_all_files_from_directory, delete_document_collection
from indexing import build_automerging_index, build_sentence_window_index

def agent_rag():
    input_files = get_all_files_from_directory("/Users/Daglas/Desktop/test-documents")
    index_name = "Yangzhiping"
    prompt = "大模型领域有哪些机会，特别是 DeepSeek 生态相关的。详细阐述，且用简体中文问答。"
    similarity_top_k = 12

    # build_basic_fixed_size_index(input_files, index_name)
    basic_query_from_documents(prompt, index_name, similarity_top_k)

def basic_rag():
    input_files=["/Users/Daglas/dalong.knowledgetext/2024002原文书籍/2023055How-to-Build-a-Career-in-AI.pdf"]
    index_name = "TestDocument"
    prompt = "What are steps to take when finding projects to build your experience?"
    similarity_top_k = 12

    # build_basic_fixed_size_index(input_files, index_name)
    # basic_query_from_documents(prompt, index_name, similarity_top_k)
    # build_automerging_index(input_files, index_name)
    # automerging_query_from_documents(prompt, index_name, similarity_top_k)
    # build_sentence_window_index(input_files, index_name)
    sentence_window_query_from_documents(prompt, index_name, 18, 6)

if __name__ == "__main__":
    start_time = time.time()
    print('waiting...\n')
    basic_rag()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        print(f'Time Used: {elapsed_time:.2f} seconds')
    else:
        print(f'Time Used: {elapsed_time/60:.2f} minutes')