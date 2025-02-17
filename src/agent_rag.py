import os, time
from retrieval import basic_query_from_documents, automerging_query_from_documents, sentence_window_query_from_documents
from indexing import build_basic_fixed_size_index, get_all_files_from_directory, delete_document_collection
from indexing import build_automerging_index, build_sentence_window_index

# index_name = "TestDocument"
# index_name = "ChemicalDesign"
# index_name = "Fangjun"
index_name = "Yangzhiping"
# index_name = "ITStudyLLM"

# prompt = "火灾危险类别为甲类的定义"
# prompt = "设计甲类车间时，车间的设备布置有哪些注意事项"
# prompt = "甲类厂房的楼梯设置有哪些原则，需要注意什么"
#-------------------------------------------------------------------------------------#
# prompt = "What are steps to take when finding projects to build your experience?"
# 大模型领域有哪些机会，特别是 DeepSeek 生态相关的。详细阐述，且用简体中文问答。
prompt = '''
如何采用`认知方式`的思路来阅读一本著作
'''

def build_index():
    # input_files = ["/Users/Daglas/dalong.knowledgetext/2024002原文书籍/2023055How-to-Build-a-Career-in-AI.pdf"]
    input_files = get_all_files_from_directory("/Users/Daglas/dalong.github/dalong.ITstudy/001大语言模型", "md")

    build_basic_fixed_size_index(input_files, index_name)
    # build_automerging_index(input_files, index_name)
    # build_sentence_window_index(input_files, index_name)

def query_from_documents():
    print(prompt)
    basic_query_from_documents(prompt, index_name, 14)
    # sentence_window_query_from_documents(prompt, index_name, 18, 6)
    # automerging_query_from_documents(prompt, index_name, similarity_top_k)

def agent_rag():
    # build_basic_fixed_size_index(input_files, index_name)
    basic_query_from_documents(prompt, index_name, 15)

if __name__ == "__main__":
    start_time = time.time()
    print('waiting...\n')
    # build_index()
    query_from_documents()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        print(f'Time Used: {elapsed_time:.2f} seconds')
    else:
        print(f'Time Used: {elapsed_time/60:.2f} minutes')