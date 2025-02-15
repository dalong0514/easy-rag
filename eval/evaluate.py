import os, time
from retrieval import basic_query_from_documents
from indexing import build_basic_fixed_size_index, get_all_files_from_directory, delete_document_collection
from trulens_eval import Tru
from eval.utils_eval import get_prebuilt_trulens_recorder

def eval_rag():
    eval_questions = []
    with open('eval_questions.txt', 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            print(item)
            eval_questions.append(item)
    # You can try your own question:
    new_question = "What is the right AI job for me?"
    eval_questions.append(new_question)
    print(eval_questions)

    tru = Tru()
    tru.reset_database()
    # tru_recorder = get_prebuilt_trulens_recorder(query_engine,
    #                                             app_id="Direct Query Engine")

    # index_name = "TestDocument"
    # prompt = "What are steps to take when finding projects to build your experience?"
    # similarity_top_k = 3
    # basic_query_from_documents(prompt, index_name, similarity_top_k)

if __name__ == "__main__":
    start_time = time.time()
    print('waiting...\n')
    eval_rag()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        print(f'Time Used: {elapsed_time:.2f} seconds')
    else:
        print(f'Time Used: {elapsed_time/60:.2f} minutes')