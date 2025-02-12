import os, time
from utils import router_query_engine, auto_retrieval_tool_call, multi_document_agent_rag

def agent_rag():
    router_query_engine()

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