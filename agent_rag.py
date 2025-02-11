import os, time
from utils import get_router_query_engine

def agent_rag():
    query_engine = get_router_query_engine(["/Users/Daglas/Downloads/metagpt.pdf"])

    response = query_engine.query("What is the summary of the document?")
    print(str(response))
    print(len(response.source_nodes))

    response = query_engine.query("How do agents share information with other agents?")
    print(len(response.source_nodes))
    print(str(response))

    response = query_engine.query("Tell me about the ablation study results?")
    print(len(response.source_nodes))
    print(str(response))

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