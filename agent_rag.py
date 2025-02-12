import os, time
from helper import get_api_key
from utils import get_router_query_engine, auto_retrieval_tool_call, get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from pathlib import Path

api_key = get_api_key()
base_url= "https://api.302.ai/v1"
model_name = "gpt-3.5-turbo"

llm = OpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name
)

def router_query_engine():
    query_engine = get_router_query_engine()

    response = query_engine.query("What is the summary of the document?")
    print(str(response))
    print(len(response.source_nodes))

    response = query_engine.query("How do agents share information with other agents?")
    print(len(response.source_nodes))
    print(str(response))

    response = query_engine.query("Tell me about the ablation study results?")
    print(len(response.source_nodes))
    print(str(response))

def agent_reasoning_loop():
    vector_tool, summary_tool = get_doc_tools("/Users/Daglas/Downloads/metagpt.pdf", "metagpt")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool], 
        llm=llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)
    response = agent.query(
        "Tell me about the agent roles in MetaGPT, "
        "and then how they communicate with each other."
    )

    print(response.source_nodes[0].get_content(metadata_mode="all"))
    response = agent.chat(
    "Tell me about the evaluation datasets used."
    )
    print(response.source_nodes[0].get_content(metadata_mode="all"))
    response = agent.chat("Tell me the results over one of the above datasets.")
    print(response.source_nodes[0].get_content(metadata_mode="all"))


def agent_reasoning_loop_step():
    vector_tool, summary_tool = get_doc_tools("/Users/Daglas/Downloads/metagpt.pdf", "metagpt")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool], 
        llm=llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    task = agent.create_task(
        "Tell me about the agent roles in MetaGPT, "
        "and then how they communicate with each other."
    )
    step_output = agent.run_step(task.task_id)

    completed_steps = agent.get_completed_steps(task.task_id)
    print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
    print(completed_steps[0].output.sources[0].raw_output)

    upcoming_steps = agent.get_upcoming_steps(task.task_id)
    print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
    upcoming_steps[0]

    step_output = agent.run_step(
        task.task_id, input="What about how agents share information?"
    )

    step_output = agent.run_step(task.task_id)
    print(step_output.is_last)

    response = agent.finalize_response(task.task_id)
    print(str(response))

def multi_document_agent_rag():
    papers = [
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/metagpt.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/onglora.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/selfrag.pdf",
    ]

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    len(initial_tools)

    agent_worker = FunctionCallingAgentWorker.from_tools(
        initial_tools, 
        llm=llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    response = agent.query(
        "Tell me about the evaluation dataset used in LongLoRA, "
        "and then tell me about the evaluation results"
    )

    response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
    print(str(response))


def agent_rag():
    papers = [
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/metagpt.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/longlora.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/selfrag.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/knowledge_card.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/loftq.pdf", 
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/swebench.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/zipformer.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/values.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/finetune_fair_diffusion.pdf",
        "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/metra.pdf",
        # "/Users/Daglas/dalong.knowledgevideo/DeepLearning-AI/2025006Building-and-Evaluating-Advanced-RAG/papers/vr_mcl.pdf"
    ]

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    # define an "object" index and retriever over these tools
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )

    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    tools = obj_retriever.retrieve(
        "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
    )
    tools[2].metadata

    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=llm, 
        system_prompt=""" \
    You are an agent designed to answer queries over a set of given papers.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    response = agent.query(
        "Tell me about the evaluation dataset used "
        "in MetaGPT and compare it against SWE-Bench"
    )
    print(str(response))

    response = agent.query(
        "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
        "Analyze the approach in each paper first. "
    )

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