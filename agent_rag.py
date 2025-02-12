import os, time
from helper import get_api_key
from utils import get_router_query_engine, auto_retrieval_tool_call, get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

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


def agent_rag():
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