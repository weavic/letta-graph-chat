# langgraph_demo.py
from datetime import datetime
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from typing import TypedDict
from memory_adapter import ChromaMemoryAdapter

# LangChain agent & tools
from langchain.agents import AgentExecutor


# define the state of the agent
class AgentState(TypedDict):
    input: str
    output: str
    history: str
    summary: str
    decision: str
    prompt: str  # for poc, to see the LLM input


# definitions of the nodes in the graph


# Load summary
def summary_load_node(state: AgentState, memory) -> AgentState:
    summary_docs = memory._summary_store.similarity_search(
        "", k=1, filter={"session_id": memory.session_id}
    )
    summary = summary_docs[0].page_content if summary_docs else ""
    return {**state, "summary": summary}


# Load memory
def load_memory_node(state: AgentState, memory) -> AgentState:
    history = memory.load_memory_variables({"input": state["input"]})["history"]
    return {**state, "history": history}


# Decide next action based on input
# 「要約を含むか」で分岐するだけなので、今後「復習」なども追加しやすい設計
def decide_next(state: AgentState) -> str:
    if "要約" in state["input"] or "summarize" in state["input"].lower():
        return "summarize"
    return "answer"


# Decide next action as a node (must return a full AgentState + decision key)
def decide_node(state: AgentState) -> AgentState:
    decision = (
        "summarize"
        if "要約" in state["input"] or "summarize" in state["input"].lower()
        else "answer"
    )
    return {**state, "decision": decision}


# Summarize session
def summarize_session(state: AgentState, agent, memory) -> AgentState:
    prompt = f"以下の会話履歴を要約してください(ポイント形式で):\n{state['history']}"  # TODO: prompt message in Japanese
    response = agent.invoke({"input": prompt})
    summary = response["output"]

    memory._summary_store.add_texts(
        [summary],
        metadatas=[
            {"session_id": memory.session_id, "timestamp": datetime.now().isoformat()}
        ],
    )
    return {**state, "output": summary}


# Run agent
def run_agent_node(state: AgentState, agent) -> AgentState:
    agent_input = f"{state['summary']}\n{state['history']}"
    print("LLM Input:\n", agent_input)
    response = agent.invoke({"input": agent_input})
    return {**state, "output": response["output"], "prompt": agent_input}


# Save memory
def save_memory_node(state: AgentState, memory) -> AgentState:
    memory.save_context({"input": state["input"]}, {"output": state["output"]})
    return state


# Build the state graph for the agent
def build_graph(agent: AgentExecutor, memory: ChromaMemoryAdapter) -> CompiledGraph:
    builder = StateGraph(AgentState)
    builder.add_node("summary_load", partial(summary_load_node, memory=memory))
    builder.add_node("load_memory", partial(load_memory_node, memory=memory))
    builder.add_node("decide", decide_node)
    builder.add_node(
        "summarize_session", partial(summarize_session, agent=agent, memory=memory)
    )
    builder.add_node("run_agent", partial(run_agent_node, agent=agent))
    builder.add_node("save_memory", partial(save_memory_node, memory=memory))

    builder.set_entry_point("summary_load")
    builder.add_edge("summary_load", "load_memory")
    builder.add_edge("load_memory", "decide")
    builder.add_conditional_edges(
        "decide",
        lambda state: state["decision"],
        {
            "answer": "run_agent",
            "summarize": "summarize_session",
        },
    )
    builder.add_edge("run_agent", "save_memory")
    builder.add_edge("summarize_session", END)
    builder.add_edge("save_memory", END)

    return builder.compile()
