# langgraph_demo.py
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from typing import TypedDict
from memory_adapter import ChromaMemoryAdapter

# LangChain agent & tools
from langchain.agents import AgentExecutor


# 1. 状態構造の定義
class AgentState(TypedDict):
    input: str
    output: str
    history: str


# 2. 各ノードの定義


# Load memory
def load_memory_node(state: AgentState, memory) -> AgentState:
    history = memory.load_memory_variables({"input": state["input"]})["history"]
    return {**state, "history": history}


# Run agent
def run_agent_node(state: AgentState, agent) -> AgentState:
    agent_input = f"{state['history']}\nUser: {state['input']}"
    response = agent.invoke({"input": agent_input})
    return {**state, "output": response["output"]}


# Save memory
def save_memory_node(state: AgentState, memory) -> AgentState:
    memory.save_context({"input": state["input"]}, {"output": state["output"]})
    return state


def build_graph(agent: AgentExecutor, memory: ChromaMemoryAdapter) -> CompiledGraph:
    """
    グラフを構築するヘルパー関数
    """
    # 3. LangGraph定義と構築
    builder = StateGraph(AgentState)
    builder.add_node("load_memory", partial(load_memory_node, memory=memory))
    builder.add_node("run_agent", partial(run_agent_node, agent=agent))
    builder.add_node("save_memory", partial(save_memory_node, memory=memory))

    # 順番に接続
    builder.set_entry_point("load_memory")
    builder.add_edge("load_memory", "run_agent")
    builder.add_edge("run_agent", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile()
