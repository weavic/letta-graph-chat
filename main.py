import streamlit as st

# from memory_adapter import MemoryAdapter
# from memory_adapter import InMemoryAdapter
from memory_adapter import ChromaMemoryAdapter

# from memory_adapter import PineconeMemoryAdapter

from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langgraph_demo import (
    build_graph,
)  # LangGraph application for exercise/feedback.
from datetime import datetime


# dummy tool for agent initialization
def dummy_tool(input_text: str) -> str:
    return f"(dummy tool response to '{input_text}')"


def summarize_session(memory: ChromaMemoryAdapter, agent):
    # 全履歴を取得
    history = memory.load_memory_variables({"input": ""})["history"]

    # 要約生成
    summary_prompt = f"以下の会話履歴を要約してください:\n{history}"
    response = agent.invoke({"input": summary_prompt})
    summary = response["output"]

    # 要約を summary_store に保存
    memory._summary_store.add_texts(
        [summary],
        metadatas=[
            {"session_id": memory.session_id, "timestamp": datetime.now().isoformat()}
        ],
    )

    st.success(f"✅ セッションの要約を保存しました.\n\n{summary}")


def main():
    st.title("🧠 Memory Chat UI")

    selected_session_id = st.selectbox(
        "Select or create a session",
        options=["demo-session", "user-a", "user-b"],
        index=0,
    )
    memory = ChromaMemoryAdapter(session_id=selected_session_id)
    # memory = InMemoryAdapter(session_id=selected_session_id)
    user_input = st.text_input("Your message")

    if user_input:
        memory.save(selected_session_id, user_input)
        response = f"Echo: {user_input}"
        memory.save(selected_session_id, response)

    history = memory.retrieve(selected_session_id)
    unique_history = list(
        dict.fromkeys(history)
    )  # Remove duplicates while preserving order

    st.write("## Conversation History")
    if st.button("💾 Summarize session history"):
        memory.summarize_session()
        st.success("Summary added to memory!")
    if st.button("📝 セッションを要約して保存"):
        summarize_session(memory, agent)

    for msg in unique_history:
        st.write(f"History: {msg}")

    docs = memory._summary_store.similarity_search("東京", k=1)
    for doc in docs:
        st.write(doc.page_content)

    if not user_input:
        response = graph_app.invoke({"input": user_input})
        st.write("## Agent Response")
        st.write(response["output"])
        print("Agent Response:", response["output"])
        # Save the context to memory


if __name__ == "__main__":
    # memory = InMemoryAdapter()
    agent = initialize_agent(
        tools=[Tool(name="DummyTool", func=dummy_tool, description="A dummy tool")],
        llm=ChatOpenAI(),
        agent_type="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
    )

    graph_app = build_graph(
        agent=agent,
        memory=ChromaMemoryAdapter(session_id="demo-session"),
    )

    main()
