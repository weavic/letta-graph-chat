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


# dummy tool for agent initialization
def dummy_tool(input_text: str) -> str:
    return f"(dummy tool response to '{input_text}')"


def main():
    st.title("🧠 Memory Chat UI")

    session_id = st.selectbox(
        "Select or create a session",
        options=["demo-session", "user-a", "user-b"],
        index=0,
    )

    user_input = st.text_input("Your message")

    if user_input:
        memory.save(session_id, user_input)
        response = f"Echo: {user_input}"
        memory.save(session_id, response)

    history = memory.retrieve(session_id)
    st.write("## Conversation History")
    if st.button("💾 Summarize session history"):
        memory.summarize_session()
        st.success("Summary added to memory!")

    for msg in history:
        st.write(f"History: {msg}")

    if not user_input:
        response = graph_app.invoke({"input": user_input})
        st.write("## Agent Response")
        st.write(response["output"])
        print("Agent Response:", response["output"])
        # Save the context to memory


if __name__ == "__main__":
    # memory = InMemoryAdapter()
    memory = ChromaMemoryAdapter(session_id="user_001")

    graph_app = build_graph(
        agent=initialize_agent(
            tools=[Tool(name="DummyTool", func=dummy_tool, description="A dummy tool")],
            llm=ChatOpenAI(),
            agent_type="zero-shot-react-description",
            verbose=True,
        ),
        memory=memory,
    )

    main()
