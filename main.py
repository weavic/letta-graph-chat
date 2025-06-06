import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI
from langgraph_demo import (
    build_graph,
)
from memory_adapter import ChromaMemoryAdapter

selected_session_id = "demo-session"  # Default session ID for the Streamlit app


# dummy tool for agent initialization
def dummy_tool(input_text: str) -> str:
    return f"(dummy tool response to '{input_text}')"


def main():
    st.title("🧠 Letta-style Memory Chat UI")

    selected_session_id = st.selectbox(
        "Select or create a session(as a user)",
        options=["demo-session", "user-a", "user-b"],
        index=0,
    )
    memory = ChromaMemoryAdapter(session_id=selected_session_id)
    user_input = st.text_input("Your message")

    if user_input:
        if "要約" not in user_input and "summarize" not in user_input.lower():
            memory.save(selected_session_id, user_input)

    st.write("## Conversation History")

    history = memory.retrieve(selected_session_id)
    unique_history = list(
        dict.fromkeys(history)
    )  # Remove duplicates while preserving order
    for msg in unique_history:
        st.write(f"History: {msg}")

    # Uncomment below to see
    # docs = memory._summary_store.similarity_search("東京", k=1)
    # for doc in docs:
    #     st.write(doc.page_content)

    st.write("## Agent Response")
    if user_input:
        response = graph_app.invoke({"input": user_input})
        output_text = response["output"]
        if "summarize" in user_input.lower() or "要約" in user_input:
            st.write("### Conversation Summary")
            st.write(output_text)
        else:
            st.write("### Agent Response")
            st.write(output_text)

        if "prompt" in response:
            with st.expander("Agent Prompt(LLM input)", expanded=True):
                st.markdown("```\n" + response["prompt"] + "\n```")


if __name__ == "__main__":
    agent = initialize_agent(
        tools=[Tool(name="DummyTool", func=dummy_tool, description="A dummy tool")],
        llm=ChatOpenAI(),
        agent_type="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
    )

    graph_app = build_graph(
        agent=agent,
        memory=ChromaMemoryAdapter(session_id=selected_session_id),
    )

    main()
