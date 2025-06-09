import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langgraph_demo import (
    build_graph,
)
from memory_adapter import ChromaMemoryAdapter

selected_session_id = "demo-session"  # Default session ID for the Streamlit app


# dummy tokyo info tool for agent initialization
def dummy_tool(input_text: str) -> str:
    # English & Japanese keyword detection
    if "天気" in input_text or "weather" in input_text.lower():
        return "Tokyo's weather today is mostly sunny with a high of 27°C."
    if (
        "観光" in input_text
        or "観光地" in input_text
        or "tourist" in input_text.lower()
    ):
        return "Famous tourist spots in Tokyo include Asakusa, Tokyo Tower, and Shibuya Crossing."
    if (
        "良いところ" in input_text
        or "good point" in input_text.lower()
        or "best thing" in input_text.lower()
    ):
        return "Tokyo is safe, has great public transport, and offers amazing food."
    if "公園" in input_text or "park" in input_text.lower():
        return "Ueno Park, Yoyogi Park, and Shinjuku Gyoen are popular parks in Tokyo."
    if "summarize" in input_text.lower() or "要約" in input_text:
        return "This conversation is about Tokyo's weather, sightseeing, and its advantages."
    # Fallback
    return "I'm a demo tool! Try asking about Tokyo's weather, tourist attractions, or good points."


def main(graph_app=None, agent=None):
    """Main function to run the Streamlit app."""
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
            memory.save_context({"input": user_input})

    # Auto Summarize checkbox
    auto_summary = st.sidebar.checkbox("Auto Summarize", value=False)

    # Showing internal state for demo purposes
    # short-term memory display
    st.sidebar.subheader("Short-term Memory")

    for turn in memory.short_term_memory:
        st.sidebar.write(turn)

    # Long-term memory display
    st.sidebar.subheader("Long-term Memory")
    summary = memory.long_term_memory
    if summary.strip():
        st.sidebar.text_area("Summary", value=summary, height=100)
    else:
        st.sidebar.write("No summary yet.")

    st.write("## Conversation History")

    history = memory.get_all_history()
    full_history = "\n".join([f"{msg}" for msg in history])
    st.text_area("Conversation History (All)", value=full_history, height=300)

    st.write("## Agent Response")
    if user_input:
        response = graph_app.invoke({"input": user_input})
        output_text = response["output"]

        if auto_summary:
            memory.maybe_generate_summary(agent=agent)

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
    agent = create_react_agent(
        model=ChatOpenAI(temperature=0),
        tools=[
            Tool(
                name="TokyoInfoTool",
                func=dummy_tool,
                description="Answers questions about Tokyo, such as weather, sightseeing spots, parks, and good points. Use the user's full question as input.",
            )
        ],
        prompt="You are a helpful assistant.",
    )
    graph_app = build_graph(
        agent=agent,
        memory=ChromaMemoryAdapter(session_id=selected_session_id),
    )

    main(graph_app=graph_app, agent=agent)
