import streamlit as st

# from memory_adapter import MemoryAdapter
from memory_adapter import InMemoryAdapter

# from memory_adapter.chroma import ChromaMemoryAdapter
# from memory_adapter.pinecone import PineconeMemoryAdapter
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI

memory = InMemoryAdapter()
# memory = ChromaMemoryAdapter()

st.title("🧠 Memory Chat UI")

session_id = st.selectbox(
    "Select or create a session", options=["demo-session", "user-a", "user-b"], index=0
)

user_input = st.text_input("Your message")

if user_input:
    memory.save(session_id, user_input)
    response = f"Echo: {user_input}"
    memory.save(session_id, response)

history = memory.retrieve(session_id)
st.write("## Conversation History")
for msg in history:
    st.write(msg)


# dummy tool for agent initialization
def dummy_tool(input_text: str) -> str:
    return f"(dummy tool response to '{input_text}')"


tools = [
    Tool(
        name="DummyTool",
        func=dummy_tool,
        description="A dummy tool that repeats the input.",
    )
]

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(),
    # memory=MemoryAdapter(),
    memory=memory,
    agent_type="chat-zero-shot-react-description",
)
