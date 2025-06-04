# main.py (Flask version)
# app.py (Streamlit version)
import streamlit as st
from memory_adapter import MemoryAdapter


print("Hello from memory-adaptor-demo!")

memory = MemoryAdapter()
session_id = "demo-session"

st.title("🧠 Simple Memory Chat UI")
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
from langchain.agents import Tool


def dummy_tool(input_text: str) -> str:
    return f"(dummy tool response to '{input_text}')"


# main.py (Flask version)
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from memory_adapter import MemoryAdapter

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
    memory=MemoryAdapter(),
    agent_type="chat-zero-shot-react-description",
)
