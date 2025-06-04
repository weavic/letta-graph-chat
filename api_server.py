# api_server.py (FastAPI version)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from memory_adapter import MemoryAdapter

app = FastAPI()
memory = MemoryAdapter()


class UserMessage(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat_endpoint(msg: UserMessage):
    # Store message
    memory.save(msg.session_id, msg.message)

    # Retrieve past context
    history = memory.retrieve(msg.session_id)

    # Simulate LLM (just echoing for now)
    response = f"Echo: {msg.message}"
    memory.save(msg.session_id, response)

    return {"session_id": msg.session_id, "history": history, "response": response}
