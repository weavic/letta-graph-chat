# memory_adapter.py
# memory_adapter.py
from langchain_core.memory import BaseMemory
from typing import Dict, List, Any
from pydantic import Field


class MemoryAdapter(BaseMemory):
    store: Dict[str, List[str]] = Field(default_factory=dict)

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

    def save(self, session_id: str, message: str):
        self.store.setdefault(session_id, []).append(message)

    def retrieve(self, session_id: str) -> List[str]:
        return self.store.get(session_id, [])

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        session_id = inputs.get("session_id", "default")
        return {"history": "\n".join(self.retrieve(session_id))}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        session_id = inputs.get("session_id", "default")
        self.save(session_id, f"User: {inputs.get('input', '')}")
        self.save(session_id, f"AI: {outputs.get('output', '')}")

    def clear(self) -> None:
        self.store.clear()


# # memory_adapter.py
# from typing import List, Dict


# class MemoryAdapter:
#     def __init__(self):
#         self.store: Dict[str, List[str]] = {}  # Simple in-memory store

#     def save(self, session_id: str, message: str):
#         if session_id not in self.store:
#             self.store[session_id] = []
#         self.store[session_id].append(message)

#     def retrieve(self, session_id: str) -> List[str]:
#         return self.store.get(session_id, [])

#     def clear(self, session_id: str):
#         self.store.pop(session_id, None)
