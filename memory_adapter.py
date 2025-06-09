from langchain_core.memory import BaseMemory
from typing import Dict, List, Any
from pydantic import Field, PrivateAttr
from base_memory_adapter import BaseMemoryAdapter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from datetime import datetime


class ChromaMemoryAdapter(BaseMemory):
    collection_name: str = "memory"
    session_id: str = "default"
    k: int = 5  # how many past messages to retrieve

    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _vectorstore: Chroma = PrivateAttr()
    _summary_store: Chroma = PrivateAttr()

    def __init__(self, session_id: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self._embeddings = OpenAIEmbeddings()
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
        )
        self._summary_store = Chroma(
            collection_name="summary_memory",
            embedding_function=self._embeddings,
        )

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

    @property
    def short_term_memory(self, k=3) -> List[str]:
        """Short-term memory: Latest n history (e.g. 3 messages)"""
        docs = self._vectorstore.similarity_search(
            "", k=k, filter={"session_id": self.session_id}
        )
        if not docs:
            return []
        docs_sorted = sorted(docs, key=lambda x: x.metadata.get("timestamp", 0))
        short_terms = []
        for doc in docs_sorted:
            lines = doc.page_content.split("\n")
            for line in lines:
                if line.strip():
                    short_terms.append(line)
        return short_terms[-k * 2 :]

    @property
    def long_term_memory(self) -> str:
        """Long-term memory: a latest summary of history in the summary store"""
        docs = self._summary_store.similarity_search(
            query="", k=100, filter={"session_id": self.session_id}
        )
        if not docs:
            return ""
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("timestamp", ""))
        return docs_sorted[-1].page_content

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", "")
        docs = self._vectorstore.similarity_search(
            query, k=self.k, filter={"session_id": self.session_id}
        )
        return {"history": "\n".join(doc.page_content for doc in docs)}

    def save_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any] = {}
    ) -> None:
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        if ai_output:
            content = f"User: {user_input}\nAI: {ai_output}"
        else:
            content = f"User: {user_input}"
        self._vectorstore.add_texts(
            texts=[content],
            metadatas=[
                {"session_id": self.session_id, "timestamp": datetime.now().isoformat()}
            ],
        )

    def maybe_generate_summary(self, agent=None):
        history = self.get_all_history()
        if len(history) >= 3:
            prompt = self._build_summary_prompt(history)
            if agent:
                response = agent.invoke({"input": prompt})
                summary = response["messages"][-1].content
            else:
                response = self.llm.invoke(prompt)
                summary = response.content if hasattr(response, "content") else response

            self._summary_store.add_texts(
                [summary],
                metadatas=[
                    {
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
            )

    def _build_summary_prompt(self, history: List[str]) -> str:
        return "以下の会話履歴を要約してください(ポイント形式で):\n" + "\n".join(
            history
        )

    def retrieve(self, session_id: str) -> List[str]:
        """Retrieve all history based on session_id"""
        docs = self._vectorstore.similarity_search(
            query="",
            k=100,
            filter={"session_id": session_id},
        )
        return [doc.page_content for doc in docs]

    def clear(self) -> None:
        self._vectorstore.delete_collection()
        # TODO: better way to clear the collection

    def get_all_history(self) -> List[str]:
        docs = self._vectorstore.similarity_search(
            "",
            k=1000,
            filter={"session_id": self.session_id},
        )
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("timestamp", 0))
        return [doc.page_content for doc in docs_sorted]


class InMemoryAdapter(BaseMemory, BaseMemoryAdapter):
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
