from langchain_core.memory import BaseMemory
from typing import Dict, List, Any
from pydantic import Field, PrivateAttr
from base_memory_adapter import BaseMemoryAdapter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


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

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", "")
        docs = self._vectorstore.similarity_search(
            query, k=self.k, filter={"session_id": self.session_id}
        )
        return {"history": "\n".join(doc.page_content for doc in docs)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        content = f"User: {user_input}\nAI: {ai_output}"
        self._vectorstore.add_texts(
            texts=[content], metadatas=[{"session_id": self.session_id}]
        )

    def save(self, session_id: str, message: str):
        """明示的に履歴だけを保存（ユーティリティ用途）"""
        self._vectorstore.add_texts([message], metadatas=[{"session_id": session_id}])

    def retrieve(self, session_id: str) -> List[str]:
        """session_id に基づく履歴の全取得"""
        docs = self._vectorstore.similarity_search(
            query="",  # 全件取得の代替として空検索を利用（今は仮）
            k=100,
            filter={"session_id": session_id},
        )
        return [doc.page_content for doc in docs]

    def clear(self) -> None:
        self._vectorstore.delete_collection()
        # TODO: セッションID単位で履歴を削除できるようにするとなお良い

    def get_all_history(self) -> str:
        docs = self._vectorstore.similarity_search("", k=1000)  # query なしで全部
        return "\n".join(doc.page_content for doc in docs)

    def summarize_session(self):
        """
        現在の session_id に紐づく履歴を集めて、まとめて１つの要約として保存
        """
        docs = self._vectorstore.similarity_search(
            query="summary",
            k=20,
            filter={"session_id": self.session_id},
        )
        full_text = "\n".join(doc.page_content for doc in docs)

        # TODO 仮実装。あとでLLMを使って要約するようにする
        summary = f"[SUMMARY SNAPSHOT]\n{full_text:300}..."
        print(f"Generated summary: {summary}")  # TODO : remove print in production

        # 古い履歴を削除（今は未実装。あとでTTL設計と統合
        self._vectorstore.add_texts(
            texts=[summary],
            metadatas=[{"session_id": self.session_id, "type": "summary"}],
        )


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
