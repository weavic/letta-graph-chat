# base_memory_adapter.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseMemoryAdapter(ABC):
    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """List of variable names this memory class will return."""
        pass

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch memory variables based on input keys."""
        pass

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context from inputs and outputs."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""
        pass
