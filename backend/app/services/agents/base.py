from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    name: str
    description: str
    available_tools: list[str]

    @abstractmethod
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def validate_output(self, output: dict[str, Any]) -> dict[str, Any]:
        return output

    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": str(error),
            "agent": self.name,
            "context": {"repo_id": str(context.get("repo_id")) if context.get("repo_id") else None},
        }
