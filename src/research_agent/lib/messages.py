from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

from .tooling import ToolCall


class BaseMessage(BaseModel):
    role: str
    content: Optional[str] = ""

    def dict(self) -> Dict:
        """Return the message as a plain dict for the API payload."""
        return dict(self)


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AIMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[List[ToolCall]] = None
    token_usage: Optional[TokenUsage] = None
